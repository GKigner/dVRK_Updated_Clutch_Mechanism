#!/usr/bin/env python3
# Live annotated preview.
# Shows what the camera sees with the projected bounding box, ROI center,
# and individual LED predictions drawn on top, in real time. No saving.
#
#   ESC  quit
#   r    toggle the cropped-ROI window on/off
#
# This is the live-collector pipeline (led_data_collection) with the capture,
# numbering, and JSON-append logic removed -- it only previews.

import cv2 as cv
import numpy as np
import json
import os
import time
import threading
import crtk
import sys
from common import *  

CALIBRATION_FILE      = "./led_calibration.json"
LINK3_WRT_CAMERA_FILE = "./link3_wrt_camera.txt"

EXPECTED_W = 640
EXPECTED_H = 480

ROI_WIDTH_PX  = 220             # long axis (parallel to LEDs)
ROI_HEIGHT_PX = 120             # short axis (perpendicular to LEDs)
PAD_VALUE     = 0

_HALF_W = ROI_WIDTH_PX  / 2.0
_HALF_H = ROI_HEIGHT_PX / 2.0


RIGHT = True
LEFT  = False

# Hyperparameter to select what side you are working with
# must be RIGHT or LEFT
side = RIGHT

class CameraThread:
    """Background frame grabber so the main loop never stalls on USB I/O."""

    def __init__(self, camera):
        self.camera  = camera
        self.frame   = None
        self.ok      = False
        self.lock    = threading.Lock()
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self.running:
            ok, frame = self.camera.read()
            with self.lock:
                self.ok    = ok
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ok, self.frame.copy()

    def stop(self):
        self.running = False
        self._thread.join()


def setup_camera(index=0):
    camera = cv.VideoCapture(index, cv.CAP_V4L2)
    if not camera.isOpened():
        raise RuntimeError("Could not open camera")

    camera.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv.CAP_PROP_FRAME_WIDTH,  EXPECTED_W)
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, EXPECTED_H)
    camera.set(cv.CAP_PROP_BUFFERSIZE, 1)

    ok, frame = camera.read()
    if not ok:
        raise RuntimeError("Could not read from camera")
    h, w = frame.shape[:2]
    if (w, h) != (EXPECTED_W, EXPECTED_H):
        raise RuntimeError(
            f"Frame is {w}x{h}, not {EXPECTED_W}x{EXPECTED_H}. "
            "Calibration assumes 640x480."
        )
    return camera


def load_led_calibration(path=CALIBRATION_FILE):
    """Load the two LED positions (in gripper frame). The ROI center is always
    derived as the midpoint of the two LEDs -- the labeler only ever produces
    two points, so any other notion of center is not supported here."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Calibration file not found: {path}\n"
            "Run label_leds.py first to generate it."
        )
    with open(path) as f:
        calib = json.load(f)

    if "led_left" not in calib or "led_right" not in calib:
        raise ValueError(
            f"Calibration file {path} is missing 'led_left' and/or 'led_right'. "
            "Re-run label_leds.py to regenerate it."
        )

    led_left  = np.asarray(calib["led_left"],  dtype=np.float64)
    led_right = np.asarray(calib["led_right"], dtype=np.float64)

    center = (led_left + led_right) / 2.0

    axis   = led_right - led_left
    length = float(np.linalg.norm(axis))
    if length < 1e-9:
        raise ValueError("LED left and right positions are coincident; bad calibration.")
    axis_unit = axis / length

    pt_axis = center + 0.01 * axis_unit  # 1 cm along the axis -> image-space angle

    pts_3d             = np.array([center, pt_axis],
                                  dtype=np.float64).reshape(-1, 1, 3)
    individual_leds_3d = np.array([led_left, led_right],
                                  dtype=np.float64).reshape(-1, 1, 3)

    print(f"Loaded LED calibration from {path}")
    print(f"  separation: {length * 1000:.2f} mm")
    return pts_3d, individual_leds_3d


def project_center_and_axis(T_ct, pts_3d):
    rvec, _ = cv.Rodrigues(T_ct[:3, :3])
    tvec    = T_ct[:3, 3]
    pts, _  = cv.projectPoints(pts_3d, rvec, tvec, camera_matrix, distortion)
    pts     = pts.reshape(-1, 2)
    center    = pts[0]
    axis_vec  = pts[1] - pts[0]
    angle_deg = float(np.degrees(np.arctan2(axis_vec[1], axis_vec[0])))
    return center, angle_deg


def project_individual_leds(T_ct, individual_leds_3d):
    rvec, _ = cv.Rodrigues(T_ct[:3, :3])
    tvec    = T_ct[:3, 3]
    pts, _  = cv.projectPoints(individual_leds_3d, rvec, tvec, camera_matrix, distortion)
    return pts.reshape(-1, 2)


def crop_rotated_window(frame, center, angle_deg):
    """Rotate around the ROI center so the LED axis is horizontal, crop fixed size."""
    cx, cy = float(center[0]), float(center[1])
    M = cv.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    M[0, 2] += _HALF_W - cx
    M[1, 2] += _HALF_H - cy
    return cv.warpAffine(
        frame, M, (ROI_WIDTH_PX, ROI_HEIGHT_PX),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=PAD_VALUE,
    )


def annotate_frame(frame, center, angle_deg, led_pts):
    """Draw the crop rectangle, ROI center, and both individual LED predictions."""
    img = frame.copy()

    # Crop rectangle (red)
    rect = ((float(center[0]), float(center[1])),
            (ROI_WIDTH_PX, ROI_HEIGHT_PX),
            angle_deg)
    box = cv.boxPoints(rect).astype(np.int32)
    cv.polylines(img, [box], isClosed=True, color=(0, 0, 255), thickness=2)

    # ROI center (green)
    cv.circle(img,
              (int(round(center[0])), int(round(center[1]))),
              4, (0, 255, 0), -1)

    # Individual LEDs (cyan = left, magenta = right) with axis line
    left_uv  = tuple(np.round(led_pts[0]).astype(int))
    right_uv = tuple(np.round(led_pts[1]).astype(int))
    cv.line(img, left_uv, right_uv, (255, 255, 255), 1)
    cv.circle(img, left_uv,  6, (255, 255, 0), 2)
    cv.circle(img, right_uv, 6, (255, 0, 255), 2)

    return img


def process_frame(frame, T_ct, pts_3d, individual_leds_3d):
    center, angle_deg = project_center_and_axis(T_ct, pts_3d)
    led_pts           = project_individual_leds(T_ct, individual_leds_3d)
    roi               = crop_rotated_window(frame, center, angle_deg)
    annotated         = annotate_frame(frame, center, angle_deg, led_pts)
    return roi, annotated, center, angle_deg


def draw_hud(img, center, angle_deg, fps):
    """Top banner: live status + readouts so you can sanity-check the projection."""
    text = (f"LIVE  fps={fps:4.1f}  "
            f"center=({center[0]:6.1f},{center[1]:6.1f})  "
            f"angle={angle_deg:+6.1f}   (ESC quit, r=ROI)")
    cv.putText(img, text, (10, 25),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)


def preview(cam_thread, arm, link3_wrt_camera, pts_3d, individual_leds_3d):
    show_roi  = True
    last_t    = time.time()
    fps       = 0.0

    print("Live preview. ESC to quit, 'r' to toggle the ROI window.")

    while True:
        ok, frame = cam_thread.read()
        if not ok or frame is None:
            if cv.waitKey(1) & 0xFF == 27:
                break
            continue

        # gripper-wrt-camera pose at this instant
        T_ct = link3_wrt_camera @ arm.pose()

        roi, annotated, center, angle_deg = process_frame(
            frame, T_ct, pts_3d, individual_leds_3d
        )

        # smoothed fps
        now   = time.time()
        dt    = now - last_t
        last_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        draw_hud(annotated, center, angle_deg, fps)
        cv.imshow("live", annotated)

        if show_roi:
            cv.imshow("roi", roi)

        key = cv.waitKey(1) & 0xFF
        if key == 27:                 # ESC
            break
        elif key == ord("r"):
            show_roi = not show_roi
            if not show_roi:
                cv.destroyWindow("roi")


def main():
    if (side == RIGHT):
        side_str = "RIGHT"
    elif (side == LEFT):
        side_str = "LEFT"
    else:
        side_str = "Error with side variable naming"
        raise SystemExit(side_str)
    crtk.ral.parse_argv(sys.argv[1:])
    ral = crtk.ral("live_view")

    camera     = setup_camera(index=0)
    cam_thread = CameraThread(camera)

    if (side == RIGHT):
        arm = Arm(ral, "MTMR")
    else:
        arm = Arm(ral, "MTML")

    link3_wrt_camera           = np.loadtxt(LINK3_WRT_CAMERA_FILE)
    pts_3d, individual_leds_3d = load_led_calibration()

    def run():
        cv.namedWindow("live", cv.WINDOW_NORMAL)
        preview(cam_thread, arm, link3_wrt_camera, pts_3d, individual_leds_3d)

    try:
        ral.spin_and_execute(run)
    finally:
        cam_thread.stop()
        camera.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()