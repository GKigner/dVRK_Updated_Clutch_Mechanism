#!/usr/bin/env python3
# Live LED-state detector.
#
# Reads the camera, reproduces the SAME fixed rotated ROI crop used to build
# the training data (project the LED calibration into the image using the live
# gripper pose, rotate so the LED axis is horizontal, crop a fixed 220x120
# window), runs that ROI through your trained CNN (best_model.pt), and
# publishes an Int32 on a ROS topic:  1 = LEDs ON, 0 = LEDs OFF.
#
# The crop geometry below is copied verbatim from the data-collection script so
# the live ROIs are pixel-identical to the ones the CNN was trained on. Do not
# change ROI_WIDTH_PX / ROI_HEIGHT_PX / CENTER_OFFSET unless you also retrain.
#
# >>> THREE THINGS YOU MUST CONFIRM MATCH YOUR TRAINING SCRIPT <<<
#   1. preprocess_roi():  color vs grayscale, input size, and normalization
#                         MUST be identical to your training transforms.
#   2. MODEL_OUTPUT:      single-logit (sigmoid) vs 2-class (softmax) head,
#                         and which class index means "ON" (ON_INDEX).
#   3. load_model():      whether best_model.pt is a full pickled nn.Module
#                         or a state_dict (then you must supply build_model()).

import os
import sys
import json
import threading
from collections import deque

import cv2 as cv
import numpy as np
import torch

import crtk
from std_msgs.msg import Int32

from MTMCamera.after_setup.live_scripts.common_right import *  # camera_matrix, distortion, Arm, ...

# =========================
# Configuration
# =========================
CALIBRATION_FILE      = "../common_data/led_calibration_right.json"
LINK3_WRT_CAMERA_FILE = "../common_data/link3_wrt_camera_right.txt"
MODEL_FILE            = "./best_model.pt"

ARM_NAME   = "MTML"
NODE_NAME  = "led_detector"
TOPIC_NAME = "led_state"          # published as <NODE_NAME>/led_state by crtk

EXPECTED_W = 640
EXPECTED_H = 480

# ROI sizing — must match the data-collection / fixed_roi pipeline exactly.
# CHECK DIMENSIONS!!!
ROI_WIDTH_PX  = 220               # long axis (parallel to LEDs)
ROI_HEIGHT_PX = 180               # short axis (perpendicular to LEDs)
PAD_VALUE     = 0
_HALF_W = ROI_WIDTH_PX  / 2.0
_HALF_H = ROI_HEIGHT_PX / 2.0

# Center fine-tune nudge (gripper frame, meters). The training data was
# generated with all zeros, so keep these at zero unless you retrain.
CENTER_OFFSET_GRIPPER = np.array([0.0, 0.0, 0.0], dtype=np.float64)

# ---- CNN preprocessing (MUST match training) ----
GRAYSCALE   = False               # True if you trained on 1-channel images
INPUT_SIZE  = None                # (W, H) to resize the ROI to, or None to keep 180x100
NORMALIZE   = False               # True to apply per-channel mean/std normalization
NORM_MEAN   = (0.485, 0.456, 0.406)
NORM_STD    = (0.229, 0.224, 0.225)

# ---- CNN output interpretation (MUST match training) ----
# "auto": infer from output shape. 1 value -> sigmoid; >1 -> softmax/argmax.
MODEL_OUTPUT = "auto"
ON_INDEX     = 1                  # for a 2-class head, which index means "ON".
                                  # torchvision ImageFolder sorts ['OFF','ON']
                                  # alphabetically, so OFF=0, ON=1 by default.
SIGMOID_THRESHOLD = 0.5           # for a single-logit head

# ---- Behaviour ----
SHOW_PREVIEW    = True            # show an annotated window with the prediction
SMOOTH_WINDOW   = 1               # >1 = majority vote over the last N predictions
PUBLISH_PROB    = False           # also publish the raw probability (debug)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Camera thread (unchanged from collection script)
# =========================
class CameraThread:
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


# =========================
# LED calibration + geometry (copied to match training crops exactly)
# =========================
def load_led_calibration(path=CALIBRATION_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Calibration file not found: {path}")
    with open(path) as f:
        calib = json.load(f)

    led_left  = np.asarray(calib["led_left"],  dtype=np.float64)
    led_right = np.asarray(calib["led_right"], dtype=np.float64)

    if "led_center" in calib:
        center = np.asarray(calib["led_center"], dtype=np.float64)
    else:
        center = (led_left + led_right) / 2.0
    center = center + CENTER_OFFSET_GRIPPER

    axis   = led_right - led_left
    length = float(np.linalg.norm(axis))
    if length < 1e-9:
        raise ValueError("LED endpoints coincident; bad calibration.")
    axis_unit = axis / length
    pt_axis   = center + 0.01 * axis_unit

    pts_3d = np.array([center, pt_axis], dtype=np.float64).reshape(-1, 1, 3)
    print(f"Loaded LED calibration from {path}  (separation {length*1000:.2f} mm)")
    return pts_3d


def project_center_and_axis(T_ct, pts_3d):
    rvec, _ = cv.Rodrigues(T_ct[:3, :3])
    tvec    = T_ct[:3, 3]
    pts, _  = cv.projectPoints(pts_3d, rvec, tvec, camera_matrix, distortion)
    pts     = pts.reshape(-1, 2)
    center    = pts[0]
    axis_vec  = pts[1] - pts[0]
    angle_deg = float(np.degrees(np.arctan2(axis_vec[1], axis_vec[0])))
    return center, angle_deg


def crop_rotated_window(frame, center, angle_deg):
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


# =========================
# CNN: load, preprocess, infer
# =========================
def build_model():
    """ONLY needed if best_model.pt is a state_dict (not a full model).

    Paste your exact training architecture here, e.g.:

        from torchvision.models import resnet18
        m = resnet18(num_classes=2)
        return m

    If best_model.pt is a full pickled nn.Module, this is never called.
    """
    raise NotImplementedError(
        "best_model.pt looks like a state_dict. Fill in build_model() with the "
        "architecture you trained, so the weights can be loaded into it."
    )


def load_model(path=MODEL_FILE):
    # weights_only=False is required to unpickle a full nn.Module on torch>=2.6.
    obj = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(obj, torch.nn.Module):
        model = obj
    else:
        # state_dict or checkpoint dict -> need the architecture
        state = obj
        if isinstance(obj, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in obj and isinstance(obj[key], dict):
                    state = obj[key]
                    break
        model = build_model()
        model.load_state_dict(state)
    model.to(DEVICE).eval()
    print(f"Loaded model from {path} on {DEVICE}")
    return model


def preprocess_roi(roi_bgr):
    """ROI (BGR, HxW from cv) -> normalized float tensor [1, C, H, W].

    >>> This MUST reproduce your training transforms exactly. <<<
    """
    img = roi_bgr
    if GRAYSCALE:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)        # HxW
    else:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)         # HxWx3 (torchvision uses RGB)

    if INPUT_SIZE is not None:
        img = cv.resize(img, INPUT_SIZE, interpolation=cv.INTER_LINEAR)  # (W, H)

    img = img.astype(np.float32) / 255.0

    if img.ndim == 2:
        img = img[:, :, None]                            # HxWx1

    t = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # CxHxW

    if NORMALIZE and not GRAYSCALE:
        mean = torch.tensor(NORM_MEAN).view(3, 1, 1)
        std  = torch.tensor(NORM_STD).view(3, 1, 1)
        t = (t - mean) / std

    return t.unsqueeze(0).to(DEVICE)                     # 1xCxHxW


@torch.no_grad()
def infer(model, roi_bgr):
    """Returns (state, prob) where state is 1 (ON) or 0 (OFF)."""
    out = model(preprocess_roi(roi_bgr))
    t = out.detach().float().cpu().reshape(-1)

    use_sigmoid = (MODEL_OUTPUT == "sigmoid") or (MODEL_OUTPUT == "auto" and t.numel() == 1)
    if use_sigmoid:
        prob = torch.sigmoid(t)[0].item()
        return int(prob >= SIGMOID_THRESHOLD), prob
    else:
        probs = torch.softmax(t, dim=0)
        on    = int(torch.argmax(t).item() == ON_INDEX)
        return on, float(probs[ON_INDEX].item())


# =========================
# Main loop
# =========================
def run_detector(cam_thread, arm, link3_wrt_camera, pts_3d, model, pub, prob_pub):
    votes = deque(maxlen=max(1, SMOOTH_WINDOW))
    if SHOW_PREVIEW:
        cv.namedWindow("led_detector", cv.WINDOW_NORMAL)

    print(f"Detecting LED state -> publishing Int32 on '{NODE_NAME}/{TOPIC_NAME}'. "
          "ESC in the preview window (or Ctrl-C) to quit.")

    while True:
        ok, frame = cam_thread.read()
        if not ok or frame is None:
            if SHOW_PREVIEW and (cv.waitKey(1) & 0xFF) == 27:
                break
            continue

        # live gripper-wrt-camera pose, identical to data collection
        T_ct = link3_wrt_camera @ arm.pose()
        center, angle_deg = project_center_and_axis(T_ct, pts_3d)
        roi = crop_rotated_window(frame, center, angle_deg)

        state, prob = infer(model, roi)

        votes.append(state)
        published = int(round(sum(votes) / len(votes))) if SMOOTH_WINDOW > 1 else state

        pub.publish(Int32(data=published))
        if prob_pub is not None:
            prob_pub.publish(Int32(data=int(round(prob * 100))))  # percent, debug

        if SHOW_PREVIEW:
            disp = frame.copy()
            rect = ((float(center[0]), float(center[1])),
                    (ROI_WIDTH_PX, ROI_HEIGHT_PX), angle_deg)
            box = cv.boxPoints(rect).astype(np.int32)
            color = (0, 255, 0) if published == 1 else (0, 0, 255)
            cv.polylines(disp, [box], True, color, 2)
            label = f"{'ON' if published == 1 else 'OFF'}  p(on)={prob:.2f}"
            cv.putText(disp, label, (10, 25), cv.FONT_HERSHEY_SIMPLEX,
                       0.7, color, 2, cv.LINE_AA)
            cv.imshow("led_detector", disp)
            if (cv.waitKey(1) & 0xFF) == 27:
                break


def main():
    crtk.ral.parse_argv(sys.argv[1:])
    ral = crtk.ral(NODE_NAME)

    camera     = setup_camera(index=0)
    cam_thread = CameraThread(camera)

    arm = Arm(ral, ARM_NAME)
    link3_wrt_camera = np.loadtxt(LINK3_WRT_CAMERA_FILE)
    pts_3d           = load_led_calibration()
    model            = load_model()

    pub      = ral.publisher(TOPIC_NAME, Int32, queue_size=10)
    prob_pub = ral.publisher(TOPIC_NAME + "_prob_pct", Int32, queue_size=10) if PUBLISH_PROB else None

    def run():
        run_detector(cam_thread, arm, link3_wrt_camera, pts_3d, model, pub, prob_pub)

    try:
        ral.spin_and_execute(run)
    finally:
        cam_thread.stop()
        camera.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()