#!/usr/bin/env python3

import sys
from pathlib import Path

# Add the repo root (the directory containing MTMCamera/) to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import cv2 as cv
import numpy as np
import json
import time
import threading
import crtk

RIGHT = True
LEFT  = False

side = RIGHT

if side == LEFT:
    from MTMCamera.after_setup.common_data.common_left import *
elif side == RIGHT:
    from MTMCamera.after_setup.common_data.common_right import *
else:
    raise SystemExit("side variable incorrectly entered, please check code")

EXPECTED_W = 640
EXPECTED_H = 480


class CameraThread:
    def __init__(self, camera):
        self.camera = camera
        self.frame = None
        self.ok = False
        self.lock = threading.Lock()
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self.running:
            ok, frame = self.camera.read()
            with self.lock:
                self.ok = ok
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
    camera.set(cv.CAP_PROP_FRAME_WIDTH, EXPECTED_W)
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, EXPECTED_H)
    camera.set(cv.CAP_PROP_BUFFERSIZE, 1)

    ok, frame = camera.read()
    if not ok:
        raise RuntimeError("Could not read from camera")

    h, w = frame.shape[:2]
    if (w, h) != (EXPECTED_W, EXPECTED_H):
        raise RuntimeError(f"Frame is {w}x{h}, not {EXPECTED_W}x{EXPECTED_H}")

    return camera

def collect(cam_thread, arm, link3_wrt_camera):
    data = []
    last_capture = 0.0
    if (side == LEFT):
        frames_dir = Path("./led_data_left/frames")
        annotated_dir = Path("./led_data_left/annotated_images")
    else:
        frames_dir = Path("./led_data_right/frames")
        annotated_dir = Path("./led_data_right/annotated_images")
    frames_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    # pre-allocate
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    while True:
        ok, frame = cam_thread.read()
        if not ok or frame is None:
            continue
        pose = link3_wrt_camera @ arm.pose()
        rvec_new, tvec_new = homogeneous_to_opencv(pose)
        rvec[:] = rvec_new
        tvec[:] = tvec_new
        annotated = frame.copy()
        cv.drawFrameAxes(annotated, camera_matrix, distortion, rvec, tvec, 0.01)
        now = time.time()
        if (now - last_capture) > .5:
            last_capture = now
            idx = len(data)
            image_file = frames_dir / f"frame_{idx}.png"
            annotated_file = annotated_dir / f"annotated_{idx}.png"
            cv.imwrite(str(image_file), frame)
            cv.imwrite(str(annotated_file), annotated)
            json_image_file_name = f"frame_{idx}.png"
            json_annotated_file_name =  f"annotated_{idx}.png"
            data.append({
                "gripper_wrt_camera": pose.tolist(),
                "image_name": str(json_image_file_name),
                "annotated_image_name": str(json_annotated_file_name),
            })
            print(f"Captured {idx + 1}")
        cv.imshow("image", annotated)
        if cv.waitKey(1) & 0xFF == 27:
            break
    if (side == LEFT):
        with open("./led_data_left/led_data.json", "w") as f:
            json.dump(data, f, indent="    ")
        print(f"Saved {len(data)} samples to ./led_data_left/led_data.json")
    else:
        with open("./led_data_right/led_data.json", "w") as f:
            json.dump(data, f, indent="    ")
        print(f"Saved {len(data)} samples to ./led_data_right/led_data.json")


def main():
    if (side == RIGHT):
        side_str = "RIGHT"
    else:
        side_str = "LEFT"
    print(f"About to gather LED calibration data for side: {side_str}")
    resp = input("\nPress ENTER to continue, press escape then enter to cancel: ")
    if resp != "":
        raise SystemExit("Cancelled.")


    argv = crtk.ral.parse_argv(sys.argv[1:])
    ral = crtk.ral("led_test_collection")

    camera = setup_camera(index=0)
    cam_thread = CameraThread(camera)

    if (side == RIGHT):
        arm = Arm(ral, "MTMR")
        link3_wrt_camera = np.loadtxt("../common_data/link3_wrt_camera_right.txt")
    else:
        arm = Arm(ral, "MTML")
        link3_wrt_camera = np.loadtxt("../common_data/link3_wrt_camera_left.txt")

    def run():
        cv.namedWindow("image", cv.WINDOW_NORMAL)
        collect(cam_thread, arm, link3_wrt_camera)

    try:
        ral.spin_and_execute(run)
    finally:
        cam_thread.stop()
        camera.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()