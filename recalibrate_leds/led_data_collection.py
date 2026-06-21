#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import json
import shutil
import time
import threading
import crtk
import sys
from pathlib import Path


RIGHT = True
LEFT  = False

# Hyperparameter to select what side you are working with
# must be RIGHT or LEFT
side = RIGHT

if (side == RIGHT):
    from common_right import *
else:
    from common_left import *


# Everything lives inside this directory except for link3_wrt_camera_{side}.txt,
# which is pulled from ../live_scripts.
HERE = Path(__file__).resolve().parent
LIVE_SCRIPTS_DIR = HERE.parent / "live_scripts"


# If True, wipe ../led_calibration/led_data/frames and
# ../led_calibration/led_data/annotated_images at the start of every run
# so the new capture starts from a clean slate. Set to False to keep
# previously captured frames around.
COLLECT_DATA = True

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
    if (side == RIGHT):
        frames_dir = HERE / "led_data_right" / "frames"
        annotated_dir = HERE / "led_data_right" / "annotated_images"
    else:
        frames_dir = HERE / "led_data_left" / "frames"
        annotated_dir = HERE / "led_data_left" / "annotated_images"

    # If collecting fresh data, wipe whatever was in there from a previous run
    # so frame indices restart at 0 against a clean directory.
    if COLLECT_DATA:
        for d in (frames_dir, annotated_dir):
            if d.exists():
                shutil.rmtree(d)
                print(f"Cleared {d}")

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
    if (side == RIGHT):
        out_json = HERE / "led_data_right" / "led_data.json"
    else:
        out_json = HERE / "led_data_left" / "led_data.json"
    with open(out_json, "w") as f:
        json.dump(data, f, indent="    ")
    print(f"Saved {len(data)} samples to {out_json}")



def main():

    if (side == RIGHT):
        arm = Arm(ral, "MTMR")
        link3_wrt_camera = np.loadtxt(str(LIVE_SCRIPTS_DIR / "link3_wrt_camera_right.txt"))
    elif (side == LEFT):
        arm = Arm(ral, "MTML")
        link3_wrt_camera = np.loadtxt(str(LIVE_SCRIPTS_DIR / "link3_wrt_camera_left.txt"))
    else:
        raise SystemExit("side error - check that you formatted the side variable correctly.")

    argv = crtk.ral.parse_argv(sys.argv[1:])
    ral = crtk.ral("led_test_collection")

    camera = setup_camera(index=0)
    cam_thread = CameraThread(camera)

 
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