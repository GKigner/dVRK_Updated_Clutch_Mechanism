#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import time
import threading
import crtk
import sys
from common import *


RIGHT = True
LEFT  = False

# Hyperparameter to select what side you are working with
# must be RIGHT or LEFT
side = LEFT

EXPECTED_W = 640
EXPECTED_H = 480


class CameraThread:
    """Grabs frames in a background thread so the main loop never stalls on USB I/O."""

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
            return self.ok, self.frame

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
    camera.set(cv.CAP_PROP_BUFFERSIZE, 1)  # minimize latency

    reported_w = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
    reported_h = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(f"Requested: {EXPECTED_W}x{EXPECTED_H}  Got: {reported_w}x{reported_h}")

    ok, frame = camera.read()
    if not ok:
        raise RuntimeError("Could not read from camera")
    check_frame_size(frame)

    return camera


def check_frame_size(frame, expected_w=EXPECTED_W, expected_h=EXPECTED_H):
    h, w = frame.shape[:2]
    if (w, h) != (expected_w, expected_h):
        raise RuntimeError(
            f"Frame is {w}x{h}, not {expected_w}x{expected_h}. "
            "Do not use this with the 640x480 calibration."
        )


def calibrate(target_poses, robot_poses):
    robot_poses_r = np.array(
        [homogeneous_to_opencv(np.linalg.inv(p))[0] for p in robot_poses],
        dtype=np.float64,
    )
    robot_poses_t = np.array(
        [homogeneous_to_opencv(np.linalg.inv(p))[1] for p in robot_poses],
        dtype=np.float64,
    )
    target_poses_r = np.array(
        [homogeneous_to_opencv(p)[0] for p in target_poses], dtype=np.float64
    )
    target_poses_t = np.array(
        [homogeneous_to_opencv(p)[1] for p in target_poses], dtype=np.float64
    )

    rotation, translation = cv.calibrateHandEye(
        robot_poses_r,
        robot_poses_t,
        target_poses_r,
        target_poses_t,
        method=cv.CALIB_HAND_EYE_PARK,
    )

    T = np.eye(4)
    T[0:3, 0:3] = rotation
    T[0:3, 3] = translation.reshape(3)
    return T


def draw_target_axes(frame, target_pose):
    rvec, tvec = homogeneous_to_opencv(target_pose)
    cv.drawFrameAxes(frame, camera_matrix, distortion, rvec, tvec, 0.01)


def visualize(cam_thread, arm, camera_to_base):
    base_to_camera = np.linalg.inv(camera_to_base)  # compute once

    while True:
        ok, frame = cam_thread.read()
        if not ok or frame is None:
            continue

        EE_to_camera = base_to_camera @ arm.pose()
        rvec, tvec = homogeneous_to_opencv(EE_to_camera)
        cv.drawFrameAxes(frame, camera_matrix, distortion, rvec, tvec, 0.01)
        cv.imshow("image", frame)

        if cv.waitKey(1) & 0xFF == 27:
            break

def collect(cam_thread, arm, target):
    enter_key = 13
    escape_key = 27
    data = []
    last_capture = 0.0

    while True:
        ok, frame = cam_thread.read()
        if not ok or frame is None:
            continue

        key = cv.waitKey(1) & 0xFF

        if key == escape_key:
            break

        # Run detection every frame for live preview
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        detection = target.find(gray)
        target_pose = None
        if detection is not None:
            target_pose = target.pose(detection)

        # Draw detection + axes on the frame
        if target_pose is not None:
            target.draw(frame, detection)
            draw_target_axes(frame, target_pose)
            cv.putText(frame, "target detected - press ENTER to capture", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv.putText(frame, "no target detected", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv.putText(frame, f"{len(data)} poses collected", (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Capture only on ENTER, and only if the live detection above succeeded
        if key == enter_key and (time.time() - last_capture) > 1.0:
            if target_pose is not None:
                arm_pose = arm.pose()
                data.append((arm_pose, target_pose))
                last_capture = time.time()
                print(f"{len(data)} poses collected")
            else:
                print("Enter pressed — no target detected")

        cv.imshow("image", frame)

    print(f"Collection finished with {len(data)} poses")
    return data

def main():
    if (side == RIGHT):
        side_str = "RIGHT"
    elif (side == LEFT):
        side_str = "LEFT"
    else:
        side_str = "Error with side variable naming"
        raise SystemExit(side_str)

    print(f"About to overwrite hand eye calibration for side: {side_str}")
    resp = input("\nPress ENTER to continue, press escape then enter to cancel: ")
    if resp != "":
        raise SystemExit("Cancelled.")
    
    argv = crtk.ral.parse_argv(sys.argv[1:])
    ral = crtk.ral("led_test_collection")

    camera = setup_camera(index=0)
    cam_thread = CameraThread(camera)

    
    if (side == RIGHT):
        arm = Arm(ral, "MTMR")
    else:
        arm = Arm(ral, "MTML")

    # adjust if target is different - this is default from the original registration
    target_size = 0.0075 / 2.0
    target = AsymCirclesTarget(size=target_size)

    collect_data = True # True if you want to do the hand eye calibration, False to view it

    def run():
        cv.namedWindow("image", cv.WINDOW_NORMAL)

        if collect_data:
            data = collect(cam_thread, arm, target)
            np.save("hand_eye_data.npy", np.array(data, dtype=object))
        else:
            data = np.load("hand_eye_data.npy", allow_pickle=True)

        robot_poses = [r for (r, t) in data]
        target_poses = [t for (r, t) in data]

        camera_to_base = calibrate(target_poses, robot_poses)
        print("camera_to_base:")
        print(camera_to_base)

        link3_wrt_camera = np.linalg.inv(camera_to_base)
        print("link3_wrt_camera:")
        print(link3_wrt_camera)
        np.savetxt("link3_wrt_camera.txt", link3_wrt_camera)

        visualize(cam_thread, arm, camera_to_base)

    try:
        ral.spin_and_execute(run)
    finally:
        cam_thread.stop()
        camera.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()