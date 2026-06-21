#!/usr/bin/env python3
# clutch_signal_left_6.py
# Self-contained live LED ON/OFF detector for the left arm.
#
# Pipeline every 10 ms:  capture 640x480 frame -> extract 220x120 ROI
#                         (projected LED center, rotated) -> CNN -> 0/1
#                         -> debounce -> publish.

import os
import sys
import json
import time

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn

import crtk
from std_msgs.msg import Int32
from std_msgs.msg import Bool

# camera_matrix, distortion, Arm
from common_left import *

# =========================
# Configuration
# =========================
CALIBRATION_FILE      = "./led_calibration_left.json"
MODEL_FILE            = "./led_detection_CNN.pt.pt"

EXPECTED_W = 640
EXPECTED_H = 480

# ROI size MUST match training (the CNN was trained on 220x120 crops). THROW ERROR OTHERWISE? SHOULD NO BE MODIFID
ROI_WIDTH_PX  = 220
ROI_HEIGHT_PX = 120
PAD_VALUE     = 0
_HALF_W = ROI_WIDTH_PX  / 2.0
_HALF_H = ROI_HEIGHT_PX / 2.0

# =========================
# Debounce hyperparameters
# =========================
# Require this many consecutive identical raw readings before the published
# state is allowed to flip. At 10 ms/frame: 5 -> 50 ms, 10 -> 100 ms, etc.
DEBOUNCE_N = 5

DEBOUNCE_INITIAL_STATE = 0


DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ON_INDEX = 1                   # class_names = ['OFF', 'ON'] -> ON is index 1

# Normalization constants (built once, reused every frame): (x - 0.5) / 0.5
_MEAN = torch.tensor(0.5, device=DEVICE)
_STD  = torch.tensor(0.5, device=DEVICE)


# =========================
# Model architecture
# =========================
class CNN6(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,   32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32,  64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


# =========================
# 1. Load the CNN
# =========================
def load_model(path=MODEL_FILE):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict):
        state = None
        for key in ("model_state", "model_state_dict", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
        if state is None:
            state = ckpt
    else:
        raise RuntimeError(f"Expected a checkpoint dict; got {type(ckpt).__name__}.")

    model = CNN6(num_classes=2)
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
    # Warmup so the first real frame isn't penalized by lazy CUDA init.
    with torch.inference_mode():
        model(torch.zeros(1, 3, ROI_HEIGHT_PX, ROI_WIDTH_PX, device=DEVICE))

    return model


# =========================
# Setup helpers (camera + calibration)
# =========================
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
        raise RuntimeError(f"Frame is {w}x{h}, not {EXPECTED_W}x{EXPECTED_H}.")
    return camera


def load_led_calibration(path=CALIBRATION_FILE):
    """Load the LED center + a point along the axis (gripper-frame 3D points)."""
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

    axis   = led_right - led_left
    length = float(np.linalg.norm(axis))
    if length < 1e-9:
        raise ValueError("LED endpoints coincident; bad calibration.")
    pt_axis = center + 0.01 * (axis / length)

    pts_3d = np.array([center, pt_axis], dtype=np.float64).reshape(-1, 1, 3)
    return pts_3d


# =========================
# 2. Capture an image
# =========================
def capture_frame(camera):
    """Grab one 640x480 BGR frame. Returns the frame, or None if the read failed."""
    ok, frame = camera.read()
    return frame if ok else None


# =========================
# 3. Extract the ROI (live pose -> projected center -> rotated 220x120 crop)
# =========================
def extract_roi(frame, arm, link3_wrt_camera, pts_3d):
    T_ct = link3_wrt_camera @ arm.pose()                 # gripper wrt camera, live
    rvec, _ = cv.Rodrigues(T_ct[:3, :3])
    tvec    = T_ct[:3, 3]
    pts, _  = cv.projectPoints(pts_3d, rvec, tvec, camera_matrix, distortion)
    pts     = pts.reshape(-1, 2)
    center    = pts[0]
    axis_vec  = pts[1] - pts[0]
    angle_deg = float(np.degrees(np.arctan2(axis_vec[1], axis_vec[0])))

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
# 4. Pass the ROI into the CNN -> 0 or 1 (1 = ON)
# =========================
@torch.inference_mode()
def classify_roi(model, roi_bgr):
    img = cv.cvtColor(roi_bgr, cv.COLOR_BGR2RGB)         # training used RGB
    t = torch.from_numpy(img).to(DEVICE)
    t = t.permute(2, 0, 1).float().div_(255.0)           # 3xHxW, [0,1]
    t = t.sub_(_MEAN).div_(_STD)                         # (x - 0.5) / 0.5
    out = model(t.unsqueeze(0)).reshape(-1)              # 2 logits
    return int(out.argmax().item() == ON_INDEX)


# =========================
# 5. Debounce: require N consecutive identical readings before flipping state.
# =========================
class Debouncer:
    """
    Symmetric N-in-a-row debounce.

    State only flips after `n_required` consecutive raw readings that disagree
    with the currently published state. Any reading that matches the published
    state immediately resets the counter.

    Usage:
        deb = Debouncer(n_required=5, initial_state=0)
        published = deb.update(raw_reading)   # 0 or 1
    """

    def __init__(self, n_required: int = DEBOUNCE_N,
                 initial_state: int = DEBOUNCE_INITIAL_STATE):
        if n_required < 1:
            raise ValueError("n_required must be >= 1")
        if initial_state not in (0, 1):
            raise ValueError("initial_state must be 0 or 1")
        self.n_required   = int(n_required)
        self.state        = int(initial_state)
        self._disagree_n  = 0   # consecutive readings disagreeing with self.state

    def update(self, raw: int) -> int:
        """Feed one raw 0/1 reading; return the debounced published state."""
        if raw == self.state:
            self._disagree_n = 0
        else:
            self._disagree_n += 1
            if self._disagree_n >= self.n_required:
                self.state = raw
                self._disagree_n = 0
        return self.state


# =========================
# main
# =========================
def main():
    crtk.ral.parse_argv(sys.argv[1:])
    ral = crtk.ral("clutch_signal_left")

    model            = load_model(MODEL_FILE)
    camera           = setup_camera(index=0) # EDIT TO MATCH CORRECT CAMERA

    arm              = Arm(ral, "MTML")
    link3_wrt_camera = np.loadtxt("./link3_wrt_camera_left.txt")
    pts_3d           = load_led_calibration()

    debouncer = Debouncer(n_required=DEBOUNCE_N,
                          initial_state=DEBOUNCE_INITIAL_STATE)

    led_publisher = ral.publisher("led_left_on", Bool, queue_size=1)

    def run():
        next_t = time.perf_counter()
        while True:
            next_t += 0.010  # 100 Hz target
            frame = capture_frame(camera)
            if frame is not None:
                roi    = extract_roi(frame, arm, link3_wrt_camera, pts_3d)
                raw    = classify_roi(model, roi)         # 1 = ON, 0 = OFF
                led_on = debouncer.update(raw)            # debounced published state

                # now publish led_on as ROS node
                led_msg = Bool()
                led_msg.data = (led_on == 1)
                led_publisher.publish(led_msg)

            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)

    try:
        ral.spin_and_execute(run)
    finally:
        camera.release()


if __name__ == "__main__":
    main()