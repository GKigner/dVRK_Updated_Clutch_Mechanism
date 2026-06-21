# capture_roi.py
# Batch ROI extraction with a fixed pixel-size, rotation-aligned output.
# Reads the two calibrated LED positions from led_calibration.json (produced
# by label_leds.py) and derives the ROI center as their midpoint. Annotated
# output also shows the predicted positions of both individual LEDs.

import cv2
import numpy as np
import json
import os
from os.path import join
import json
import numpy as np
from pathlib import Path


RIGHT = True
LEFT  = False

# Hyperparameter to select what side you are working with
# must be RIGHT or LEFT
side = RIGHT

HERE = Path(__file__).resolve().parent

if side == RIGHT:
    side_tag = "right"
elif side == LEFT:
    side_tag = "left"
else:
    raise SystemExit("side variable incorrectly entered, please check code")

CALIBRATION_FILE = HERE / f"led_calibration_{side_tag}.json"   # written by label_leds.py
_CALIB_FILE      = HERE / f"camera_intrinsics_{side_tag}.json"

TARGET_FOLDER       = HERE / f"led_data_{side_tag}"
OUTPUT_FOLDER       = HERE / f"fixed_roi_output_{side_tag}"

with open(_CALIB_FILE) as f:
    _calib = json.load(f)
camera_matrix = np.array(_calib["camera_matrix"], dtype=np.float64)
distortion    = np.array(_calib["distortion"],    dtype=np.float64)

ROI_WIDTH_PX  = 220   # long axis (parallel to LEDs)
ROI_HEIGHT_PX = 120   # short axis (perpendicular to LEDs)
PAD_VALUE = 0          # fill for any out-of-image pixels
_HALF_W = ROI_WIDTH_PX  / 2.0
_HALF_H = ROI_HEIGHT_PX / 2.0


def load_calibration(path=CALIBRATION_FILE):
    """Load the two LED positions and derive the ROI center as their midpoint.

    The labeler (label_leds.py) only ever produces two points -- LEFT and
    RIGHT. The ROI center is always the midpoint of those two LEDs; the axis
    direction comes from LEFT -> RIGHT.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Calibration file not found: {path}\n"
            f"Run label_leds.py first to generate it."
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
    length = np.linalg.norm(axis)
    if length < 1e-9:
        raise ValueError("LED left and right positions are coincident; bad calibration.")
    axis_unit = axis / length
    # center + a point 1 cm along the LED axis (used to recover image angle)
    pt_axis = center + 0.01 * axis_unit
    pts_3d = np.array([center, pt_axis],
                      dtype=np.float64).reshape(-1, 1, 3)
    individual_leds_3d = np.array([led_left, led_right],
                                  dtype=np.float64).reshape(-1, 1, 3)
    print(f"Loaded calibration from {path}")
    print(f"  frames used:   {calib.get('n_frames_used', '?')}")
    print(f"  separation:    {calib.get('separation_mm', length*1000):.2f} mm")
    rep = calib.get("reprojection_error_px")
    if rep:
        bits = []
        for k in ("left", "right"):
            if k in rep:
                bits.append(f"{k} median={rep[k]['median']:.2f} max={rep[k]['max']:.2f}")
        if bits:
            print("  reproj err px: " + " | ".join(bits))
    return pts_3d, individual_leds_3d
_PTS_3D, _INDIVIDUAL_LEDS_3D = load_calibration()

def project_center_and_axis(T_ct):
    """Project ROI center + axis point. Returns (center_uv, angle_deg)."""
    R = T_ct[:3, :3]
    t = T_ct[:3,  3]
    rvec, _ = cv2.Rodrigues(R)
    pts, _ = cv2.projectPoints(_PTS_3D, rvec, t, camera_matrix, distortion)
    pts = pts.reshape(-1, 2)
    center = pts[0]
    axis_vec = pts[1] - pts[0]
    angle_deg = float(np.degrees(np.arctan2(axis_vec[1], axis_vec[0])))
    return center, angle_deg

def project_individual_leds(T_ct):
    """Project the two LED positions. Returns 2x2 array of pixel coords."""
    R = T_ct[:3, :3]
    t = T_ct[:3,  3]
    rvec, _ = cv2.Rodrigues(R)
    pts, _ = cv2.projectPoints(_INDIVIDUAL_LEDS_3D, rvec, t, camera_matrix, distortion)
    return pts.reshape(-1, 2)

def crop_rotated_window(frame, center, angle_deg):
    """Rotate around the ROI center so the LED axis is horizontal, crop fixed size."""
    cx, cy = float(center[0]), float(center[1])
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    M[0, 2] += _HALF_W - cx
    M[1, 2] += _HALF_H - cy
    return cv2.warpAffine(
        frame, M, (ROI_WIDTH_PX, ROI_HEIGHT_PX),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=PAD_VALUE,
    )

def annotate_frame(frame, center, angle_deg, T_ct):
    """Draw the crop rect, ROI center, and individual LED predictions."""
    img = frame.copy()
    # Crop rectangle (red)
    rect = ((float(center[0]), float(center[1])),
            (ROI_WIDTH_PX, ROI_HEIGHT_PX),
            angle_deg)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.polylines(img, [box], isClosed=True, color=(0, 0, 255), thickness=2)
    # ROI center (green dot)
    cv2.circle(img,
               (int(round(center[0])), int(round(center[1]))),
               4, (0, 255, 0), -1)
    # Individual LEDs (cyan = left, magenta = right) with axis line between them
    led_pts = project_individual_leds(T_ct)
    left_uv  = tuple(np.round(led_pts[0]).astype(int))
    right_uv = tuple(np.round(led_pts[1]).astype(int))
    cv2.line(img, left_uv, right_uv, (255, 255, 255), 1)
    cv2.circle(img, left_uv,  6, (255, 255, 0), 2)   # cyan circle
    cv2.circle(img, right_uv, 6, (255, 0, 255), 2)   # magenta circle
    return img

def find_fixed_roi(frame, T_ct):
    """Full per-frame pipeline. Returns dict with roi, annotated, center, angle."""
    center, angle_deg = project_center_and_axis(T_ct)
    roi = crop_rotated_window(frame, center, angle_deg)
    annotated = annotate_frame(frame, center, angle_deg, T_ct)
    return {
        "roi":       roi,
        "annotated": annotated,
        "center_uv": (float(center[0]), float(center[1])),
        "angle_deg": angle_deg,
    }

def process_folder(target_folder=TARGET_FOLDER, output_folder=OUTPUT_FOLDER):
    target_folder = Path(target_folder)
    output_folder = Path(output_folder)
    with open(target_folder / "led_data.json") as f:
        data = json.load(f)
    annotated_dir = output_folder / "annotated"
    roi_dir       = output_folder / "rois"
    for d in (annotated_dir, roi_dir):
        d.mkdir(parents=True, exist_ok=True)
    for i, entry in enumerate(data):
        T_ct = np.asarray(entry['gripper_wrt_camera'], dtype=np.float64)
        img_path = target_folder / "annotated_images" / entry['annotated_image_name']
        frame = cv2.imread(str(img_path))
        if frame is None:
            raise FileNotFoundError(f"Could not read {img_path}")
        result = find_fixed_roi(frame, T_ct)
        cv2.imwrite(str(annotated_dir / f"annotated_{i}.png"), result["annotated"])
        cv2.imwrite(str(roi_dir       / f"roi_{i}.png"),       result["roi"])
        u, v = result["center_uv"]
        print(f"Frame {i}: center=({u:.1f}, {v:.1f})  angle={result['angle_deg']:+.1f}°  "
              f"roi_shape={result['roi'].shape[:2]}")
        
def remove_processed_files(output_folder=OUTPUT_FOLDER):
    output_folder = Path(output_folder)
    if not output_folder.is_dir():
        return
    for root, _, files in os.walk(output_folder):
        for f in files:
            if f.endswith('.png'):
                os.remove(join(root, f))

if __name__ == "__main__":
    remove_processed_files(OUTPUT_FOLDER)
    process_folder(TARGET_FOLDER, output_folder=OUTPUT_FOLDER)