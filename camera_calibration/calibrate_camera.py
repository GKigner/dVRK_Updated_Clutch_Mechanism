#!/usr/bin/env python3
import cv2
import numpy as np
import glob
import random
import json
from pathlib import Path


RIGHT = True
LEFT  = False

# Hyperparameter to select what side you are working with
# must be RIGHT or LEFT
side = LEFT

side_str = "RIGHT" if side == RIGHT else "LEFT"
print(f"About to overwrite intrinsics for side: {side_str}")
resp = input("\nPress ENTER to continue, press escape then enter to cancel: ")
if resp != "":
    raise SystemExit("Cancelled.")


CALIB_FILE_1 = Path(__file__).resolve().parent.parent / "hand_eye_calibration/camera_intrinsics.json"
CALIB_FILE_2 = Path(__file__).resolve().parent.parent / "led_calibration/camera_intrinsics.json"
if (side == RIGHT):
    CALIB_FILE_3 = Path(__file__).resolve().parent.parent / "live_scripts/camera_intrinsics_right.json"
    CALIB_FILE_4 = Path(__file__).resolve().parent.parent / "recalibrate_leds/camera_intrinsics_right.json"
elif (side == LEFT):
    CALIB_FILE_3 = Path(__file__).resolve().parent.parent / "live_scripts/camera_intrinsics_left.json"
    CALIB_FILE_4 = Path(__file__).resolve().parent.parent / "recalibrate_leds/camera_intrinsics_left.json"
else:
    print("Error: invalide side variable")


# to allow pasting into common.py easily
def format_camera_matrix(M):
    fx, fy = M[0, 0], M[1, 1]
    cx, cy = M[0, 2], M[1, 2]
    rows = [
        [f"{fx:.2f}", "0.0", f"{cx:.2f}"],
        ["0.0", f"{fy:.2f}", f"{cy:.2f}"],
        ["0.0", "0.0", "1.0"],
    ]
    w = [max(len(rows[r][c]) for r in range(3)) for c in range(3)]
    lines = []
    for r in rows:
        cells = [(r[c] + ("," if c < 2 else "")).ljust(w[c] + 2) for c in range(3)]
        lines.append("    [" + "".join(cells).rstrip() + "],")
    
    return "camera_matrix= np.array([\n" + "\n".join(lines) + "\n], dtype=np.float64)"


def format_distortion(d):
    vals = np.asarray(d).flatten()
    body = ", ".join(f"{v:.8f}" for v in vals)
    return f"distortion = np.array([{body}], dtype=np.float64)"


# Define board
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
board = cv2.aruco.CharucoBoard(
    (12, 16),    # (columns, rows)
    0.017,       # square size in METERS (17mm = 0.017m)
    0.012,       # marker size in METERS
    dictionary)
board.setLegacyPattern(True)  # Match calib.io convention


detector = cv2.aruco.CharucoDetector(board)
all_charuco_corners = []
all_charuco_IDs = []
image_size = None
all_images = glob.glob('target_images/target_*.jpg')
n_select =  len(all_images)
images = random.sample(all_images, n_select)
if (n_select < 50):
    print(f"warning: only have {n_select} images. Might need to gather more than 50")
n_used = 0

# Loop through images
for image_name in images:
    image = cv2.imread(image_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect ChArUco corners
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    
    if charuco_ids is not None and len(charuco_ids) > 110:
        all_charuco_corners.append(charuco_corners)
        all_charuco_IDs.append(charuco_ids)
        image_size = gray.shape[::-1]
        n_used += 1

print(f"Images used for calibration (>110 corners): {n_used}")
if n_used < 20:
    raise SystemExit(f"Only {n_used} usable images — need more (at least 20)")

# Calibrate Camera
all_obj_points = []
all_img_points = []
for corners, ids in zip(all_charuco_corners, all_charuco_IDs):
    obj_pts, img_pts = board.matchImagePoints(corners, ids)
    all_obj_points.append(obj_pts)
    all_img_points.append(img_pts)

_, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
    all_obj_points, all_img_points, image_size, None, None
)

# Write to the proper files!
data = {
    "camera_matrix": camera_matrix.tolist(),
    "distortion":    dist_coeffs.flatten().tolist(),
    "image_size":    list(image_size),
}
with open(CALIB_FILE_1, "w") as f:
    json.dump(data, f, indent=2)
print(f"wrote {CALIB_FILE_1}")

with open(CALIB_FILE_2, "w") as f:
    json.dump(data, f, indent=2)
print(f"wrote {CALIB_FILE_2}")

with open(CALIB_FILE_3, "w") as f:
    json.dump(data, f, indent=2)
print(f"wrote {CALIB_FILE_3}")

with open(CALIB_FILE_4, "w") as f:
    json.dump(data, f, indent=2)
print(f"wrote {CALIB_FILE_4}")

# for reference
print(format_camera_matrix(camera_matrix))
print(format_distortion(dist_coeffs))