#!/usr/bin/env python3
import cv2
import numpy as np
import glob
import random

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
n_select = len(all_images)
images = random.sample(all_images, n_select)
if (n_select < 50):
    print(f"warning: only have {n_select} images. Might need to gather more than 50")

# Counters
total_markers = 0
total_corners = 0
n_used = 0

# Loop through images
quit_early = False

for image_name in images:
    image = cv2.imread(image_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    
    n_markers = len(marker_ids) if marker_ids is not None else 0
    n_corners = len(charuco_ids) if charuco_ids is not None else 0
    total_markers += n_markers
    total_corners += n_corners
    print(f"{image_name}: {n_markers} markers, {n_corners} corners")
    
    if charuco_ids is not None and len(charuco_ids) > 110:
        all_charuco_corners.append(charuco_corners)
        all_charuco_IDs.append(charuco_ids)
        image_size = gray.shape[::-1]
        n_used += 1
    
    if charuco_corners is not None and len(charuco_corners) > 0:
        cv2.aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)
    if marker_corners is not None:
        cv2.aruco.drawDetectedMarkers(image, marker_corners, marker_ids)
    cv2.imshow('Space to continue, q to quit early', image)

    # Wait for SPACE (next) or q (quit)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord(' '):
            break
        elif key == ord('q'):
            quit_early = True
            break
    if quit_early:
        break

cv2.destroyAllWindows()

# Summary
print(f"\n--- Detection summary ---")
if (quit_early):
    print("quit early statistics")
print(f"Images processed: {n_select}")
print(f"Images used for calibration (>110 corners): {n_used}")
print(f"Total markers detected: {total_markers} (avg {total_markers/n_select:.1f} per image)")
print(f"Total ChArUco corners detected: {total_corners} (avg {total_corners/n_select:.1f} per image)")

# 4. Calibrate Camera
all_obj_points = []
all_img_points = []
for corners, ids in zip(all_charuco_corners, all_charuco_IDs):
    obj_pts, img_pts = board.matchImagePoints(corners, ids)
    all_obj_points.append(obj_pts)
    all_img_points.append(img_pts)

_, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
    all_obj_points, all_img_points, image_size, None, None
)

print("Camera matrix, formatted as:\n[[  f_x   0   c_x   ]\n [   0   f_y  c_y   ]\n [   0    0    1    ]]\n")
print(camera_matrix)
print("")
print("Distortion coefficients, formatted as:\n[[k_1  k_2  p_1  p_2  k_3]] \n")
print(dist_coeffs)