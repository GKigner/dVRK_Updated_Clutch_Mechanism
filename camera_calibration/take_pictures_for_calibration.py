#!/usr/bin/env python3
import cv2
import os
import re


# SET HYPERPARAMETER FOR DATA COLLECTION
# True: start from target_1.jpg and overwrite
# False: append to existing
overwrite = True  

# Setup
save_dir = 'target_images'
os.makedirs(save_dir, exist_ok=True)

# Determine starting number
if overwrite:
    next_num = 1
    print("Overwrite mode: starting at target_1.jpg (existing files will be overwritten)")
else:
    existing = [
        f for f in os.listdir(save_dir)
        if f.startswith('target_') and f.endswith('.jpg')
    ]
    nums = []
    for f in existing:
        m = re.match(r"target_(\d+)\.jpg", f)
        if m:
            nums.append(int(m.group(1)))
    next_num = max(nums) + 1 if nums else 1
    print(f"Append mode: starting at target_{next_num}.jpg")

print("Controls:")
print("  SPACE - capture image")
print("  q     - quit")

# Open camera and set camera properly
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Could not open camera.")
    exit()
cap.set(cv2.CAP_PROP_SETTINGS, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
correct_dimensions = actual_w == 640 and actual_h == 480


if (not correct_dimensions):
    print(f"Error with resolution of images, should be 640 x 480, but is instead reported by driver as {actual_w} x {actual_h}")
else:
    start_num = next_num
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error")
            break
        h, w = frame.shape[:2]
        # Note that if this error occurs, the driver is incorrectly reading the images incorrectly too
        if (w, h) != (640, 480):
            print(f"ERROR: Actual frame is {w} x {h}, not 640 x 480.")
            print("Do not use these images for 640x480 calibration.")
            break
        cv2.imshow('Capture - SPACE to save, q to quit', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Space to capture
            filename = f'{save_dir}/target_{next_num}.jpg'
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename} | size: {w} x {h}")
            next_num += 1
        elif key == ord('q'):  # Q to quit
            break


cap.release()
cv2.destroyAllWindows()
print(f"Done. Captured {next_num - start_num} images this session.")