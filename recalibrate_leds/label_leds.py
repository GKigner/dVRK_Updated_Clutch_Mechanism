import cv2
import numpy as np
import json
import os
from os.path import join
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path
import glob


RIGHT = True
LEFT  = False

# Hyperparameter to select what side you are working with
# must be RIGHT or LEFT
side = RIGHT


HERE = Path(__file__).resolve().parent
LIVE_SCRIPTS_DIR = HERE.parent / "live_scripts"

if side == RIGHT:
    side_tag = "right"
elif side == LEFT:
    side_tag = "left"
else:
    raise SystemExit("side variable incorrectly entered, please check code")


TARGET_FOLDER = HERE / f"led_data_{side_tag}"
_CALIB_FILE   = HERE / f"camera_intrinsics_{side_tag}.json"

OUTPUT_LABELS = HERE / f"led_labels_{side_tag}.json"

CALIBRATION_FILE_1 = HERE / f"led_calibration_{side_tag}.json"             
CALIBRATION_FILE_2 = LIVE_SCRIPTS_DIR / f"led_calibration_{side_tag}.json" 


# How many frames to label. Each frame gets exactly two clicks: LEFT then RIGHT.
all_images = glob.glob(str(TARGET_FOLDER / "frames" / "frame_*.png"))
N_TO_LABEL = len(all_images)
print(N_TO_LABEL)

RESET_LABELS  = True   # True = ignore cached labels and start over

# residuals below this (pixels) are treated as inliers by the Huber loss
HUBER_F_SCALE = 2.0


with open(_CALIB_FILE) as f:
    _calib = json.load(f)

camera_matrix = np.array(_calib["camera_matrix"], dtype=np.float64)
distortion    = np.array(_calib["distortion"],    dtype=np.float64)


def pick_frame_indices(n_total, n_pick):
    n_pick = min(n_pick, n_total)
    return [int(x) for x in np.linspace(0, n_total - 1, n_pick, dtype=int)]


def label_frame(img_path, progress_idx, total, data_idx):
    """
    Two left-clicks = LEFT then RIGHT LED.
    Use the matplotlib toolbar to zoom in before clicking if the LEDs are
    small; the home button resets zoom.
    Right-click / backspace = undo last click.
    Middle-click = finish (also auto-finishes after 2 clicks).
    Press 'x' to skip this frame, 'q' to quit labeling entirely.
    Returns: 2x2 array of clicks, or "skip" / "quit".
    """
    img = cv2.imread(img_path)
    if img is None:
        return "skip"
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_rgb)
    ax.set_title(
        f"Frame {progress_idx + 1}/{total}  (data idx {data_idx})\n"
        "click LEFT LED then RIGHT LED  |  x = skip  |  q = quit  |  right-click = undo"
    )
    plt.tight_layout()

    state = {"action": None}

    def on_key(event):
        if event.key in ('x', 'q'):
            state["action"] = event.key
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)

    try:
        clicks = plt.ginput(n=2, timeout=0, show_clicks=True,
                            mouse_add=1, mouse_pop=3, mouse_stop=2)
    except Exception:
        clicks = []
    plt.close(fig)

    if state["action"] == 'q':
        return "quit"
    if state["action"] == 'x' or len(clicks) != 2:
        return "skip"
    return np.array(clicks, dtype=np.float64)


def _proj_mat(T_ct):
    R = T_ct[:3, :3]; t = T_ct[:3, 3:4]
    return camera_matrix @ np.hstack([R, t])


def _undistort(uv):
    return cv2.undistortPoints(uv.reshape(1, 1, 2), camera_matrix, distortion,
                               P=camera_matrix).reshape(2)


def triangulate(observations):
    """
    Robust multi-view triangulation:
      (1) Linear DLT initialization from all views (SVD on stacked equations)
      (2) Nonlinear refinement of pixel reprojection error with a Huber loss
    observations: list of (T_ct, uv_pixel)
    Returns: (X, per_view_reproj_errors_pixels)
    """
    Ps  = [_proj_mat(T)   for T, _  in observations]
    uvs = [_undistort(uv) for _, uv in observations]

    # --- (1) multi-view linear DLT init ---
    A = []
    for P, (u, v) in zip(Ps, uvs):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.asarray(A)
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    X0 = X_h[:3] / X_h[3]

    # --- (2) nonlinear refinement with Huber loss ---
    def residuals(X):
        r = []
        for P, (u, v) in zip(Ps, uvs):
            x = P @ np.append(X, 1.0)
            r.append(x[0] / x[2] - u)
            r.append(x[1] / x[2] - v)
        return np.asarray(r)

    sol = least_squares(residuals, X0, loss='huber',
                        f_scale=HUBER_F_SCALE, method='trf')
    X = sol.x

    final = residuals(X).reshape(-1, 2)
    per_view_err = np.linalg.norm(final, axis=1)
    return X, per_view_err


def main():
    side_str = side_tag.upper()

    print(f"About to overwrite led calibration for side: {side_str}")
    resp = input("\nPress ENTER to continue, press escape then enter to cancel: ")
    if resp != "":
        raise SystemExit("Cancelled.")


    with open(TARGET_FOLDER / "led_data.json") as f:
        data = json.load(f)
    total_frames = len(data)

    # load cache unless resetting
    labels = {}
    if RESET_LABELS:
        if OUTPUT_LABELS.exists():
            print(f"RESET_LABELS=True: ignoring cached labels at {OUTPUT_LABELS}")
    elif OUTPUT_LABELS.exists():
        try:
            with open(OUTPUT_LABELS) as f:
                labels = {int(k): v for k, v in json.load(f).items()}
            print(f"Loaded {len(labels)} cached labels from {OUTPUT_LABELS}")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Cache at {OUTPUT_LABELS} is corrupt ({e}); starting fresh.")
            labels = {}

    # evenly spaced frames across the dataset, minus anything already cached
    queue = [j for j in pick_frame_indices(total_frames, N_TO_LABEL)
             if j not in labels]

    for idx in queue:
        entry = data[idx]
        img_path = str(TARGET_FOLDER / "annotated_images" / entry['annotated_image_name'])
        out = label_frame(img_path, len(labels), N_TO_LABEL, idx)

        if isinstance(out, str) and out == "quit":
            print("Quitting at user request.")
            break
        if isinstance(out, str):
            print(f"Frame {idx}: skipped")
            continue

        labels[idx] = out.tolist()
        with open(OUTPUT_LABELS, 'w') as f:
            json.dump(labels, f, indent=2)
        print(f"Frame {idx}: labeled  ({len(labels)}/{N_TO_LABEL})")

    if len(labels) < 2:
        print("Need at least 2 labeled frames to triangulate.")
        return

    # triangulate each LED across all labeled frames
    frame_ids = []
    left_obs  = []
    right_obs = []
    for idx, pts in labels.items():
        T_ct = np.asarray(data[idx]['gripper_wrt_camera'], dtype=np.float64)
        frame_ids.append(idx)
        left_obs.append((T_ct,  np.asarray(pts[0])))
        right_obs.append((T_ct, np.asarray(pts[1])))

    p_left,  err_left  = triangulate(left_obs)
    p_right, err_right = triangulate(right_obs)
    center = (p_left + p_right) / 2.0
    axis   = p_right - p_left
    length = np.linalg.norm(axis)

    print(f"\nLabeled frames used: {len(labels)}")
    print(f"LED left  (gripper frame): {p_left}")
    print(f"LED right (gripper frame): {p_right}")
    print(f"Midpoint:                  {center}")
    print(f"Separation:                {length*1000:.2f} mm")

    print(f"\nReprojection error (pixels):")
    print(f"  left  LED: mean={err_left.mean():.2f}  "
          f"median={np.median(err_left):.2f}  max={err_left.max():.2f}")
    print(f"  right LED: mean={err_right.mean():.2f}  "
          f"median={np.median(err_right):.2f}  max={err_right.max():.2f}")

    worst_n = min(5, len(frame_ids))
    combined = np.maximum(err_left, err_right)
    worst_idx = np.argsort(combined)[::-1][:worst_n]
    print(f"\nWorst {worst_n} frames by reprojection error (max of left/right):")
    for k in worst_idx:
        print(f"  frame {frame_ids[k]:4d}: "
              f"left={err_left[k]:6.2f} px, right={err_right[k]:6.2f} px")

    # ---- write calibration file consumed by capture_roi.py ----
    calib = {
        "led_left":  p_left.tolist(),
        "led_right": p_right.tolist(),
        "n_frames_used": int(len(labels)),
        "separation_mm": float(length * 1000.0),
        "reprojection_error_px": {
            "left": {
                "mean":   float(err_left.mean()),
                "median": float(np.median(err_left)),
                "max":    float(err_left.max()),
            },
            "right": {
                "mean":   float(err_right.mean()),
                "median": float(np.median(err_right)),
                "max":    float(err_right.max()),
            },
        },
    }
    with open(CALIBRATION_FILE_1, 'w') as f:
        json.dump(calib, f, indent=2)
    with open(CALIBRATION_FILE_2, 'w') as f:
        json.dump(calib, f, indent=2)

    print(f"\nWrote calibration to:")
    print(f"  {CALIBRATION_FILE_1}")
    print(f"  {CALIBRATION_FILE_2}")


if __name__ == "__main__":
    main()