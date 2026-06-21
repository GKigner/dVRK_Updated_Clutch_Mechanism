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

N_LABELS_PER_FRAME = 2

TARGET_FOLDER     = './led_data/'
OUTPUT_LABELS     = './led_labels.json'
CALIBRATION_FILE  = './led_calibration.json'

# How many frames to label. Each frame gets exactly two clicks: LEFT then RIGHT.
all_images = glob.glob('led_data/frames/frame_*.png')
N_TO_LABEL    = len(all_images)
print (N_TO_LABEL)
RESET_LABELS  = True   # True = ignore cached labels and start over

# residuals below this (pixels) are treated as inliers by the Huber loss
HUBER_F_SCALE = 2.0

_CALIB_FILE = (
    Path(__file__).resolve().parent / "camera_intrinsics.json"
)

with open(_CALIB_FILE) as f:
    _calib = json.load(f)

camera_matrix = np.array(_calib["camera_matrix"], dtype=np.float64)
distortion    = np.array(_calib["distortion"],    dtype=np.float64)


def pick_frame_indices(n_total, n_pick):
    n_pick = min(n_pick, n_total)
    return [int(x) for x in np.linspace(0, n_total - 1, n_pick, dtype=int)]


def label_frame(img_path, progress_idx, total, data_idx):
    """
    Exactly two left-clicks per frame: LEFT LED, then RIGHT LED.
    ginput is hard-capped at N_LABELS_PER_FRAME (=2); a third click is impossible.

    Use the matplotlib toolbar to zoom in before clicking if the LEDs are
    small; the home button resets zoom.
    Right-click / backspace = undo last click.
    Middle-click = finish early (rejected unless exactly 2 clicks have been made).
    Press 'x' to skip this frame, 'q' to quit labeling entirely.
    Returns: 2x2 array of clicks (LEFT then RIGHT), or "skip" / "quit".
    """
    img = cv2.imread(img_path)
    if img is None:
        return "skip"
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_rgb)
    ax.set_title(
        f"Frame {progress_idx + 1}/{total}  (data idx {data_idx})\n"
        "click LEFT LED then RIGHT LED (exactly 2 clicks)  |  "
        "x = skip  |  q = quit  |  right-click = undo"
    )
    plt.tight_layout()

    state = {"action": None}

    def on_key(event):
        if event.key in ('x', 'q'):
            state["action"] = event.key
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)

    try:
        clicks = plt.ginput(n=N_LABELS_PER_FRAME, timeout=0, show_clicks=True,
                            mouse_add=1, mouse_pop=3, mouse_stop=2)
    except Exception:
        clicks = []
    plt.close(fig)

    if state["action"] == 'q':
        return "quit"
    # Hard requirement: exactly two clicks. Anything else is a skip.
    if state["action"] == 'x' or len(clicks) != N_LABELS_PER_FRAME:
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

    A = []
    for P, (u, v) in zip(Ps, uvs):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.asarray(A)
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    X0 = X_h[:3] / X_h[3]

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
    if (side == RIGHT):
        side_str = "RIGHT"
    elif(side == LEFT):
        side_str = "LEFT"
    else:
        side_str = "side error - check that you formatted the side variable correctly"

    print(f"About to overwrite led calibration for side: {side_str}")
    resp = input("\nPress ENTER to continue, press escape then enter to cancel: ")
    if resp != "":
        raise SystemExit("Cancelled.")


    with open(join(TARGET_FOLDER, 'led_data.json')) as f:
        data = json.load(f)
    total_frames = len(data)

    # load cache unless resetting
    labels = {}
    if RESET_LABELS:
        if os.path.exists(OUTPUT_LABELS):
            print(f"RESET_LABELS=True: ignoring cached labels at {OUTPUT_LABELS}")
    elif os.path.exists(OUTPUT_LABELS):
        try:
            with open(OUTPUT_LABELS) as f:
                cached = {int(k): v for k, v in json.load(f).items()}
            labels = {k: v for k, v in cached.items()
                      if isinstance(v, list) and len(v) == N_LABELS_PER_FRAME}
            dropped = len(cached) - len(labels)
            if dropped:
                print(f"Dropped {dropped} cached entries that did not have exactly "
                      f"{N_LABELS_PER_FRAME} points.")
            print(f"Loaded {len(labels)} cached labels from {OUTPUT_LABELS}")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Cache at {OUTPUT_LABELS} is corrupt ({e}); starting fresh.")
            labels = {}

    # evenly spaced frames across the dataset, minus anything already cached
    queue = [j for j in pick_frame_indices(total_frames, N_TO_LABEL)
             if j not in labels]

    for idx in queue:
        entry = data[idx]
        img_path = join(TARGET_FOLDER, 'annotated_images', entry['annotated_image_name'])
        out = label_frame(img_path, len(labels), N_TO_LABEL, idx)

        if isinstance(out, str) and out == "quit":
            print("Quitting at user request.")
            break
        if isinstance(out, str):
            print(f"Frame {idx}: skipped")
            continue

        # Sanity check: the labeler should only ever hand back exactly 2 points.
        assert out.shape == (N_LABELS_PER_FRAME, 2), \
            f"Internal error: expected ({N_LABELS_PER_FRAME}, 2) clicks, got {out.shape}"

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

    # Only the two LEDs are written. Downstream consumers always derive the
    # ROI center as the midpoint of these two points.
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
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(calib, f, indent=2)

    if (side == RIGHT):
        with open('../live_scripts/led_calibration_right.json', 'w') as f:
            json.dump(calib, f, indent=2)
        with open('../recalibrate_leds/led_calibration_right.json', 'w') as f:
            json.dump(calib, f, indent=2)
    else:
        with open('../live_scripts/led_calibration_left.json', 'w') as f:
            json.dump(calib, f, indent=2)
        with open('../recalibrate_leds/led_calibration_left.json', 'w') as f:
            json.dump(calib, f, indent=2)

    print(f"\nWrote calibration to {CALIBRATION_FILE}")
    print("fixed_roi.py will pick it up automatically on next run.")


if __name__ == "__main__":
    main()