import cv2 as cv
import numpy as np
import crtk

import json
import numpy as np
from pathlib import Path

_CALIB_FILE = (
    Path(__file__).resolve().parent / "camera_intrinsics_left.json"
)

with open(_CALIB_FILE) as f:
    _calib = json.load(f)

camera_matrix = np.array(_calib["camera_matrix"], dtype=np.float64)
distortion    = np.array(_calib["distortion"],    dtype=np.float64)

def kdl_to_4x4(kdl_frame):
    T = np.eye(4, dtype=np.float64)
    p = kdl_frame.p
    T[0, 3] = p.x()
    T[1, 3] = p.y()
    T[2, 3] = p.z()
    M = kdl_frame.M
    # unrolled — avoids nested python loops
    T[0, 0] = M[0, 0]; T[0, 1] = M[0, 1]; T[0, 2] = M[0, 2]
    T[1, 0] = M[1, 0]; T[1, 1] = M[1, 1]; T[1, 2] = M[1, 2]
    T[2, 0] = M[2, 0]; T[2, 1] = M[2, 1]; T[2, 2] = M[2, 2]
    return T


def homogeneous_to_opencv(T):
    tvec = T[0:3, 3].reshape(3, 1).copy()
    rvec, _ = cv.Rodrigues(T[0:3, 0:3])
    return (rvec, tvec)


class Arm:
    class Local:
        def __init__(self, ral, expected_interval, operating_state_instance):
            self.crtk_utils = crtk.utils(self, ral, expected_interval, operating_state_instance)
            self.crtk_utils.add_forward_kinematics()

    def __init__(self, ral, arm_name, ros_namespace="", expected_interval=0.1):
        self.ral = ral.create_child(arm_name)
        self.crtk_utils = crtk.utils(self, self.ral, expected_interval)
        self.crtk_utils.add_operating_state()
        self.crtk_utils.add_measured_js()
        self.local = Arm.Local(
            self.ral.create_child("local"),
            expected_interval,
            operating_state_instance=self,
        )
        self.name = arm_name

    def pose(self, link=3):
        jp, ts = self.measured_jp()
        ee_cp = kdl_to_4x4(self.local.forward_kinematics(jp))
        link_cp = kdl_to_4x4(self.local.forward_kinematics(jp[:link]))
        cp = np.linalg.inv(link_cp) @ ee_cp
        return cp


class AsymCirclesTarget:
    def __init__(self, size, pattern=(3, 5)):
        self.size = size
        self.pattern = pattern

        pts = []
        for i in range(pattern[1]):
            for j in range(pattern[0]):
                pts.append([(2 * j + i % 2) * size, i * size, 0.0])
        self.object_points = np.ascontiguousarray(pts, dtype=np.float64)

        # pre-build a tuned blob detector — much faster than the default
        params = cv.SimpleBlobDetector_Params()
        params.minArea = 10
        params.maxArea = 2000
        params.filterByColor = True
        params.blobColor = 0           # dark circles
        params.filterByCircularity = True
        params.minCircularity = 0.7
        params.filterByConvexity = False
        params.filterByInertia = False
        self._detector = cv.SimpleBlobDetector_create(params)

        # pre-allocate for solvePnP
        self._rvec = np.zeros((3, 1), dtype=np.float64)
        self._tvec = np.zeros((3, 1), dtype=np.float64)

    def find(self, gray):
        """Pass a grayscale image. Returns centers or None."""
        ok, centers = cv.findCirclesGrid(
            gray,
            self.pattern,
            flags=cv.CALIB_CB_ASYMMETRIC_GRID,
            blobDetector=self._detector,
        )
        if not ok or len(centers) != len(self.object_points):
            return None
        return centers

    def pose(self, detection, frame=None):
        ok, rvec, tvec = cv.solvePnP(
            self.object_points,
            detection,
            camera_matrix,
            distortion,
            rvec=self._rvec,
            tvec=self._tvec,
            useExtrinsicGuess=False,
            flags=cv.SOLVEPNP_IPPE,   # fast solver for planar targets
        )
        if not ok:
            return None
        self._rvec[:] = rvec
        self._tvec[:] = tvec
        T = np.eye(4, dtype=np.float64)
        R, _ = cv.Rodrigues(rvec)
        T[0:3, 0:3] = R
        T[0:3, 3] = tvec.flatten()
        return T

    def draw(self, frame, detection):
        """Call separately only when you actually want to draw."""
        cv.drawChessboardCorners(frame, self.pattern, detection, True)
        cv.drawFrameAxes(
            frame, camera_matrix, distortion,
            self._rvec, self._tvec, 0.01,
        )