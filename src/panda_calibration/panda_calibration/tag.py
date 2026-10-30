"""Calibration tag (ArUco / AprilTag) detection.

Neither ``apriltag_ros`` nor ``apriltag`` is installed in this workspace, so
detection goes straight through ``cv2.aruco``, which carries the 36h11
dictionary.  OpenCV here is 4.5.4, i.e. the *old* aruco API
(``getPredefinedDictionary`` / ``DetectorParameters_create`` /
``detectMarkers``), not the 4.7+ ``ArucoDetector`` class.

Captures store the detected corners in detections.yaml; solve_handeye reads
them back (or re-detects from the saved images).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DICTIONARIES = {
    '36h11': cv2.aruco.DICT_APRILTAG_36h11,
    '4x4_50': cv2.aruco.DICT_4X4_50,
}


def dictionary(family: str = '36h11'):
    """Return a supported OpenCV marker dictionary by calibration family name."""
    try:
        dictionary_id = DICTIONARIES[family.lower()]
    except KeyError as error:
        available = ', '.join(sorted(DICTIONARIES))
        raise ValueError(f'unsupported marker family {family!r}; choose one of: {available}') from error
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def detector_parameters():
    params = cv2.aruco.DetectorParameters_create()
    # SUBPIX, not APRILTAG.  The apriltag quad refiner in OpenCV 4.5.4 does not
    # just refine corners, it can reject the marker outright: on a clean, fully
    # visible 84 px tag rendered in Gazebo it returned nothing while every other
    # refinement mode found id 0.  SUBPIX still buys the sub-pixel corner accuracy
    # that PnP wants, without the rejections.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    # Keep the threshold window large enough for the native 4K Azure images.
    # The old 23 px ceiling worked for the small Gazebo tag, but could miss the
    # same printed tag at several hundred pixels across in a full-resolution
    # capture.  A slightly higher constant also survives the dark workbench
    # background without changing the marker dictionary or requested tag ID.
    params.minMarkerPerimeterRate = 0.01
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 100
    params.adaptiveThreshWinSizeStep = 10
    params.adaptiveThreshConstant = 13
    return params


@dataclass
class Detection:
    detected: bool
    corners: np.ndarray | None = None   # 4x2 pixel corners, OpenCV order
    apparent_px: float = 0.0
    tag_id: int | None = None

    def as_dict(self) -> dict:
        return {
            'detected': bool(self.detected),
            'tag_id': None if self.tag_id is None else int(self.tag_id),
            'apparent_px': round(float(self.apparent_px), 2),
            'corners_px': (None if self.corners is None
                           else [[round(float(v), 2) for v in pt] for pt in self.corners]),
        }


def detect(image: np.ndarray, tag_id: int = 0, family: str = '36h11') -> Detection:
    """Find marker *tag_id* from *family* in a BGR or grayscale image."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary(family),
                                              parameters=detector_parameters())
    if ids is None:
        return Detection(detected=False)
    ids = ids.flatten()
    matches = np.where(ids == tag_id)[0]
    if matches.size == 0:
        return Detection(detected=False)
    quad = corners[int(matches[0])].reshape(4, 2)
    edges = [float(np.linalg.norm(quad[(i + 1) % 4] - quad[i])) for i in range(4)]
    return Detection(detected=True, corners=quad, apparent_px=float(np.mean(edges)),
                     tag_id=tag_id)


def draw(image: np.ndarray, detection: Detection) -> np.ndarray:
    """Annotate a copy of *image* with the detected quad."""
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if not detection.detected or detection.corners is None:
        cv2.putText(out, 'NO TAG', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
        return out
    quad = detection.corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [quad], True, (0, 255, 0), 2)
    cv2.putText(out, f'id={detection.tag_id} {detection.apparent_px:.0f}px',
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out
