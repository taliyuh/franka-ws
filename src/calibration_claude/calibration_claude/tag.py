"""AprilTag 36h11 asset generation and detection.

Neither ``apriltag_ros`` nor ``apriltag`` is installed in this workspace, so
detection goes straight through ``cv2.aruco``, which carries the 36h11
dictionary.  OpenCV here is 4.5.4, i.e. the *old* aruco API
(``getPredefinedDictionary`` / ``DetectorParameters_create`` /
``detectMarkers``), not the 4.7+ ``ArucoDetector`` class.

Detection is used only as a visibility check on the captured images -- the
calibration solve itself is out of scope for this package.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import cv2
import numpy as np

# AprilTag 36h11 is 6x6 data bits inside a 1-bit black border, so the black
# square that the physical 106 mm measurement refers to is 8 cells across.
BLACK_SQUARE_CELLS = 8
QUIET_ZONE_CELLS = 1


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


# --------------------------------------------------------------------------- #
# Gazebo assets
# --------------------------------------------------------------------------- #

def render_tag_image(tag_id: int = 0, cell_px: int = 64) -> np.ndarray:
    """Tag bitmap with a 1-cell white quiet zone around the black square."""
    marker = cv2.aruco.drawMarker(dictionary(), tag_id,
                                  BLACK_SQUARE_CELLS * cell_px)
    pad = QUIET_ZONE_CELLS * cell_px
    return cv2.copyMakeBorder(marker, pad, pad, pad, pad,
                              cv2.BORDER_CONSTANT, value=255)


def write_assets(directory: str, tag_id: int = 0, tag_size: float = 0.106,
                 cell_px: int = 64) -> dict:
    """Write ``.png``/``.obj``/``.mtl`` for a flat tag quad.

    The mesh spans the *padded* square (black square plus quiet zone) so that
    the black square itself measures exactly ``tag_size``.  The quad lies in the
    mesh XY plane with its normal along +Z, matching the tag frame used
    everywhere else in this package.
    """
    os.makedirs(directory, exist_ok=True)
    stem = f'apriltag_36h11_id{tag_id}'
    png_path = os.path.join(directory, f'{stem}.png')
    obj_path = os.path.join(directory, f'{stem}.obj')
    mtl_path = os.path.join(directory, f'{stem}.mtl')

    cv2.imwrite(png_path, render_tag_image(tag_id, cell_px))

    total_cells = BLACK_SQUARE_CELLS + 2 * QUIET_ZONE_CELLS
    quad = tag_size * total_cells / BLACK_SQUARE_CELLS
    half = quad / 2.0

    with open(mtl_path, 'w', encoding='utf-8') as handle:
        handle.write(
            f'newmtl {stem}\n'
            'Ka 1.0 1.0 1.0\n'
            'Kd 1.0 1.0 1.0\n'
            'Ks 0.0 0.0 0.0\n'
            'Ns 1.0\n'
            'illum 1\n'
            f'map_Kd {stem}.png\n')

    with open(obj_path, 'w', encoding='utf-8') as handle:
        handle.write(
            f'# AprilTag 36h11 id {tag_id}; black square = {tag_size * 1000:.1f} mm,\n'
            f'# quad including the 1-cell quiet zone = {quad * 1000:.1f} mm.\n'
            f'mtllib {stem}.mtl\n'
            f'o {stem}\n'
            f'v {-half:.6f} {-half:.6f} 0.0\n'
            f'v {half:.6f} {-half:.6f} 0.0\n'
            f'v {half:.6f} {half:.6f} 0.0\n'
            f'v {-half:.6f} {half:.6f} 0.0\n'
            'vt 0.0 0.0\nvt 1.0 0.0\nvt 1.0 1.0\nvt 0.0 1.0\n'
            'vn 0.0 0.0 1.0\n'
            f'usemtl {stem}\n'
            's off\n'
            'f 1/1/1 2/2/1 3/3/1 4/4/1\n')

    return {'png': png_path, 'obj': obj_path, 'mtl': mtl_path, 'quad_size': quad}


def make_assets_main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='meshes',
                        help='directory to write the assets into')
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--size', type=float, default=0.106,
                        help='black-square side length in metres')
    parser.add_argument('--cell-px', type=int, default=64)
    args = parser.parse_args(argv)
    written = write_assets(args.output, args.id, args.size, args.cell_px)
    print(f"wrote {written['png']}, {written['obj']}, {written['mtl']} "
          f"(quad {written['quad_size'] * 1000:.1f} mm)")
    return 0
