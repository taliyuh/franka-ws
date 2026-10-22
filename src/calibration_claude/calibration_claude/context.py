"""Shared setup: locating config files, resolving package:// URIs, building the chain."""

from __future__ import annotations

import os

import yaml

from .cameras import load_cameras
from .kinematics import Chain

PACKAGE = 'calibration_claude'


def share_directory() -> str:
    """Installed share directory, falling back to the source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory
        return get_package_share_directory(PACKAGE)
    except Exception:  # not built / not sourced -- run straight from source
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve(path: str) -> str:
    """Resolve a ``package://pkg/rest`` URI (or return *path* unchanged)."""
    if not path.startswith('package://'):
        return path
    package, _, rest = path[len('package://'):].partition('/')
    if package == PACKAGE:
        return os.path.join(share_directory(), rest)
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory(package), rest)


def default_config_path(name: str) -> str:
    return os.path.join(share_directory(), 'config', name)


def load_config(path: str | None = None) -> dict:
    with open(path or default_config_path('calibration.yaml'), 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def camera_config_path(path: str | None = None) -> str:
    return path or default_config_path('cameras_sim.yaml')


def load_camera_set(path: str | None = None) -> dict:
    """``{name: PinholeCamera}`` from a cameras yaml."""
    return load_cameras(camera_config_path(path))


def load_camera_specs(path: str | None = None) -> dict:
    """The raw yaml entries, which also carry the image/camera_info topic names."""
    with open(camera_config_path(path), 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)['cameras']


def build_chain(config: dict, urdf_xml: str | None = None) -> Chain:
    """Chain from a live ``/robot_description`` string, or by running xacro."""
    robot = config['robot']
    kinematic_base_link = robot.get('kinematic_base_link', robot['base_link'])
    if urdf_xml:
        return Chain.from_urdf(urdf_xml, robot['base_link'], robot['tip_link'])
    return Chain.from_xacro(resolve(robot['xacro']), kinematic_base_link, robot['tip_link'],
                            mappings=robot.get('xacro_mappings'))
