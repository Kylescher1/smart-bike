#!/usr/bin/env python3
"""
Turret System - 3D Object Tracking

This script controls the turret system. Its goal is to use 3 cameras, 
2 servos and a Time of Flight sensor to detect objects, track them,
get depth and return 3D positions.

Structure:
- 2 static fisheye cameras (left/right) at z=0, pointing 30° outward
- 1 center camera on turret platform (active tracking)
- 1 ToF sensor for depth measurement
- Arduino controls servos via serial

The fisheye cameras scout for targets and tell the turret where to point.
The center camera actively tracks the selected target.
When locked on target, ToF sensor measures depth.
3D positions are computed and mapped to spherical coordinates (unit sphere).

Usage:
    from src.hal.Turret import Turret
    
    turret = Turret(
        port='COM3',
        cameras={'left': 0, 'right': 1, 'center': 2}
    )
    turret.start()
    
    # Read detections with 3D positions
    output = turret.read()
    for det3d in output.detections_3d:
        az, el, dist = det3d.position_spherical
        print(f"{det3d.detection.class_name} at azimuth={az:.1f}°, "
              f"elevation={el:.1f}°, distance={dist:.2f}in")
    
    turret.stop()

See turret_demo.py for complete example.
"""

# Import main Turret class
from .Turret import Turret, TurretOutput, TurretPose
from .TurretGeometry import TurretGeometry
from .TurretController import TurretController
from .PositionCalculator import Position3D

__all__ = [
    'Turret',
    'TurretOutput', 
    'TurretPose',
    'Position3D',
    'TurretGeometry',
    'TurretController'
]
