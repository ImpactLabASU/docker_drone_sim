# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from numpy import sin, cos, tan

def phiThetaPsiDotToPQR(phi, theta, psi, phidot, thetadot, psidot):
    
    p = -sin(theta)*psidot + phidot
    
    q = sin(phi)*cos(theta)*psidot + cos(phi)*thetadot
    
    r = -sin(phi)*thetadot + cos(phi)*cos(theta)*psidot
    
    return np.array([p, q, r])


def xyzDotToUVW_euler(phi, theta, psi, xdot, ydot, zdot):
    u = xdot*cos(psi)*cos(theta) + ydot*sin(psi)*cos(theta) - zdot*sin(theta)
    
    v = (sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*ydot + (sin(phi)*sin(theta)*cos(psi) - sin(psi)*cos(phi))*xdot + zdot*sin(phi)*cos(theta)
    
    w = (sin(phi)*sin(psi) + sin(theta)*cos(phi)*cos(psi))*xdot + (-sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*ydot + zdot*cos(phi)*cos(theta)
    
    return np.array([u, v, w])


def xyzDotToUVW_Flat_euler(phi, theta, psi, xdot, ydot, zdot):
    uFlat = xdot * cos(psi) + ydot * sin(psi)

    vFlat = -xdot * sin(psi) + ydot * cos(psi)

    wFlat = zdot

    return np.array([uFlat, vFlat, wFlat])

def xyzDotToUVW_Flat_quat(q, xdot, ydot, zdot):
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    uFlat = 2*(q0*q3 - q1*q2)*ydot + (q0**2 - q1**2 + q2**2 - q3**2)*xdot

    vFlat = -2*(q0*q3 + q1*q2)*xdot + (q0**2 + q1**2 - q2**2 - q3**2)*ydot

    wFlat = zdot

    return np.array([uFlat, vFlat, wFlat])
    
    
def quaternion_to_euler(q1, q2, q3, q4):
    """Converts a quaternion (q1, q2, q3, q4) to Euler angles (roll, pitch, yaw)."""
    
    # Roll (phi)
    sin_phi = 2 * (q4 * q1 + q2 * q3)
    cos_phi = 1 - 2 * (q1**2 + q2**2)
    phi = np.arctan2(sin_phi, cos_phi)
    
    # Pitch (theta)
    sin_theta = 2 * (q4 * q2 - q3 * q1)
    theta = np.arcsin(np.clip(sin_theta, -1, 1))
    
    # Yaw (psi)
    sin_psi = 2 * (q4 * q3 + q1 * q2)
    cos_psi = 1 - 2 * (q2**2 + q3**2)
    psi = np.arctan2(sin_psi, cos_psi)
    
    return phi, theta, psi

def rotation_matrix(phi, theta, psi):
    """Computes the rotation matrix from world frame to body frame."""
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    c_psi, s_psi = np.cos(psi), np.sin(psi)
    
    R = np.array([
        [c_psi * c_theta, c_psi * s_theta * s_phi - s_psi * c_phi, c_psi * s_theta * c_phi + s_psi * s_phi],
        [s_psi * c_theta, s_psi * s_theta * s_phi + c_psi * c_phi, s_psi * s_theta * c_phi - c_psi * s_phi],
        [-s_theta, c_theta * s_phi, c_theta * c_phi]
    ])
    return R

def accelerometer_readings(xdd, ydd, zdd, q1, q2, q3, q4, g=9.81):
    """Converts world-frame accelerations to accelerometer readings in the body frame using quaternions."""
    
    # Convert quaternion to Euler angles
    phi, theta, psi = quaternion_to_euler(q1, q2, q3, q4)
    
    # Acceleration in world frame
    a_world = np.array([xdd, ydd, zdd])
    
    # Gravity vector in world frame
    g_world = np.array([0, 0, -g])
    
    # Compute rotation matrix from world to body frame
    R = rotation_matrix(phi, theta, psi)
    
    # Compute accelerometer readings in the body frame
    a_body = R.T @ (a_world - g_world)
    
    return a_body

def angular_rates_to_euler(p, q, r, q1, q2, q3, q4):
    """Converts body-frame angular rates (p, q, r) to Euler angle rates using quaternions."""
    
    # Convert quaternion to Euler angles
    phi, theta, _ = quaternion_to_euler(q1, q2, q3, q4)
    
    T = np.array([
        [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]
    ])
    
    euler_rates = T @ np.array([p, q, r])
    
    return euler_rates    
