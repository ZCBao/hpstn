import numpy as np
from math import cos, sin, atan2
from env.utils.utils import norm, wraptopi

# Reference: Modern Robotics: Mechanics, Planning, and Control by Kevin M. Lynch and Frank C. Park
def motion_diff(state, vel, delta_t, noise=False, alpha=[0.01, 0.01]):
    # state: np.array([x, y, yaw])
    # vel: np.array([v, w])
    # alpha: noise parameters for [v, w]
    if noise:
        std_v = np.sqrt(alpha[0])
        std_w = np.sqrt(alpha[1])
        vel_noise = vel + np.random.normal([0, 0], scale=[std_v, std_w])
    else:
        vel_noise = vel

    v = vel_noise[0]
    w = vel_noise[1]
    yaw = state[2]

    if abs(w) < 1e-6:
        next_state = state + np.array([v * delta_t * cos(yaw), v * delta_t * sin(yaw), 0])
    else:
        ratio = v / w
        next_state = state + np.array([ratio * sin(yaw + w * delta_t) - ratio * sin(yaw),
                                      -ratio * cos(yaw + w * delta_t) + ratio * cos(yaw),
                                       w * delta_t])
    next_state[2] = float(wraptopi(next_state[2]))

    return next_state

def motion_omni(state, vel, delta_t, noise=False, alpha=[0.01, 0.01]):
    # state: np.array([x, y, yaw])
    # vel: np.array([vx, vy])
    # alpha: noise parameters for [vx, vy]
    if noise:
        std_vx = np.sqrt(alpha[0])
        std_vy = np.sqrt(alpha[1])
        vel_noise = vel + np.random.normal([0, 0], scale=[std_vx, std_vy])
    else:
        vel_noise = vel

    yaw = state[2]
    vx_abs = vel_noise[0] * cos(yaw) - vel_noise[1] * sin(yaw)
    vy_abs = vel_noise[0] * sin(yaw) + vel_noise[1] * cos(yaw)

    next_state = state + np.array([vx_abs * delta_t, vy_abs * delta_t, 0])
    next_state[2] = float(wraptopi(next_state[2]))

    return next_state

def motion_ros(state, vel, delta_t, noise=False, alpha=[0.01, 0.01, 0.01]):
    # state: np.array([x, y, yaw])
    # vel: np.array([vx, vy, wz])
    # alpha: noise parameters for [vx, vy, wz]
    if noise:
        std_vx = np.sqrt(alpha[0])
        std_vy = np.sqrt(alpha[1])
        std_wz = np.sqrt(alpha[2])
        vel_noise = vel + np.random.normal([0, 0, 0], scale=[std_vx, std_vy, std_wz])
    else:
        vel_noise = vel

    vx = vel_noise[0]
    vy = vel_noise[1]
    wz = vel_noise[2]
    yaw = state[2]

    # tangent line model
    next_state = state + np.array([vx * delta_t * cos(yaw) - vy * delta_t * sin(yaw),
                                   vx * delta_t * sin(yaw) + vy * delta_t * cos(yaw),
                                   wz * delta_t])
    
    # # cut line model
    # v = norm(np.array([vx, vy]))
    # w = wz
    # direction = yaw + w * delta_t / 2
    # next_state = state + np.array([v * delta_t * cos(direction), v * delta_t * sin(direction), w * delta_t])

    # # arc model
    # v = norm(np.array([vx, vy]))
    # w = wz
    # if abs(w) < 1e-6:
    #     next_state = state + np.array([v * delta_t * cos(yaw), v * delta_t * sin(yaw), 0])
    # else:
    #     ratio = v / w
    #     next_state = state + np.array([ratio * sin(yaw + w * delta_t) - ratio * sin(yaw),
    #                                   -ratio * cos(yaw + w * delta_t) + ratio * cos(yaw),
    #                                    w * delta_t])

    next_state[2] = float(wraptopi(next_state[2]))

    return next_state

def motion_abs(state, vel, delta_t, noise=False, alpha=[0.01, 0.01]):
    # state: np.array([x, y, yaw])
    # vel: np.array([vx, vy])
    # alpha: noise parameters for [vx, vy]
    if noise:
        std_vx = np.sqrt(alpha[0])
        std_vy = np.sqrt(alpha[1])
        vel_noise = vel + np.random.normal([0, 0], scale=[std_vx, std_vy])
    else:
        vel_noise = vel

    vx_abs = vel_noise[0]
    vy_abs = vel_noise[1]

    next_state = state + np.array([vx_abs * delta_t, vy_abs * delta_t, 0])
    next_state[2] = float(wraptopi(next_state[2]))

    return next_state