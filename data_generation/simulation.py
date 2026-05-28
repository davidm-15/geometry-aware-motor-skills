import numpy as np
from scipy.spatial.transform import Rotation as R
from .pure_pursuit import PurePursuitController

class VirtualEndEffector:
    def __init__(self, m=1.0, I=None, cv=2.0, comega=2.0, dt=0.01, lookahead_distance=5.0, kp_linear=10.0, kp_angular=10.0):
        self.m = m
        self.I = I if I is not None else np.eye(3)
        self.I_inv = np.linalg.inv(self.I)
        self.cv = cv
        self.comega = comega
        self.dt = dt
        self.controller = PurePursuitController(lookahead_distance=lookahead_distance, kp_linear=kp_linear, kp_angular=kp_angular)
        
    def reset(self, start_pos, start_quat):
        self.p = np.array(start_pos, dtype=float)
        self.q = np.array(start_quat, dtype=float)
        self.v = np.zeros(3)
        self.omega = np.zeros(3)
        self.time = 0.0
        
    def get_v_dot(self, v, f_ctrl):
        return (f_ctrl - self.cv * v) / self.m

    def get_omega_dot(self, omega, tau_ctrl):
        gyroscopic = np.cross(omega, self.I @ omega)
        return self.I_inv @ (tau_ctrl - self.comega * omega - gyroscopic)

    def step(self, f_ctrl, tau_ctrl, f_noise_std=0.0, tau_noise_std=0.0):
        # Inject process noise if specified
        if f_noise_std > 0.0:
            f_ctrl = f_ctrl + np.random.normal(0, f_noise_std, size=f_ctrl.shape)
        if tau_noise_std > 0.0:
            tau_ctrl = tau_ctrl + np.random.normal(0, tau_noise_std, size=tau_ctrl.shape)
            
        # Runge-Kutta 4th Order (RK4) Integration
        
        # 1. Linear Dynamics (RK4)
        kv1 = self.get_v_dot(self.v, f_ctrl)
        kv2 = self.get_v_dot(self.v + 0.5 * self.dt * kv1, f_ctrl)
        kv3 = self.get_v_dot(self.v + 0.5 * self.dt * kv2, f_ctrl)
        kv4 = self.get_v_dot(self.v + self.dt * kv3, f_ctrl)
        
        kp1 = self.v
        kp2 = self.v + 0.5 * self.dt * kv1
        kp3 = self.v + 0.5 * self.dt * kv2
        kp4 = self.v + self.dt * kv3
        
        self.v += (self.dt / 6.0) * (kv1 + 2 * kv2 + 2 * kv3 + kv4)
        self.p += (self.dt / 6.0) * (kp1 + 2 * kp2 + 2 * kp3 + kp4)
        
        # 2. Rotational Dynamics (RK4)
        ko1 = self.get_omega_dot(self.omega, tau_ctrl)
        ko2 = self.get_omega_dot(self.omega + 0.5 * self.dt * ko1, tau_ctrl)
        ko3 = self.get_omega_dot(self.omega + 0.5 * self.dt * ko2, tau_ctrl)
        ko4 = self.get_omega_dot(self.omega + self.dt * ko3, tau_ctrl)
        
        self.omega += (self.dt / 6.0) * (ko1 + 2 * ko2 + 2 * ko3 + ko4)
        
        # Quat Update via Rotvec
        r_omega = R.from_rotvec(self.omega * self.dt)
        r_current = R.from_quat(self.q)
        r_next = r_omega * r_current
        self.q = r_next.as_quat()
        
        self.time += self.dt
        
        return self.time, self.p.copy(), self.q.copy(), np.linalg.norm(self.v), np.linalg.norm(self.omega)
