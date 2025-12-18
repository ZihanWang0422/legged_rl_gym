#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
统一的 Sim2Sim (MuJoCo) 和 Sim2Real (Unitree SDK) 控制脚本

支持两种模式:
1. Sim2Sim: MuJoCo 仿真环境 (--mode sim)
2. Sim2Real: Unitree Go1 真实机器人 (--mode real)

共享配置参数和策略推理逻辑

用法:
  仿真模式: python sim2sim_sim2real_unified.py --mode sim
  真机模式: python sim2sim_sim2real_unified.py --mode real
"""

import time
import math
import numpy as np
import argparse
import torch
import threading
import sys
import os
from gamepad_linux import F710GamePadLinux, apply_deadzone

# ========== 共享配置类 ==========
class UnifiedConfig:
    """统一配置参数 (sim 和 real 共享)"""
    
    # 控制频率
    # go1_amp_config: sim_dt - 0.005  decimation 6 -> policy_dt 0.03 -> policy_hz 33Hz
    sim_dt = 0.005       # 仿真/控制时间步长 (200Hz)
    policy_hz = 33       # 策略推理频率 (Hz)
    policy_dt = 1.0 / policy_hz
    
    # 默认站立角度 (训练环境顺序: FL, FR, RL, RR)
    default_dof_pos = np.array([
        0.0, 0.9, -1.8,   # FL (hip, thigh, calf)
        0.0, 0.9, -1.8,   # FR
        0.0, 0.9, -1.8,   # RL
        0.0, 0.9, -1.8    # RR
    ], dtype=np.float32)
    
    # 观测缩放因子 (与训练环境一致)
    obs_scales = {
        'lin_vel': 2.0,
        'ang_vel': 0.25,
        'dof_pos': 1.0,
        'dof_vel': 0.05,
        'commands': np.array([2.0, 2.0, 0.25], dtype=np.float32),  # [lin_vel_x, lin_vel_y, ang_vel_yaw]
    }
    
    # 动作缩放
    action_scale = 0.25
    
    # 观测/动作裁剪
    clip_observations = 100.0
    clip_actions = 100.0
    
    # PD 增益
    kp_stand = 60.0      # 站立阶段
    kd_stand = 2.0
    kp_walk = 60.0       # 行走阶段
    kd_walk = 1.0
    
    # 关节限位 (训练环境顺序: FL, FR, RL, RR)
    joint_limit_low = np.array([
        -0.8, -1.0, -2.7,   # FL
        -0.8, -1.0, -2.7,   # FR
        -0.8, -1.0, -2.7,   # RL
        -0.8, -1.0, -2.7    # RR
    ], dtype=np.float32)
    
    joint_limit_high = np.array([
        0.8, 2.5, -0.9,    # FL
        0.8, 2.5, -0.9,    # FR
        0.8, 2.5, -0.9,    # RL
        0.8, 2.5, -0.9     # RR
    ], dtype=np.float32)
    
    # 站立/稳定阶段时间
    standup_duration = 2.0     # 站立阶段 (秒)
    stabilize_duration = 0.5   # 稳定阶段 (秒)
    
    # 速度命令范围
    vx_range = (0.0, 1.17)       # m/s (只能前进,不支持后退)
    vy_range = (-0.3, 0.3)      # m/s
    vyaw_range = (-1.57, 1.57)  # rad/s
    
    # Sim2Real 特定配置
    robot_ip = "192.168.123.10"
    robot_port = 8007
    local_port = 8080
    
    # 关节映射：训练环境顺序 (FL, FR, RL, RR) -> SDK 顺序 (FR, FL, RR, RL)
    train_to_sdk_map = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
    sdk_to_train_map = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]


# ========== 辅助函数 ==========

def quat_from_euler_xyz(roll, pitch, yaw):
    """从欧拉角计算四元数 [x, y, z, w]"""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([x, y, z, w], dtype=np.float32)

def quat_rotate_inverse(q, v):
    """将向量从世界坐标系旋转到机体坐标系"""
    q_w = q[3]
    q_vec = q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

def quat_to_euler_xyz(q):
    """将四元数 [x, y, z, w] 转换为欧拉角 [roll, pitch, yaw]"""
    x, y, z, w = q
    
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw], dtype=np.float32)

def compute_projected_gravity(quat):
    """计算投影重力向量"""
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    projected_gravity = quat_rotate_inverse(quat, gravity_world)
    return projected_gravity

def build_obs_45(base_ang_vel, projected_gravity, commands, dof_pos, dof_vel, last_action, config):
    """
    构建 45 维观测向量:
    1-3: base_ang_vel [wx, wy, wz] * ang_vel_scale
    4-6: projected_gravity [gx, gy, gz]
    7-9: commands [lin_vel_x, lin_vel_y, ang_vel_yaw] * commands_scale
    10-21: (dof_pos - default_dof_pos) * dof_pos_scale
    22-33: dof_vel * dof_vel_scale
    34-45: last_actions
    """
    obs = []
    
    # 1-3: Base angular velocity (scaled)
    obs.extend(list(base_ang_vel * config.obs_scales['ang_vel']))
    
    # 4-6: Projected gravity
    obs.extend(list(projected_gravity))
    
    # 7-9: Commands (scaled)
    commands_scaled = commands * config.obs_scales['commands']
    obs.extend(list(commands_scaled))
    
    # 10-21: dof_pos - default_dof_pos (scaled)
    pos_delta = (dof_pos - config.default_dof_pos) * config.obs_scales['dof_pos']
    obs.extend(list(pos_delta))
    
    # 22-33: dof_vel (scaled)
    obs.extend(list(dof_vel * config.obs_scales['dof_vel']))
    
    # 34-45: Last action
    obs.extend(list(last_action))
    
    return np.array(obs, dtype=np.float32)

def normalize_obs(obs, clip_value=100.0):
    """裁剪观测值"""
    return np.clip(obs, -clip_value, clip_value)


# ========== 手柄控制器 ==========

class GamepadController:
    """线程安全的手柄控制器 (Logitech F710 - Linux 原生接口)"""
    def __init__(self, vx_range=(0.0, 1.2), vy_range=(-0.3, 0.3), vyaw_range=(-1.57, 1.57)):
        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0
        self.vx_range = vx_range
        self.vy_range = vy_range
        self.vyaw_range = vyaw_range
        self.lock = threading.Lock()
        self.running = True
        self.exit_requested = False
        self.thread = None
        
        # 初始化手柄 (直接使用 Linux 设备)
        try:
            self.gamepad = F710GamePadLinux()
            self.gamepad.start()
            print("✅ Gamepad initialized successfully (Linux native)")
        except Exception as e:
            print(f"❌ Failed to initialize gamepad: {e}")
            self.gamepad = None
        
        # 死区设置 (归一化值)
        self.deadzone = 0.05  # 5% 死区
        
        # 速度平滑参数 (指数移动平均)
        self.alpha = 0.6  # 平滑系数: 60%新值+40%旧值 (提高响应速度)
        self.vx_smooth = 0.0
        self.vy_smooth = 0.0
        self.vyaw_smooth = 0.0
        
        # 速度档位控制 (D-pad增量调节)
        self.vx_increment = 0.1  # 每次按键增加/减少0.1 m/s
        self.vx_target = 0.0     # 目标速度档位
        self.dpad_last_state = {'up': False, 'down': False}  # 防止连续触发
    
    def get_velocity(self):
        with self.lock:
            return self.vx, self.vy, self.vyaw
    
    def set_velocity(self, vx, vy, vyaw):
        with self.lock:
            self.vx = np.clip(vx, self.vx_range[0], self.vx_range[1])
            self.vy = np.clip(vy, self.vy_range[0], self.vy_range[1])
            self.vyaw = np.clip(vyaw, self.vyaw_range[0], self.vyaw_range[1])
    
    def gamepad_thread(self):
        """手柄读取线程 - 与策略频率同步"""
        if self.gamepad is None:
            print("Gamepad not available, using zero velocity")
            return
        
        last_print_time = time.time()
        
        # 与策略频率同步: 33Hz
        update_interval = 1.0 / 33.0  # 0.0303秒
        
        while self.running:
            try:
                loop_start = time.time()
                
                # 获取摇杆值 (归一化到 [-1, 1])
                left_x, left_y = self.gamepad.get_left_stick(normalize=True)
                right_x, right_y = self.gamepad.get_right_stick(normalize=True)
                
                # 应用死区
                left_x = apply_deadzone(left_x, self.deadzone)
                left_y = apply_deadzone(left_y, self.deadzone)
                right_x = apply_deadzone(right_x, self.deadzone)
                
                # D-pad按键增量控制 (HAT轴: 6=X轴, 7=Y轴)
                with self.gamepad.lock:
                    dpad_y = self.gamepad.axes[7] if len(self.gamepad.axes) > 7 else 0  # Y轴: -32767=上, +32767=下
                
                dpad_up_pressed = (dpad_y < -16000)    # 上
                dpad_down_pressed = (dpad_y > 16000)   # 下
                
                # 边缘检测：只在按键从未按下->按下时触发
                if dpad_up_pressed and not self.dpad_last_state['up']:
                    self.vx = min(self.vx + self.vx_increment, self.vx_range[1])
                    print(f"\n[D-pad UP] 速度档位: {self.vx:.1f} m/s")
                
                if dpad_down_pressed and not self.dpad_last_state['down']:
                    self.vx = max(self.vx - self.vx_increment, 0.0)  # 最小0，不后退
                    print(f"\n[D-pad DOWN] 速度档位: {self.vx:.1f} m/s")
                
                # 更新按键状态
                self.dpad_last_state['up'] = dpad_up_pressed
                self.dpad_last_state['down'] = dpad_down_pressed
                
                # 映射到速度
                # 优先使用D-pad档位速度，摇杆可作为微调
                # 左摇杆 Y 轴: -1(向上推) ~ +1(向下推)
                if abs(left_y) > 0.1:  # 摇杆有明显输入时，使用摇杆控制
                    if left_y <= 0:  # 向上推 (Y为负值)
                        self.vx = (-left_y) * self.vx_range[1]  # 转为正值并映射: 0 ~ 1.2 m/s
                    else:  # 向下推 (Y为正值)
                        self.vx = 0.0  # 不支持后退
                else:  # 摇杆归中，使用D-pad档位速度
                    self.vx = self.vx
                
                # 左摇杆 X 轴: -1(向左推) ~ +1(向右推)
                # 速度映射: 向右推=右移(+vy), 向左推=左移(-vy)
                self.vy = -left_x * (self.vy_range[1])   # 直接映射: -0.3 ~ +0.3 m/s
                
                # 右摇杆 X 轴: -1(向左推) ~ +1(向右推)
                # 速度映射: 向左推=左转(+vyaw), 向右推=右转(-vyaw)
                self.vyaw = -right_x * self.vyaw_range[1]  # 反向映射
                
                
                # 更新速度 (使用平滑后的值)
                self.set_velocity(self.vx, self.vy, self.vyaw)
                
                # 检查退出按钮 (Start = 按钮 7)
                if self.gamepad.is_button_pressed(self.gamepad.BTN_START):
                    print("\n✅ Start button pressed - exiting")
                    self.exit_requested = True
                    break
                
                # 每0.5秒打印一次当前速度
                current_time = time.time()
                if current_time - last_print_time > 0.5:
                    vx_cur, vy_cur, vyaw_cur = self.get_velocity()
                    mode = "档位" if abs(left_y) <= 0.1 else "摇杆"
                    print(f"\r[Gamepad] {mode}: vx={vx_cur:+.2f} m/s | vy={vy_cur:+.2f} | yaw={vyaw_cur:+.2f} rad/s", end='', flush=True)
                    last_print_time = current_time
                
                # 与策略频率同步: 33Hz (每0.03秒更新一次)
                elapsed = time.time() - loop_start
                sleep_time = max(0, update_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                print(f"\nGamepad error: {e}")
                time.sleep(0.1)
    
    def start(self):
        self.thread = threading.Thread(target=self.gamepad_thread, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.gamepad:
            self.gamepad.stop()
        if self.thread:
            self.thread.join(timeout=1.0)


# ========== Sim2Sim (MuJoCo) 控制器 ==========

class Sim2SimController:
    """MuJoCo 仿真控制器"""
    
    # MuJoCo 模型中的关节和执行器名称 (FL, FR, RL, RR 顺序)
    JOINT_NAMES = [
        'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
        'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
        'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
        'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
    ]
    
    ACTUATOR_NAMES = [
        'FL_hip', 'FL_thigh', 'FL_calf',
        'FR_hip', 'FR_thigh', 'FR_calf',
        'RL_hip', 'RL_thigh', 'RL_calf',
        'RR_hip', 'RR_thigh', 'RR_calf',
    ]
    
    def __init__(self, config, xml_path, policy_path):
        self.config = config
        
        # 加载 MuJoCo 模型
        import mujoco
        import mujoco.viewer
        self.mujoco = mujoco
        self.mujoco_viewer = mujoco.viewer
        
        print(f"Loading MuJoCo model: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = config.sim_dt
        
        # 获取关节和执行器索引
        self.joint_qpos_addrs = []
        self.joint_dof_addrs = []
        self.actuator_ids = []
        
        for joint_name, actuator_name in zip(self.JOINT_NAMES, self.ACTUATOR_NAMES):
            joint_id = self.model.joint(joint_name).id
            qpos_addr = self.model.jnt_qposadr[joint_id]
            dof_addr = self.model.jnt_dofadr[joint_id]
            actuator_id = self.model.actuator(actuator_name).id
            
            self.joint_qpos_addrs.append(qpos_addr)
            self.joint_dof_addrs.append(dof_addr)
            self.actuator_ids.append(actuator_id)
        
        # 加载策略
        print(f"Loading policy: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location='cpu')
        self.policy.eval()
        
        # 初始化状态
        self.last_action = np.zeros(12, dtype=np.float32)
        self.qDes = np.zeros(12, dtype=np.float32)
        
        # 策略频率控制
        self.policy_decimation = int(config.policy_dt / config.sim_dt)
        self.policy_counter = 0
        
        # 初始化机器人位置
        for i, qpos_addr in enumerate(self.joint_qpos_addrs):
            self.data.qpos[qpos_addr] = config.default_dof_pos[i]
        self.data.qpos[2] = 0.35  # 初始高度
        mujoco.mj_forward(self.model, self.data)
        
        print(f"Sim2Sim controller initialized")
        
    
    def get_state(self):
        """获取当前状态"""
        # Base angular velocity
        base_ang_vel = self.data.qvel[3:6].copy()
        
        # Projected gravity
        quat_wxyz = self.data.qpos[3:7].copy()
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
        projected_gravity = compute_projected_gravity(quat_xyzw)
        
        # Joint positions and velocities
        dof_pos = np.array([self.data.qpos[addr] for addr in self.joint_qpos_addrs], dtype=np.float32)
        dof_vel = np.array([self.data.qvel[addr] for addr in self.joint_dof_addrs], dtype=np.float32)
        
        return base_ang_vel, projected_gravity, dof_pos, dof_vel
    
    def send_command(self, target_pos):
        """发送控制命令"""
        for i, actuator_id in enumerate(self.actuator_ids):
            self.data.ctrl[actuator_id] = target_pos[i]
    
    def step(self):
        """执行一步仿真"""
        self.mujoco.mj_step(self.model, self.data)
    
    def run(self, gamepad):
        """运行仿真循环"""
        motiontime = 0
        start_time = time.time()  # 记录开始时间
        
        while True:
            
            motiontime += 1
            sim_time = motiontime * self.config.policy_dt
            loop_start = time.time()  # 记录循环开始时间
                
            if gamepad.exit_requested:
                print("\nExit request detected, ending simulation...")
                break
                
            self._control_step(sim_time, gamepad)
            self.step()
            real_time = time.time() - start_time
                
            if motiontime % int(1.0 / self.config.sim_dt) == 0:
                print(f"Sim time: {sim_time:.1f}s, Base height: {self.data.qpos[2]:.3f}m")
                # 每秒打印一次循环状态（简化,避免阻塞）
                
                
            # 精确的时间控制：补偿执行时间
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.config.sim_dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)    
            # 可视化模式
            with self.mujoco_viewer.launch_passive(self.model, self.data) as viewer:
                viewer.cam.lookat[:] = self.data.qpos[:3]
                viewer.cam.distance = 2.0
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -20
                
                while viewer.is_running():
                    motiontime += 1
                    sim_time = motiontime * self.config.sim_dt
                    loop_start = time.time()  # 记录循环开始时间
                    
                    if gamepad.exit_requested:
                        print("\nExit request detected, ending simulation...")
                        break
                    
                    self._control_step(sim_time, gamepad)
                    self.step()
                    real_time = time.time() - start_time
                    
                    viewer.cam.lookat[:] = self.data.qpos[:3]
                    viewer.sync()
                    
                    if motiontime % int(1.0 / self.config.sim_dt) == 0:
                        print(f"[Sim time]: t={sim_time:.1f}s, Base height: {self.data.qpos[2]:.3f}m")
                    if motiontime % int(1.0 / self.config.sim_dt) == 0:
                        actual_hz = motiontime / real_time if real_time > 0 else 0
                        print(f"[Real Time]: t={real_time:.1f}s, Actual Hz: {actual_hz:.2f} Hz")  
                    
                    elapsed = time.time() - loop_start
                    sleep_time = max(0, self.config.sim_dt - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)      
    
    def _control_step(self, sim_time, gamepad):
        """单步控制逻辑"""
        # Phase 1: Stand up
        if sim_time <= self.config.standup_duration:
            rate = min(sim_time / self.config.standup_duration, 1.0)
            for i, qpos_addr in enumerate(self.joint_qpos_addrs):
                current_q = self.data.qpos[qpos_addr]
                self.qDes[i] = current_q * (1 - rate) + self.config.default_dof_pos[i] * rate
            self.send_command(self.qDes)
        
        # Phase 2: Stabilize
        elif sim_time <= self.config.standup_duration + self.config.stabilize_duration:
            self.qDes = self.config.default_dof_pos.copy()
            self.send_command(self.qDes)
        
        # Phase 3: Policy control
        else:
            # Check tilt
            quat_wxyz = self.data.qpos[3:7].copy()
            quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
            rpy = quat_to_euler_xyz(quat_xyzw)
            if abs(rpy[0]) > 0.8 or abs(rpy[1]) > 0.8:
                print(f"\nWarning at {sim_time:.2f}s: Robot tilted! roll={rpy[0]:.2f}, pitch={rpy[1]:.2f}")
            
            # Policy inference
            self.policy_counter += 1
            if self.policy_counter >= self.policy_decimation:
                self.policy_counter = 0
                
                # Get commands
                cmd_vx, cmd_vy, cmd_vyaw = gamepad.get_velocity()
                commands = np.array([cmd_vx, cmd_vy, cmd_vyaw], dtype=np.float32)
                
                # Get state
                base_ang_vel, projected_gravity, dof_pos, dof_vel = self.get_state()
                
                # Build observation
                obs = build_obs_45(base_ang_vel, projected_gravity, commands, 
                                 dof_pos, dof_vel, self.last_action, self.config)
                obs = normalize_obs(obs, self.config.clip_observations)
                obs_batch = obs[np.newaxis, :].astype(np.float32)
                
                # Policy inference
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs_batch)
                    action_tensor = self.policy(obs_tensor)
                    if isinstance(action_tensor, tuple):
                        action_tensor = action_tensor[0]
                    action = action_tensor.cpu().numpy().flatten().astype(np.float32)
                
                # Scale action
                action = np.clip(action, -self.config.clip_actions, self.config.clip_actions)
                self.last_action = action[:12].copy()
                
                self.qDes = action[:12] * self.config.action_scale + self.config.default_dof_pos
                self.qDes = np.clip(self.qDes, self.config.joint_limit_low, self.config.joint_limit_high)
            
                
            self.send_command(self.qDes)


# ========== Sim2Real (Unitree SDK) 控制器 ==========

class Sim2RealController:
    """Unitree Go1 真机控制器"""
    
    def __init__(self, config, policy_path):
        self.config = config
        
        # 导入 SDK
        SDK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'unitree_legged_sdk', 'lib', 'python', 'amd64'))
        sys.path.append(SDK_DIR)
        import robot_interface as sdk
        self.sdk = sdk
        
        # SDK 关节索引
        self.d = {
            'FR_0':0, 'FR_1':1, 'FR_2':2,
            'FL_0':3, 'FL_1':4, 'FL_2':5,
            'RR_0':6, 'RR_1':7, 'RR_2':8,
            'RL_0':9, 'RL_1':10,'RL_2':11
        }
        self.joint_order = ['FR_0','FR_1','FR_2',
                           'FL_0','FL_1','FL_2',
                           'RR_0','RR_1','RR_2',
                           'RL_0','RL_1','RL_2']
        
        # 初始化 UDP
        LOWLEVEL = 0xff
        self.udp = sdk.UDP(LOWLEVEL, config.local_port, 
                          config.robot_ip, config.robot_port)
        self.low_cmd = sdk.LowCmd()
        self.low_state = sdk.LowState()
        self.udp.InitCmdData(self.low_cmd)
        
        print(f"UDP initialized: {config.robot_ip}:{config.robot_port}")
        
        # 加载策略
        print(f"Loading policy: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location='cpu')
        self.policy.eval()
        
        # 初始化状态
        self.last_action = np.zeros(12, dtype=np.float32)
        self.qDes_train = np.zeros(12, dtype=np.float32)
        
        # 策略频率控制
        self.policy_decimation = int(config.policy_dt / config.sim_dt)
        self.policy_counter = 0
        
        # 🔧 关键: 缓存最新的SDK命令,用于200Hz恒定发送
        self.current_qDes_sdk = None
        self.current_kp = config.kp_walk
        self.current_kd = config.kd_walk
        
        print("Sim2Real controller initialized")
    
    def wait_for_connection(self):
        """等待机器人连接"""
        print("Waiting for robot connection...")
        
        # 初始化命令（阻尼模式，Kp=0）
        for i in range(12):
            self.low_cmd.motorCmd[i].q = 0.0
            self.low_cmd.motorCmd[i].dq = 0.0
            self.low_cmd.motorCmd[i].Kp = 0.0
            self.low_cmd.motorCmd[i].Kd = 3.0
            self.low_cmd.motorCmd[i].tau = 0.0
        
        # 先发送命令，激活通信
        for i in range(100):
            self.udp.Recv()
            self.udp.GetRecv(self.low_state)
            self.udp.SetSend(self.low_cmd)
            self.udp.Send()
            time.sleep(self.config.sim_dt)
        
        # 检查是否收到有效数据
        q_sum = sum(abs(self.low_state.motorState[i].q) for i in range(12))
        if q_sum < 0.01:
            print("Error: No valid joint data received!")
            print("Please check:")
            print("  1. Robot is powered on")
            print("  2. Network connection is working")
            print("  3. IP address is correct (current: {})".format(self.config.robot_ip))
            return False
        
        print("Robot connected successfully!")
        self._print_state()
        return True
    
    def _print_state(self):
        """打印机器人状态"""
        print("\nCurrent joint angles (SDK order: FR, FL, RR, RL):")
        for leg in ['FR', 'FL', 'RR', 'RL']:
            hip = self.low_state.motorState[self.d[f'{leg}_0']].q
            thigh = self.low_state.motorState[self.d[f'{leg}_1']].q
            calf = self.low_state.motorState[self.d[f'{leg}_2']].q
            print(f"  {leg}: hip={hip:+.3f}, thigh={thigh:+.3f}, calf={calf:+.3f}")
        
        rpy = self.low_state.imu.rpy
        print(f"IMU: roll={rpy[0]:+.3f}, pitch={rpy[1]:+.3f}, yaw={rpy[2]:+.3f}")
    
    def get_state(self):
        """获取当前状态 (仅格式转换,不进行UDP接收)
        
        注意: 调用前需确保 run() 已经更新了 self.low_state
        """
        # Base angular velocity (SDK format)
        base_ang_vel = np.array([
            self.low_state.imu.gyroscope[0],
            self.low_state.imu.gyroscope[1],
            self.low_state.imu.gyroscope[2]
        ], dtype=np.float32)
        
        # Projected gravity from IMU
        rpy = np.array(self.low_state.imu.rpy, dtype=np.float32)
        quat = quat_from_euler_xyz(rpy[0], rpy[1], rpy[2])
        projected_gravity = compute_projected_gravity(quat)
        
        # Joint positions and velocities (SDK -> training order)
        q_sdk = np.array([self.low_state.motorState[i].q for i in range(12)], dtype=np.float32)
        dq_sdk = np.array([self.low_state.motorState[i].dq for i in range(12)], dtype=np.float32)
        
        dof_pos = q_sdk[self.config.sdk_to_train_map]
        dof_vel = dq_sdk[self.config.sdk_to_train_map]
        
        return base_ang_vel, projected_gravity, dof_pos, dof_vel
    
    def send_command(self, target_sdk, kp, kd):
        """发送控制命令
        
        Args:
            target_sdk: 目标关节角度 (SDK顺序, 12维数组)
            kp: PD控制比例增益
            kd: PD控制微分增益
        """
        # 设置电机命令
        for i, jname in enumerate(self.joint_order):
            self.low_cmd.motorCmd[self.d[jname]].q = float(target_sdk[i])
            self.low_cmd.motorCmd[self.d[jname]].dq = 0.0
            self.low_cmd.motorCmd[self.d[jname]].Kp = float(kp)
            self.low_cmd.motorCmd[self.d[jname]].Kd = float(kd)
            self.low_cmd.motorCmd[self.d[jname]].tau = 0.0
        
        # 发送命令 (200Hz让udp保持通讯)
        self.udp.SetSend(self.low_cmd)
        self.udp.Send()
    
    def run(self, gamepad):
        """运行真机控制循环"""
        if not self.wait_for_connection():
            print("Failed to connect to robot!")
            return
        
        print("\n" + "="*70)
        print("Starting real robot control...")
        print("⚠️  CAUTION: Robot will start moving after standup phase!")
        print("    Press Start button on gamepad to emergency stop")
        print("="*70 + "\n")
        
        motiontime = 0
        start_time = time.time()  # 记录开始时间
        
        while True:
            loop_start = time.time()  # 记录循环开始时间
            
            # ⚠️ 先接收low_state
            self.udp.Recv()
            self.udp.GetRecv(self.low_state)
            
            if gamepad.exit_requested:
                print("\nEmergency stop requested!")
                # Send damping command
                for i in range(12):
                    self.low_cmd.motorCmd[i].q = 0.0
                    self.low_cmd.motorCmd[i].dq = 0.0
                    self.low_cmd.motorCmd[i].Kp = 0.0
                    self.low_cmd.motorCmd[i].Kd = 6.0
                    self.low_cmd.motorCmd[i].tau = 0.0
                self.udp.SetSend(self.low_cmd)
                self.udp.Send()
                break
            
            # 使用实际时间而非累加计数
            real_time = time.time() - start_time
            motiontime += 1
            
            # 🔧 传入motiontime用于控制打印频率
            self._control_step(real_time, motiontime, gamepad)
            
            # 每秒打印一次循环状态（简化,避免阻塞）
            if motiontime % int(1.0 / self.config.sim_dt) == 0:
                actual_hz = motiontime / real_time if real_time > 0 else 0
                print(f"[Loop] t={real_time:.1f}s | count={motiontime} | Hz={actual_hz:.1f}")
            
            # 精确的时间控制：补偿执行时间
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.config.sim_dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _control_step(self, sim_time, motiontime, gamepad):
        """单步控制逻辑 (200Hz高频调用)"""
        # 对之前获取的low_state先进行格式转换成obs
        base_ang_vel, projected_gravity, dof_pos, dof_vel = self.get_state()
        rpy = np.array(self.low_state.imu.rpy, dtype=np.float32)
        
        # Phase 1: Stand up
        if sim_time <= self.config.standup_duration:
            rate = min(sim_time / self.config.standup_duration, 1.0)
            self.qDes_train = dof_pos * (1 - rate) + self.config.default_dof_pos * rate
            
            # 更新缓存命令 (SDK顺序)
            self.current_qDes_sdk = self.qDes_train[self.config.train_to_sdk_map]
            self.current_kp = self.config.kp_stand
            self.current_kd = self.config.kd_stand
        
        # Phase 2: Stabilize
        elif sim_time <= self.config.standup_duration + self.config.stabilize_duration:
            self.qDes_train = self.config.default_dof_pos.copy()
            
            # 更新缓存命令 (SDK顺序)
            self.current_qDes_sdk = self.qDes_train[self.config.train_to_sdk_map]
            self.current_kp = self.config.kp_walk
            self.current_kd = self.config.kd_walk
        
        # Phase 3: Policy control
        else:
            # Check tilt
            if abs(rpy[0]) > 0.8 or abs(rpy[1]) > 0.8:
                print("\n⚠️  WARNING: Robot tilted!")
                print(f"roll={rpy[0]:.2f}, pitch={rpy[1]:.2f}")
            
            # Policy inference (33Hz: 每6个sim_dt执行一次)
            self.policy_counter += 1
            if self.policy_counter >= self.policy_decimation:
                self.policy_counter = 0
                
                # Get commands from gamepad
                cmd_vx, cmd_vy, cmd_vyaw = gamepad.get_velocity()
                commands = np.array([cmd_vx, cmd_vy, cmd_vyaw], dtype=np.float32)
                
                # Transform state
                base_ang_vel, projected_gravity, dof_pos, dof_vel = self.get_state()
                
                # Build observation
                obs = build_obs_45(base_ang_vel, projected_gravity, commands,
                                 dof_pos, dof_vel, self.last_action, self.config)
                obs = normalize_obs(obs, self.config.clip_observations)
                obs_batch = obs[np.newaxis, :].astype(np.float32)
                
                # Policy inference
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs_batch)
                    action_tensor = self.policy(obs_tensor)
                    if isinstance(action_tensor, tuple):
                        action_tensor = action_tensor[0]
                    action = action_tensor.cpu().numpy().flatten().astype(np.float32)
                
                # Scale action
                action = np.clip(action, -self.config.clip_actions, self.config.clip_actions)
                self.last_action = action[:12].copy()
                
                self.qDes_train = action[:12] * self.config.action_scale + self.config.default_dof_pos
                self.qDes_train = np.clip(self.qDes_train, self.config.joint_limit_low, self.config.joint_limit_high)
                
                # 🔧 更新缓存命令 (这会在接下来的6帧中被重复发送)
                self.current_qDes_sdk = self.qDes_train[self.config.train_to_sdk_map]
                self.current_kp = self.config.kp_walk
                self.current_kd = self.config.kd_walk
                
                # 🔧 减少打印频率: 每秒打印2次 (避免I/O阻塞导致的抖动)
                if motiontime % 100 == 0:  # 200Hz / 100 = 2Hz
                    print(f"\n{'='*80}")
                    print(f"[t={sim_time:.2f}s | Policy@33Hz | Cmd@200Hz]")
                    print(f"{'='*80}")
                    print(f"🎮 GAMEPAD: vx={cmd_vx:+.3f} m/s  |  vy={cmd_vy:+.3f} m/s  |  vyaw={cmd_vyaw:+.3f} rad/s")
                    print(f"🤖 LOWSTATE: q_FL=[{dof_pos[0]:+.3f}, {dof_pos[1]:+.3f}, {dof_pos[2]:+.3f}]  |  "
                          f"q_FR=[{dof_pos[3]:+.3f}, {dof_pos[4]:+.3f}, {dof_pos[5]:+.3f}]  |  "
                          f"IMU_rpy=[{rpy[0]:+.3f}, {rpy[1]:+.3f}, {rpy[2]:+.3f}]")
                    print(f"🧠 POLICY  : qDes_FL=[{self.qDes_train[0]:+.3f}, {self.qDes_train[1]:+.3f}, {self.qDes_train[2]:+.3f}]  |  "
                          f"qDes_FR=[{self.qDes_train[3]:+.3f}, {self.qDes_train[4]:+.3f}, {self.qDes_train[5]:+.3f}]  |  "
                          f"action_norm={np.linalg.norm(self.last_action):.3f}")
        
        # ✅ 关键修复: 无论是否推理,每个200Hz循环都发送命令
        if self.current_qDes_sdk is not None:
            self.send_command(self.current_qDes_sdk, self.current_kp, self.current_kd)
        else:
            # 初始化阶段: 发送阻尼模式 (避免电机通电时突然动作)
            damping_cmd = np.zeros(12, dtype=np.float32)
            self.send_command(damping_cmd, 0.0, 3.0)


# ========== 主函数 ==========

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Unified Sim2Sim/Sim2Real Controller')
    parser.add_argument('--mode', type=str, default='sim', choices=['sim', 'real'],
                       help='Control mode: sim (MuJoCo) or real (Unitree SDK)')
    parser.add_argument('--model', type=str, default='policy_45_continus.pt',
                       help='PyTorch JIT model file (.pt)')
    parser.add_argument('--xml', type=str, default='scene.xml',
                       help='MuJoCo XML model file (sim mode only)')
    args = parser.parse_args()
    
    # 创建统一配置
    config = UnifiedConfig()
    
    # 创建手柄控制器
    gamepad = GamepadController(
        vx_range=config.vx_range,
        vy_range=config.vy_range,
        vyaw_range=config.vyaw_range
    )
    gamepad.start()
    
    print("\n" + "="*70)
    print("🎮 Gamepad Control (Logitech F710)")
    print("="*70)
    print("  Left Joystick:")
    print("    - Up/Down: Forward/Backward speed (vx)")
    print("    - Left/Right: Strafe speed (vy)")
    print("  Right Joystick:")
    print("    - Left/Right: Turn speed (vyaw)")
    print("  Start Button: Exit program")
    print("  Note: Release joystick to stop immediately")
    print("="*70 + "\n")
    
    # 根据模式创建控制器
    if args.mode == 'sim':
        print("Mode: Sim2Sim (MuJoCo)")
        
        # 路径设置（相对于本脚本的 deploy/ 目录）
        script_dir = os.path.dirname(__file__)
        assets_dir = os.path.abspath(os.path.join(script_dir, 'assets', 'go1'))
        xml_path = os.path.join(assets_dir, args.xml)

        policy_dir = os.path.abspath(os.path.join(script_dir, 'exported_policy', 'go1'))
        policy_path = os.path.join(policy_dir, args.model)
        
        controller = Sim2SimController(config, xml_path, policy_path)
        controller.run(gamepad)
        
    else:  # args.mode == 'real'
        print("Mode: Sim2Real (Unitree SDK)")
        
        # 路径设置（相对于本脚本的 deploy/ 目录）
        script_dir = os.path.dirname(__file__)
        policy_dir = os.path.abspath(os.path.join(script_dir, 'exported_policy', 'go1'))
        policy_path = os.path.join(policy_dir, args.model)
        
        controller = Sim2RealController(config, policy_path)
        controller.run(gamepad)
    
    # 停止手柄控制器
    gamepad.stop()
    print("\nProgram ended.")