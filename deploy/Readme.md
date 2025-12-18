# Go1 sim2sim2real

## sim2sim

### 运行命令
进入 `deploy` 目录，运行：
```bash
cd ./isaacgym/AMP_for_hardware/deploy/
conda activate amp_hw
# Use keyboard to control
python sim2sim2real_keyboard.py --mode sim --model policy_45_continus.pt  #play in mujoco
python sim2sim2real_keyboard.py --mode real --model policy_45_continus.pt #play in real-world
# Use joystick to control
python sim2sim2real_joystick.py --mode sim --model policy_45_continus.pt  #play in mujoco
python sim2sim2real_joystick.py --mode real --model policy_45_continus.pt #play in real-world
```

### 键盘控制
- **前进/后退**：I/K 或 小键盘 8/2（步进：0.1 m/s）
- **左右平移**：U/O 或 小键盘 7/9（步进：0.05 m/s）
- **左右转向**：J/L 或 小键盘 4/6（步进：0.1 rad/s）
- **紧急停止**：空格 或 小键盘 5
- **退出仿真**：Q 或 ESC

## sim2real

### 




我的policy总是有个lin y 向左的偏差速度 经过试验 大致在vy为-0.27是偏左 vy为-0.29是往右 因此实际的vy 的分界点应该是 -0.28  
帮我矫正一下这个偏移量
