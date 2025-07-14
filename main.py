import numpy as np
from DH import *
from LMsolver import *
from jacobian import *
from interpolation import *
from Solve import *
import matplotlib.pyplot as plt

# 设置机器人DH参数(工业机器人 NB12s-1214-5A)  
dh_params = {  
    'a0': 0,      # 连杆1长度(mm)
    'a1': 100,    # 连杆2长度(mm) 
    'a2': 680,    # 连杆3长度(mm)
    'a3': 50,     # 连杆4长度(mm)
    'a4': 0,      # 连杆5长度(mm)
    'a5': 0,      # 连杆6长度(mm)
    'd1': 500,    # 关节1偏移(mm)
    'd2': 0,      # 关节2偏移(mm)
    'd3': 0,      # 关节3偏移(mm
    'd4': 660,    # 关节4偏移(mm)
    'd5': 0,      # 关节5偏移(mm)
    'd6': 98,     # 关节6偏移(mm)
    'alpha0': 0,          # 连杆0扭角(弧度)
    'alpha1': -np.pi/2,   # 连杆1扭角(弧度)
    'alpha2': 0,          # 连杆2扭角(弧度)
    'alpha3': -np.pi/2,   # 连杆3扭角(弧度)
    'alpha4': np.pi/2,    # 连杆4扭角(弧度)
    'alpha5': -np.pi/2,   # 连杆5扭角(弧度)
}

start_pos=[1018.1418,40.4859,656.71229] # 起始位置
end_pos=[1018.1418,40.4859,1015.41402] # 目标位置
start_quat=[-0.0163,0.8205,0.0114,0.5713] # 起始四元数
end_quat=[-0.0163,0.8205,0.0114,0.5713] # 目标四元数
start_joint=[-0.00,-22.770,0.00]  # 起始关节角
end_joint=[-180,-6.578,180] # 目标关节角

points=interpolation_for_integration(start_pos,end_pos,start_quat,end_quat,start_joint,end_joint,d6=0.15,num_points=100)
# 运行测试
results = test_multiple_points(points,dh_params)

# 统计分析
print("\n=== 统计分析 ===")
errors = [r['position_error'] for r in results]
print(f"平均位置误差: {np.mean(errors):.4f} mm")
print(f"最大位置误差: {np.max(errors):.4f} mm")
print(f"最小位置误差: {np.min(errors):.4f} mm")
print(f"误差标准差: {np.std(errors):.4f} mm")


#=======================绘图=======================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 绘制关节角度曲线
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))


# 提取所有关节角度数据
joint_angles = np.array([result['joint_angles'] for result in results])

# 绘制前3个关节角度（算法求解得到的）
time_points = np.arange(len(joint_angles))
ax1.plot(time_points, joint_angles[:, 0], 'r-', label='J1')
ax1.plot(time_points, joint_angles[:, 1], 'g-', label='J2')
ax1.plot(time_points, joint_angles[:, 2], 'b-', label='J3')
ax1.set_title('前3个关节角度变化曲线')
ax1.set_xlabel('插补点序号')
ax1.set_ylabel('关节角度 (度)')
ax1.grid(True)
ax1.legend()

# 绘制后3个关节角度（线性插值得到的）
ax2.plot(time_points, joint_angles[:, 3], 'r--', label='J4')
ax2.plot(time_points, joint_angles[:, 4], 'g--', label='J5')
ax2.plot(time_points, joint_angles[:, 5], 'b--', label='J6')
ax2.set_title('后3个关节角度变化曲线')
ax2.set_xlabel('插补点序号')
ax2.set_ylabel('关节角度 (度)')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

# 绘制位置误差随时间的变化
plt.figure(figsize=(10, 6))
plt.plot(time_points, [r['position_error'] for r in results], 'k-', label='位置误差')
plt.title('插补过程中的位置误差变化')
plt.xlabel('插补点序号')
plt.ylabel('位置误差 (mm)')
plt.grid(True)
plt.legend()
plt.show()