from DH import *
import numpy as np
def numerical_jacobian1(angles, dh_params, delta=1e-7):
    """
    使用数值微分法计算雅可比矩阵，避免使用欧拉角
    """
    theta1, theta2, theta3 = angles
    
    # 计算当前位姿矩阵
    T_current = forward_kinematics_first3joints(angles, dh_params)
    current_pos = T_current[:3, 3]
    current_rot = T_current[:3, :3]
    
    # 初始化雅可比矩阵
    J = np.zeros((6, 3))
    
    # 对每个关节计算雅可比列
    for i, (th1, th2, th3) in enumerate([
        (theta1 + delta, theta2, theta3),
        (theta1, theta2 + delta, theta3),
        (theta1, theta2, theta3 + delta)
    ]):
        # 计算扰动后的位姿
        T_plus = forward_kinematics_first3joints([th1, th2, th3], dh_params)
        pos_plus = T_plus[:3, 3]
        rot_plus = T_plus[:3, :3]
        
        # 计算位置雅可比列
        J[:3, i] = (pos_plus - current_pos) / delta
        
        # 计算旋转雅可比列 (使用对数映射)
        dR = rot_plus @ current_rot.T
        angle = np.arccos((np.trace(dR) - 1) / 2)
        if abs(angle) < 1e-10:
            w = np.zeros(3)
        else:
            w = np.array([
                dR[2, 1] - dR[1, 2],
                dR[0, 2] - dR[2, 0],
                dR[1, 0] - dR[0, 1]
            ]) * angle / (2 * np.sin(angle))
            
        J[3:, i] = w / delta
    
    return J
#使用李代数和中心差分法计算雅可比矩阵
def numerical_jacobian(angles, dh_params, delta=1e-6):
    """
    使用李代数和中心差分法计算雅可比矩阵
    """
    # 计算当前位姿矩阵
    T_current = forward_kinematics_first3joints(angles, dh_params)
    current_pos = T_current[:3, 3]
    current_rot = T_current[:3, :3]
    
    # 初始化雅可比矩阵
    J = np.zeros((6, 3))
    
    # 对每个关节计算雅可比列
    for i in range(3):
        # 创建前向和后向扰动的角度
        angles_plus = list(angles)
        angles_minus = list(angles)
        
        angles_plus[i] += delta
        angles_minus[i] -= delta
        
        # 计算前向扰动后的位姿
        T_plus = forward_kinematics_first3joints(angles_plus, dh_params)
        pos_plus = T_plus[:3, 3]
        rot_plus = T_plus[:3, :3]
        
        # 计算后向扰动后的位姿
        T_minus = forward_kinematics_first3joints(angles_minus, dh_params)
        pos_minus = T_minus[:3, 3]
        rot_minus = T_minus[:3, :3]
        
        # 计算位置雅可比列（中心差分）
        J[:3, i] = (pos_plus - pos_minus) / (2 * delta)
        
        # 使用李代数的对数映射计算旋转雅可比列
        log_plus = logSO3(rot_plus @ current_rot.T)
        log_minus = logSO3(rot_minus @ current_rot.T)
        
        # 使用中心差分
        J[3:, i] = (log_plus - log_minus) / (2 * delta)
    
    return J

def logSO3(R):
    """
    从旋转矩阵到so(3)的对数映射
    返回一个向量 w，使得 R = exp([w]_×)
    """
    # 确保R是一个有效的旋转矩阵
    if np.linalg.det(R) < 0:
        raise ValueError("不是有效的旋转矩阵（行列式为负）")
    
    # 计算旋转角度
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 确保在[-1,1]范围内
    theta = np.arccos(cos_theta)
    
    # 如果角度接近0，返回零向量
    if abs(theta) < 1e-10:
        return np.zeros(3)
    
    # 如果角度接近π
    if abs(theta - np.pi) < 1e-10:
        # 查找不为零的对角元
        if R[0, 0] > -1 + 1e-10:
            w = np.sqrt((1 + R[0, 0] - R[1, 1] - R[2, 2]) / 2)
            return np.array([w, R[0, 1]/(2*w), R[0, 2]/(2*w)]) * theta / np.sin(theta)
        elif R[1, 1] > -1 + 1e-10:
            w = np.sqrt((1 - R[0, 0] + R[1, 1] - R[2, 2]) / 2)
            return np.array([R[1, 0]/(2*w), w, R[1, 2]/(2*w)]) * theta / np.sin(theta)
        else:
            w = np.sqrt((1 - R[0, 0] - R[1, 1] + R[2, 2]) / 2)
            return np.array([R[2, 0]/(2*w), R[2, 1]/(2*w), w]) * theta / np.sin(theta)
    
    # 一般情况：提取旋转轴
    w = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ]) * theta / (2 * np.sin(theta))
    
    return w

def custom_jacobian(angles, dh_params):
    """
    关节角度的雅可比矩阵
    
    参数:
        angles: 当前关节角度 [theta1, theta2, theta3]
        dh_params: DH参数字典
        
    返回:
        6×3的雅可比矩阵，每列代表对应关节的影响
    """
    theta1, theta2, theta3 = angles
    # 提取DH参数
    a1 = dh_params.get('a1', 0)
    a2 = dh_params.get('a2', 0)
    d1 = dh_params.get('d1', 0)
    d2 = dh_params.get('d2', 0)
    d3 = dh_params.get('d3', 0)
    alpha1 = dh_params.get('alpha1', 0)
    alpha2 = dh_params.get('alpha2', 0)
    
    # 计算各个变换矩阵
    T0_1 = dh_transform(0, 0, d1, theta1)
    T1_2 = dh_transform(a1, alpha1, d2, theta2-np.pi/2)  # 第二个关节有90度的偏移
    T2_3 = dh_transform(a2, alpha2, d3, theta3)
    
    # 计算T0_2和T0_3
    T0_2 = np.dot(T0_1, T1_2)
    T0_3_current = np.dot(T0_2, T2_3)
    
    # 计算各关节坐标系的z轴方向（旋转轴）
    z0 = np.array([0, 0, 1])  # 基坐标系的z轴
    z1 = T0_1[:3, :3] @ np.array([0, 0, 1])  # 第1关节z轴在基坐标系中的表示
    z2 = T0_2[:3, :3] @ np.array([0, 0, 1])  # 第2关节z轴在基坐标系中的表示
    
    # 计算各关节的位置
    o0 = np.array([0, 0, 0])
    o1 = T0_1[:3, 3]
    o2 = T0_2[:3, 3]
    o3 = T0_3_current[:3, 3]  # 末端位置
    
    # 计算位置雅可比矩阵 (3×3)
    J_v = np.zeros((3, 3))
    
    # 关节1对末端位置的影响
    J_v[:, 0] = np.cross(z0, o3 - o0)
    
    # 关节2对末端位置的影响
    J_v[:, 1] = np.cross(z1, o3 - o1)
    
    # 关节3对末端位置的影响
    J_v[:, 2] = np.cross(z2, o3 - o2)
    
    # 计算旋转雅可比矩阵 (3×3)
    J_w = np.zeros((3, 3))
    J_w[:, 0] = z0  # 关节1的旋转轴
    J_w[:, 1] = z1  # 关节2的旋转轴
    J_w[:, 2] = z2  # 关节3的旋转轴
    
    # 合并为完整的雅可比矩阵 (6×3)
    J = np.vstack([J_v, J_w])
    
    return J