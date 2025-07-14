import numpy as np

def dh_transform(a, alpha, d, theta):
    """
    计算DH变换矩阵
    
    参数:
    a: 连杆长度
    alpha: 连杆扭角(弧度)
    d: 连杆偏距
    theta: 关节角度(弧度)
    
    返回:
    T: 4x4的DH变换矩阵
    """
    # 构建DH变换矩阵
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    T = np.array([
        [ct, -st, 0, a],
        [st*ca, ct*ca, -sa, -d*sa],
        [st*sa, ct*sa, ca, d*ca],
        [0, 0, 0, 1]
    ])
    
    return T

def forward_kinematics(angles, dh_params):
    """
    计算6轴机器人正向运动学
    
    参数:
        angles: 关节角度 [theta1, theta2, theta3, theta4, theta5, theta6]
        dh_params: DH参数字典
        
    返回:
        T: 末端执行器相对于基座的变换矩阵，4x4 np.array
    """
    theta1, theta2, theta3, theta4, theta5, theta6 = angles
    
    # 提取DH参数
    a0 = dh_params.get('a0', 0)
    a1 = dh_params.get('a1', 0)
    a2 = dh_params.get('a2', 0)
    a3 = dh_params.get('a3', 0)
    a4 = dh_params.get('a4', 0)
    a5 = dh_params.get('a5', 0)
    
    d1 = dh_params.get('d1', 0)
    d2 = dh_params.get('d2', 0)
    d3 = dh_params.get('d3', 0)
    d4 = dh_params.get('d4', 0)
    d5 = dh_params.get('d5', 0)
    d6 = dh_params.get('d6', 0)
    
    alpha0 = dh_params.get('alpha0', 0)
    alpha1 = dh_params.get('alpha1', 0)
    alpha2 = dh_params.get('alpha2', 0)
    alpha3 = dh_params.get('alpha3', 0)
    alpha4 = dh_params.get('alpha4', 0)
    alpha5 = dh_params.get('alpha5', 0)
    
    # 计算各个变换矩阵
    T0_1 = dh_transform(a0, alpha0, d1, theta1)
    T1_2 = dh_transform(a1, alpha1, d2, theta2-np.pi/2)  # 第二个关节偏移90度
    T2_3 = dh_transform(a2, alpha2, d3, theta3)
    T3_4 = dh_transform(a3, alpha3, d4, theta4)
    T4_5 = dh_transform(a4, alpha4, d5, theta5)
    T5_6 = dh_transform(a5, alpha5, d6, theta6+np.pi) #  第六个关节偏移180度
    
    # 计算总变换矩阵
    T0_2 = np.dot(T0_1, T1_2)
    T0_3 = np.dot(T0_2, T2_3)
    T0_4 = np.dot(T0_3, T3_4)
    T0_5 = np.dot(T0_4, T4_5)
    T0_6 = np.dot(T0_5, T5_6)
     
    return T0_6

def forward_kinematics_first3joints(angles, dh_params):
    """
    计算前3个关节的正向运动学，用于LM算法中的关节1-3求解
    
    参数:
        angles: 关节角度 [theta1, theta2, theta3]
        dh_params: DH参数字典
        
    返回:
        T: 第3个关节坐标系相对于基座的变换矩阵，4x4 np.array
    """
    theta1, theta2, theta3 = angles
    
    # 提取DH参数
    a0 = dh_params.get('a0', 0)
    a1 = dh_params.get('a1', 0)
    a2 = dh_params.get('a2', 0)
    
    d1 = dh_params.get('d1', 0)
    d2 = dh_params.get('d2', 0)
    d3 = dh_params.get('d3', 0)
    
    alpha0 = dh_params.get('alpha0', 0)
    alpha1 = dh_params.get('alpha1', 0)
    alpha2 = dh_params.get('alpha2', 0)
    
    # 计算各个变换矩阵
    T0_1 = dh_transform(a0, alpha0, d1, theta1)
    T1_2 = dh_transform(a1, alpha1, d2, theta2-np.pi/2)  # 第二个关节偏移90度
    T2_3 = dh_transform(a2, alpha2, d3, theta3)
    
    # 计算总变换矩阵
    T0_2 = np.dot(T0_1, T1_2)
    T0_3 = np.dot(T0_2, T2_3)
    

    return T0_3

def calculate_wrist_point(tcp_pose, theta4, theta5, theta6, dh_params):
    """
    
    参数:
        tcp_pose: TCP位姿矩阵 (4x4)
        theta4, theta5, theta6: 腕关节角度 (弧度)
        dh_params: DH参数字典
        
    返回:
        wrist_pose: T03位姿矩阵 (4x4)
    """
    # 提取相关DH参数
    a3 = dh_params.get('a3', 0)
    a4 = dh_params.get('a4', 0)
    a5 = dh_params.get('a5', 0)
    
    d4 = dh_params.get('d4', 0)
    d5 = dh_params.get('d5', 0)
    d6 = dh_params.get('d6', 0)
    
    alpha3 = dh_params.get('alpha3', 0)
    alpha4 = dh_params.get('alpha4', 0)
    alpha5 = dh_params.get('alpha5', 0)
    
    # 计算腕部变换矩阵
    T3_4 = dh_transform(a3, alpha3, d4, theta4)
    T4_5 = dh_transform(a4, alpha4, d5, theta5)
    T5_6 = dh_transform(a5, alpha5, d6, theta6 + np.pi)  # 腕部末端偏移180度
    
    # 计算腕部到TCP的变换矩阵
    T3_6 = np.dot(T3_4, np.dot(T4_5, T5_6))
  
    # 计算腕点位姿
    T0_3 = np.dot(tcp_pose, np.linalg.inv(T3_6))
    
    return T0_3

def compute_pose_error(T_current, T_desired):
    """
    计算两个位姿之间的误差
    
    参数:
        T_current: 当前变换矩阵，4x4 np.array
        T_desired: 目标变换矩阵，4x4 np.array
    
    返回:
        error: 位姿误差向量，6x1 np.array (前3个元素是位置误差，后3个元素是姿态误差)
    """
    # 计算位置误差 (简单的欧几里得距离)
    pos_error = T_desired[:3, 3] - T_current[:3, 3]
    
    # 计算旋转误差 (使用轴角表示法)
    R_current = T_current[:3, :3]
    R_desired = T_desired[:3, :3]
    
    # 计算旋转误差矩阵 R_d * R_c^T
    R_error = R_desired @ R_current.T
    
    
    # 将旋转矩阵转换为轴角表示
    angle = np.arccos((np.trace(R_error) - 1) / 2)
    
    if abs(angle) < 1e-10:
        axis = np.zeros(3)
    else:
        axis =  np.array([
            R_error[2, 1] - R_error[1, 2],
            R_error[0, 2] - R_error[2, 0],
            R_error[1, 0] - R_error[0, 1]
        ]) / (2 * np.sin(angle))
    
    # 轴角表示：轴 * 角度
    rot_error = axis * angle
    
    # 组合位置和旋转误差
    return np.concatenate([pos_error, rot_error])