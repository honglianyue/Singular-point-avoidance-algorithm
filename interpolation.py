import numpy as np
from scipy.spatial.transform import Rotation as R
def slerp(q1, q2, t):
    """
    实现四元数球面线性插值
    
    参数:
        q1: 起始四元数 [x, y, z, w]
        q2: 终止四元数 [x, y, z, w]
        t: 插值参数 [0, 1]
    """
    # 确保四元数为单位四元数
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # 计算四元数点积
    dot = np.dot(q1, q2)
    
    # 如果点积为负，翻转其中一个四元数
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # 如果四元数非常接近，使用线性插值
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    
    # 执行球面线性插值
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return s0 * q1 + s1 * q2

def cartesian_linear_interpolation(start_position, end_position, start_quat, end_quat, num_points):
    """
    笛卡尔空间线性插补
    
    参数:
        start_position: 起始TCP位置 [x, y, z]
        end_position: 终止TCP位置 [x, y, z]
        start_quat: 起始四元数 [x, y, z, w]
        end_quat: 终止四元数 [x, y, z, w]
        num_points: 插补点数量
        
    返回:
        positions: 插补后的位置列表
        quats: 插补后的四元数列表
    """
    positions = []
    quats = []
    
    # 确保四元数格式正确
    start_rot = start_quat
    end_rot = end_quat
    
    for i in range(num_points):
        t = i / (num_points - 1)
        
        # 线性插值位置
        pos = (1 - t) * np.array(start_position) + t * np.array(end_position)
        positions.append(pos)
        
        # 球面线性插值四元数
        interpolated_rot = slerp(start_rot, end_rot, t)
        quats.append(interpolated_rot)
    
    return positions, quats

def joint_linear_interpolation(start_joints, end_joints, num_points):
    """
    关节空间线性插补
    
    参数:
        start_joints: 起始关节角度 [j4, j5, j6]
        end_joints: 终止关节角度 [j4, j5, j6]
        num_points: 插补点数量
        
    返回:
        joints: 插补后的关节角度列表
    """
    joints = []
    start_joints = np.array(start_joints)
    end_joints = np.array(end_joints)
    
    for i in range(num_points):
        t = i / (num_points - 1)
        # 线性插值关节角度
        joint = (1 - t) * start_joints + t * end_joints
        joints.append(joint)
    
    return joints

def get_wrist_center(position, orientation, d6):
    """
    计算腕中心位置
    
    参数:
        position: TCP位置 [x, y, z]
        orientation: 四元数 [x, y, z, w]
        d6: 腕关节到TCP的偏移距离
        
    返回:
        wrist_center: 腕中心位置 [x, y, z]
    """
    # 将四元数转换为旋转矩阵
    rot = R.from_quat(orientation).as_matrix()
    
    # 计算工具z轴方向
    z_axis = rot[:, 2]
    
    # 腕中心位置 = TCP位置 - d6 * 工具z轴方向
    wrist_center = np.array(position) - d6 * z_axis
    
    return wrist_center

def interpolation_for_integration(start_position, end_position, start_quat, end_quat, 
                                  start_wrist_joints, end_wrist_joints, d6=0.15, num_points=50):
    """
    生成用于集成到您系统的插补数据
    
    参数:
        start_position: 起始TCP位置 [x, y, z]
        end_position: 终止TCP位置 [x, y, z]
        start_quat: 起始四元数 [x, y, z, w]
        end_quat: 终止四元数 [x, y, z, w]
        start_wrist_joints: 起始腕关节角度 [j4, j5, j6]
        end_wrist_joints: 终止腕关节角度 [j4, j5, j6]
        d6: 工具长度参数
        num_points: 插补点数量
        
    返回:
        插补结果字典
    """
    # 笛卡尔空间插补
    positions, quats = cartesian_linear_interpolation(
        start_position, end_position, start_quat, end_quat, num_points)
    
    # 关节空间插补
    wrist_joints = joint_linear_interpolation(
        start_wrist_joints, end_wrist_joints, num_points)
    
    # 计算腕中心位置
    wrist_centers = []
    for i in range(num_points):
        wc = get_wrist_center(positions[i], quats[i], d6)
        wrist_centers.append(wc)
    
    # 返回所有需要的数据
    return {
        'positions': np.array(positions),
        'quats': np.array(quats),
        'wrist_joints': np.array(wrist_joints),
        'wrist_centers': np.array(wrist_centers)
    }