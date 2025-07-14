import numpy as np
from DH import *
from LMsolver import *
from jacobian import *
from scipy.spatial.transform import Rotation 

def solve_inverse_kinematics_lm(x, y, z, nx, ny, nz, ox, oy, oz, ax, ay, az, 
                              theta4, theta5, theta6, dh_params, p=1,r=100,
                              initial_guess=None, jac_method='custom'):
    """
    使用LM算法求解6轴机器人逆运动学前3个关节角度
    
    参数:
        x, y, z: TCP位置坐标
        nx, ny, nz: TCP位姿的x轴方向
        ox, oy, oz: TCP位姿的y轴方向
        ax, ay, az: TCP位姿的z轴方向
        theta4, theta5, theta6: 已知的关节角度(弧度)
        dh_params: 机器人DH参数字典
        initial_guess: 初始关节角度猜测值，默认为零位
        jac_method: 雅可比矩阵计算方法 ('custom' 或 'numerical')
        
    返回:
        theta1, theta2, theta3: 计算得到的前3个关节角度(弧度)
        error_code: 错误码
    """
    # 构建TCP位姿矩阵
    tcp_pose = np.array([
        [nx, ox, ax, x],
        [ny, oy, ay, y],
        [nz, oz, az, z],
        [0, 0, 0, 1]
    ])
    
    # 计算腕点位置 (从TCP向后推算)T03矩阵
    T03 = calculate_wrist_point(tcp_pose, theta4, theta5, theta6, dh_params)
    
    # 初始化LM求解器
    solver = LMSolver(dh_params,p,r)
    solver.display_information = False  # 是否显示迭代信息
    
    # 设置初始猜测值
    if initial_guess is None:
        initial_guess = np.zeros(3)  # 默认全部为0
    
    # 使用LM算法求解前3个关节角度
    q_sol,history= solver.cartToJnt(initial_guess, T03, jac_method)
    
    # 返回计算得到的关节角度
    theta1, theta2, theta3 = q_sol
    
    # 规范化角度到[-pi, pi]范围域
    theta1 = ((theta1 + np.pi) % (2*np.pi)) - np.pi
    theta2 = ((theta2 + np.pi) % (2*np.pi)) - np.pi
    theta3 = ((theta3 + np.pi) % (2*np.pi)) - np.pi
    
    return theta1, theta2, theta3,history

def solve_with_multiple_initial_guesses(x, y, z, nx, ny, nz, ox, oy, oz, ax, ay, az, 
                                       theta4, theta5, theta6, dh_params, jac_method='custom'):
    """
    使用多个初始猜测值尝试求解逆运动学
    """
    initial_guesses = [
        [0.0, 0.0, 0.0],
        [np.pi/4, np.pi/4, np.pi/4],
        [-np.pi/4, np.pi/4, np.pi/4],
        [np.pi/4, -np.pi/4, np.pi/4],
        [np.pi/4, np.pi/4, -np.pi/4],
        [np.pi/2, 0.0, 0.0],
        [0.0, np.pi/2, 0.0],
        [0.0, 0.0, np.pi/2],
        [np.pi/3, np.pi/3, np.pi/3],
        [-np.pi/3, -np.pi/3, -np.pi/3],
        [np.pi/6, np.pi/6, np.pi/6],
        [-np.pi/6, -np.pi/6, -np.pi/6]
    ]
    
    # 跟踪最佳结果
    best_result = None
    best_error = float('inf')

    
    # 尝试每个初始猜测值
    for i, guess in enumerate(initial_guesses):
        
        try:
            # 求解逆运动学
            theta1, theta2, theta3,history= solve_inverse_kinematics_lm(
                x, y, z, nx, ny, nz, ox, oy, oz, ax, ay, az, 
                theta4, theta5, theta6, dh_params, p=1,r=10000,
                initial_guess=guess,
                jac_method=jac_method
            )
            
            # 验证结果
            joint_angles = [theta1, theta2, theta3, theta4, theta5, theta6]
            result_pose = forward_kinematics(joint_angles, dh_params)
            result_pos = result_pose[:3, 3]
            
            # 计算误差
            position_error = np.linalg.norm(result_pos - np.array([x, y, z]))
            
            # 检查是否是最佳结果
            if position_error < best_error:
                best_error = position_error
                best_result = (theta1, theta2, theta3)
           
        except Exception as e:
            print(f"  求解出错: {e}")
            continue
    
    
    # 输出最佳结果
    if best_result is not None:
        theta1, theta2, theta3 = best_result
        print(f"\n===== 最佳解 =====")
        print(f"关节角度: θ1={np.degrees(theta1):.1f}°, θ2={np.degrees(theta2):.1f}°, θ3={np.degrees(theta3):.1f}°")
        print(f"位置误差: {best_error:.4f} mm")
        
        return theta1, theta2, theta3 ,history
    else:
        print("\n未找到有效解!")
        return None, None, None
    
def test_multiple_points(points_data,dh_params):
    """测试多个路点的逆运动学求解
    
    参数:
        points_data: 包含多个路点信息的列表,每个路点是一个字典,包含:
            - position: [x, y, z]
            - quaternion: [qx, qy, qz, qw] 
            - wrist_joints: [theta4, theta5, theta6](degrees)
    """
    results = []
    poss = points_data['positions']
    quats = points_data['quats']
    wrists = [np.radians(a) for a in points_data['wrist_joints']]

    for i in range(len(poss)):
        print(f"\n========= 路点 {i+1} =========")
        pos = poss[i]
        quat = quats[i]
        wrist = wrists[i]
        print(f"位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] mm")
        print(f"四元数: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
        print(f"后三个关节角度: [{wrist[0]:.2f}, {wrist[1]:.2f}, {wrist[2]:.2f}] rad")
        
        # 创建旋转对象并获取旋转矩阵
        rot = Rotation.from_quat(quat)
        rotation_matrix = rot.as_matrix()
        
        # 提取姿态向量
        nx,ny,nz = rotation_matrix[:,0]
        ox,oy,oz = rotation_matrix[:,1]
        ax,ay,az = rotation_matrix[:,2]
        
        if len(results) == 0:
            initial_guess = [0.0, 0.0, 0.0]  # 默认初始猜测值
        else:
            initial_guess = [np.radians(a) for a in results[-1]['joint_angles'][:3]]
        print(f"初始值：{initial_guess[0]:.2f},{initial_guess[1]:.2f},{initial_guess[2]:.2f}rad")   
        # 
        theta1_c, theta2_c, theta3_c, history_c = solve_inverse_kinematics_lm(
            pos[0], pos[1], pos[2], 
            nx, ny, nz, ox, oy, oz, ax, ay, az,
            wrist[0], wrist[1], wrist[2], 
            dh_params,p=1,r=100000,initial_guess= initial_guess,
            jac_method='numerical'
        )
        
        # 验证结果
        joint_angles = [theta1_c, theta2_c, theta3_c, wrist[0], wrist[1], wrist[2]]
        result_pose = forward_kinematics(joint_angles, dh_params)
        
        error =  result_pose[:3, 3] - np.array([pos[0], pos[1], pos[2]])
        error = np.linalg.norm(error)

        #求解T46矩阵
           # 提取DH参数
        a3 = dh_params.get('a3', 0)
        a4 = dh_params.get('a4', 0)
        a5 = dh_params.get('a5', 0)
        
        d4 = dh_params.get('d4', 0)
        d5 = dh_params.get('d5', 0)
        d6 = dh_params.get('d6', 0)
        alpha3 = dh_params.get('alpha3', 0)
        alpha4 = dh_params.get('alpha4', 0)
        alpha5 = dh_params.get('alpha5', 0)

        T3_4 = dh_transform(a3, alpha3, d4, wrist[0])
        T4_5 = dh_transform(a4, alpha4, d5, wrist[1])
        T5_6 = dh_transform(a5, alpha5, d6, wrist[2]+np.pi) #  第六个关节偏移180度
        T3_6 = np.dot(T3_4, np.dot(T4_5, T5_6))

        results.append({
            'joint_angles': [np.degrees(a) for a in joint_angles],
            'position_error': error,
            'history': history_c
        })
        
        # 打印该路点的结果
        print("\n求解结果:")
        print(f"位置误差: {error:.4f} mm")
        print("关节角度(度):")
        for j, angle in enumerate(results[-1]['joint_angles']):
            print(f"J{j+1}: {angle:.3f}°")
            
    return results