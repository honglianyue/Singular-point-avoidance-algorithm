import numpy as np
from DH import *
import numpy as np
from jacobian import *
class LMSolver:
    """
    完全按照KDL实现的Levenberg-Marquardt算法求解器
    """
    def __init__(self, dh_params,P,R):
        """
        初始化LM求解器
        
        参数:
            dh_params: DH参数字典
        """
        self.dh_params = dh_params
        self.nj = 3  # 我们求解前3个关节
        
        # 设置算法参数
        self.maxiter = 1000      # 最大迭代次数
        self.eps = 1e-15          # 位姿误差收敛阈值
        self.eps_joints = 1e-15   # 关节角度收敛阈值
        
        # 权重参数 - 可以调整位置和旋转误差的权重
        self.L = np.ones(6)
        self.L[:3] = P         # 位置误差权重
        self.L[3:] = R # 姿态误差权重
     
        # 结果信息
        self.lastNrOfIter = 0
        self.lastDifference = 0
        self.lastTransDiff = 0
        self.lastRotDiff = 0
        self.lastSV = None
        
        self.display_information = True  # 是否显示迭代信息
        
        self.enable_early_termination = True # 是否启用提前终止
    
    def cartToJnt(self, q_init, target_pose, jac_method='custom'):
        """
        使用LM算法求解逆运动学
        
        参数:
            q_init: 初始关节角度, 长度为nj的np.array
            target_pose: 目标位姿(4x4变换矩阵)
            jac_method: 雅可比矩阵计算方法 ('custom' 或 'numerical')
        
        返回:
            q_out: 求解得到的关节角度
            error: 错误码
            history: 迭代历史数据
        """
        # 检查参数尺寸
        if len(q_init) != self.nj:
            return q_init, self.E_SIZE_MISMATCH, None
        
        # 初始化历史记录
        history = {
            'iterations': [],
            'q': [q_init.copy()],
            'errors': [],
            'rho':[]
        }
        
        # 初始化基本参数 
        v = 2         # 步长因子
        tau = 1e-3  # 初始阻尼因子系数
        
        # 复制初始关节角度
        q = q_init.copy()
        q_out = q.copy()  # 初始化输出变量
        
        # 计算初始位姿 - 对应compute_fwdpos(q)
        T_base_head = forward_kinematics_first3joints(q, self.dh_params)
        
        # 计算初始误差 - 对应Twist_to_Eigen(diff(...))
        delta_pos = compute_pose_error(T_base_head, target_pose)
        
        # 应用权重 - 对应L.asDiagonal()*delta_pos
        delta_pos = self.L * delta_pos
        
        # 计算误差范数 - 对应delta_pos.norm()
        delta_pos_norm = np.linalg.norm(delta_pos)
        
        # 检查初始误差是否已经足够小
        if delta_pos_norm < self.eps:
            self.lastNrOfIter = 0
            
            # 重新计算未加权的误差 - 对应Twist_to_Eigen(diff(...))
            delta_pos_unweighted = compute_pose_error(T_base_head, target_pose)
            self.lastDifference = np.linalg.norm(delta_pos_unweighted)
            self.lastTransDiff = np.linalg.norm(delta_pos_unweighted[:3])
            self.lastRotDiff = np.linalg.norm(delta_pos_unweighted[3:])
            
            # 计算雅可比矩阵
            if jac_method == 'custom':
                jac = custom_jacobian(q, self.dh_params)
            else:  # numerical
                jac = numerical_jacobian(q, self.dh_params)
            
            jac_weighted = np.diag(self.L) @ jac
            
            # 计算SVD
            _, s, _ = np.linalg.svd(jac_weighted, full_matrices=False)
            self.lastSV = s
            
            return q, self.E_NOERROR, history
        
        # 计算雅可比矩阵 - 对应compute_jacobian(q)
        if jac_method == 'custom':
            jac = custom_jacobian(q, self.dh_params)
        else:  # numerical
            jac = numerical_jacobian(q, self.dh_params)
        
        # 应用权重到雅可比矩阵 - 对应jac = L.asDiagonal()*jac
        jac = np.diag(self.L) @ jac
        
        # 初始化lambda (按照论文实现)
        lambda_ = tau * np.max(jac.T @ jac)
        
        # 主迭代循环
        for i in range(self.maxiter):
            # 计算SVD (每次迭代都重新计算) - 对应svd.compute(jac)
            U, s, Vh = np.linalg.svd(jac, full_matrices=False)

            # 修改奇异值 - 对应修改original_Aii
            original_Aii = np.zeros_like(s)
            for j in range(len(s)):  # 对所有奇异值应用
                original_Aii[j] = s[j] / (s[j]*s[j] + lambda_)
            
            # 计算关节角增量 - 对应KDL中的Segment A部分
            tmp = U.T @ delta_pos  
            tmp = tmp * original_Aii
            diffq = Vh.T @ tmp

             # 计算梯度 - 对应grad = jac.transpose()*delta_pos
            grad = jac.T @ delta_pos

            # 修改为： (jac.T @ jac + lambda_ * np.eye(self.nj)) @ diffq = grad
            #A = jac.T @ jac + lambda_ * np.eye(self.nj)
            #diffq = np.linalg.inv(A) @ grad
            
            # 输出迭代信息
            if self.display_information and (i % 100 == 0 or i < 5):
                print(f"------- iteration {i} ----------------")
                print(f"  q              = {q}")
                print(f"  lambda         = {lambda_}")
                print(f"  eigenvalues    = {s}")
                print(f"  difference norm= {delta_pos_norm}")
                print(f"  grad norm      = {np.linalg.norm(grad)}")
                print("")
            
            # 记录历史
            history['iterations'].append(i)
            history['q'].append(q.copy())
            history['errors'].append(delta_pos_norm)
            
            # 检查步长是否足够小 - 对应dnorm = diffq.lpNorm<Eigen::Infinity>()
            dnorm = np.max(np.abs(diffq))  # 无穷范数
            if self.enable_early_termination and dnorm < self.eps_joints:
                self.lastDifference = delta_pos_norm
                self.lastNrOfIter = i
                self.lastSV = s
                q_out = q.copy()
                
                # 重新计算当前位姿和误差
                T_base_head = forward_kinematics_first3joints(q, self.dh_params)
                delta_pos_unweighted = compute_pose_error(T_base_head, target_pose)
                self.lastTransDiff = np.linalg.norm(delta_pos_unweighted[:3])
                self.lastRotDiff = np.linalg.norm(delta_pos_unweighted[3:])
                print(f"步长过小，停止迭代,迭代次数：{i}")
                return q_out, history
            

            # 检查梯度是否足够小 - 对应grad.transpose()*grad < eps_joints*eps_joints
            if self.enable_early_termination and np.dot(grad, grad) < self.eps_joints * self.eps_joints:
                # 重新计算当前位姿和误差
                T_base_head = forward_kinematics_first3joints(q, self.dh_params)
                delta_pos_unweighted = compute_pose_error(T_base_head, target_pose)
                
                self.lastDifference = np.linalg.norm(delta_pos_unweighted)
                self.lastTransDiff = np.linalg.norm(delta_pos_unweighted[:3])
                self.lastRotDiff = np.linalg.norm(delta_pos_unweighted[3:])
                self.lastSV = s
                self.lastNrOfIter = i
                q_out = q.copy()
                print(f"梯度过小，停止迭代,迭代次数：{i}")
                return q_out, history
            
            # 计算新的关节角度 - 对应q_new = q+diffq
            q_new = q +  diffq
            #q_new = np.clip(q_new, [-170*np.pi/180,-105*np.pi/180,-210*np.pi/180],[170*np.pi/180,145*np.pi/180,70*np.pi/180] )
            # 计算新的位姿和误差
            T_base_head_new = forward_kinematics_first3joints(q_new, self.dh_params)
            delta_pos_new = compute_pose_error(T_base_head_new, target_pose)
            delta_pos_new = self.L * delta_pos_new  # 应用权重
            delta_pos_new_norm = np.linalg.norm(delta_pos_new)
            
            # 计算信任区域比率 - 对应KDL中的rho计算
            rho = delta_pos_norm**2 - delta_pos_new_norm**2
            history['rho'].append(rho)
            denominator = diffq.T @ (lambda_ * diffq + grad)
            
            # 防止除零
            if abs(denominator) < 1e-10:
                if rho > 0:
                    rho = 1.0
                else:
                    rho = -1.0
            else:
                rho = rho / denominator
            
            # 根据信任区域比率更新参数
            if rho > 0:
                # 更新成功，接受新的关节角度
                q = q_new.copy()
                delta_pos = delta_pos_new.copy()
                delta_pos_norm = delta_pos_new_norm
                grad = jac.T @ delta_pos
                # 检查是否收敛 按照文献流程修改
                if self.enable_early_termination and (np.max(np.abs(grad))<self.eps or delta_pos_norm* delta_pos_norm < self.eps):
                    # 重新计算未加权的误差
                    delta_pos_unweighted = compute_pose_error(T_base_head_new, target_pose)
                    self.lastDifference = np.linalg.norm(delta_pos_unweighted)
                    self.lastTransDiff = np.linalg.norm(delta_pos_unweighted[:3])
                    self.lastRotDiff = np.linalg.norm(delta_pos_unweighted[3:])
                    self.lastSV = s
                    self.lastNrOfIter = i
                    q_out = q.copy()
                    print(f"收敛，停止迭代,迭代次数：{i}")
                    return q_out, self.E_NOERROR, history
                
                # 重新计算雅可比矩阵 - 使用新的关节角度
                if jac_method == 'custom':
                    jac = custom_jacobian(q, self.dh_params)
                else:  # numerical
                    jac = numerical_jacobian(q, self.dh_params)
                
                jac = np.diag(self.L) @ jac  # 应用权重
               
                # 减小阻尼因子 (KDL方式) - 对应lambda = lambda*max(1/3.0, 1-tmp*tmp*tmp)
                fmp = 2 * rho - 1
                lambda_ = lambda_ * max(1/3, 1 - fmp**3)
                v = 2
            else:
                # 更新失败，增大阻尼因子 - 对应lambda = lambda*v; v = 2*v
                lambda_ = lambda_ * v
                v = 2 * v
        
        # 达到最大迭代次数
        self.lastDifference = delta_pos_norm
        self.lastTransDiff = np.linalg.norm(delta_pos[:3] / self.L[:3])
        self.lastRotDiff = np.linalg.norm(delta_pos[3:] / self.L[3:])
        self.lastSV = s
        self.lastNrOfIter = self.maxiter
        q_out = q.copy()
        
        print(f"达到最大迭代次数 {self.maxiter}")
        print(f"最终位置误差: {self.lastTransDiff}")
        print(f"最终旋转误差: {self.lastRotDiff}")
        
        return q_out, history