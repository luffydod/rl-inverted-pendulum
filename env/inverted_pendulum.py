import gym
import numpy as np

class InvertedPendulumEnv(gym.Env):
    def __init__(self):
        super(InvertedPendulumEnv, self).__init__()
        
        # 系统参数
        self.J = 1.91e-4  # 转动惯量
        self.m = 0.055    # 质量
        self.g = 9.81     # 重力加速度
        self.l = 0.042    # 摆杆长度
        self.b = 3.0e-6   # 阻尼系数
        self.K = 0.0536   # 转矩常数
        self.R = 9.5      # 电机电阻
        
        # 定义状态空间 [角度α, 角速度α_dot]
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.pi, -15*np.pi]),
            high=np.array([np.pi, 15*np.pi]),
            dtype=np.float32
        )
        
        # 定义动作空间 (电压u)
        self.action_space = gym.spaces.Box(
            low=-3.0,
            high=3.0,
            shape=(1,),
            dtype=np.float32
        )
        
        self.state = None
        
    def reset(self):
        # 初始状态设置为最低点 [π, 0]
        self.state = np.array([np.pi, 0.0])
        return self.state
    
    def step(self, action):
        alpha, alpha_dot = self.state
        u = np.clip(action[0], -3.0, 3.0)  # 确保电压在[-3,3]范围内
        
        # 实现系统动力学方程
        # α̈ = (1/J)(mgl*sin(α) - bα̇ - (K²/R)α̇ + (K/R)u)
        alpha_ddot = (1/self.J) * (
            self.m * self.g * self.l * np.sin(alpha) - 
            self.b * alpha_dot - 
            (self.K**2/self.R) * alpha_dot + 
            (self.K/self.R) * u
        )
        
        # 使用欧拉方法进行数值积分
        dt = 0.01  # 时间步长
        alpha_dot_new = alpha_dot + alpha_ddot * dt
        alpha_new = alpha + alpha_dot * dt
        
        # 确保角度在[-π,π]范围内
        alpha_new = ((alpha_new + np.pi) % (2 * np.pi)) - np.pi
        
        self.state = np.array([alpha_new, alpha_dot_new])
        
        # 计算奖励
        reward = self._compute_reward(u)
        
        # 判断是否达到目标
        done = self._is_done()
        
        return self.state, reward, done, {}
    
    def _compute_reward(self, u):
        # 实现给定的奖励函数 R(s,a) = -s^T diag(5,0.1)s - u²
        # R(s, a) = - 5 * alpha^2 - 0.1 * alpha_dot^2 - u^2
        s = self.state
        reward = -5 * s[0]**2 - 0.1 * s[1]**2 - u**2
        return reward
    
    def _is_done(self):
        # 稳定在最高点 s=[0,0]
        alpha, alpha_dot = self.state
        return abs(alpha) < 0.01 and abs(alpha_dot) < 0.01
