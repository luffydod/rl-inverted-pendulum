import gymnasium as gym
import numpy as np
import pygame
import math

class InvertedPendulumEnv(gym.Env):
    def __init__(self, is_discrete=False, render_mode='test'):
        super(InvertedPendulumEnv, self).__init__()
        
        self.is_discrete = is_discrete
        # 系统参数
        '''
        如果摆动太快，可以增加阻尼系数 b
        如果摆动太慢，可以减小转动惯量 J
        如果控制效果不够明显，可以增大电机转矩常数 K
        '''
        self.l = 0.3        # 摆杆长度 (m)
        self.m = 0.055      # 质量 (kg)
        self.J = (1/5) * self.m * self.l**2  # 转动惯量 (kg⋅m²)
        self.g = 9.81       # 重力加速度 (m/s²)
        self.b = 3.0e-6   # 阻尼系数 (N⋅m⋅s/rad)
        self.K = 0.0536   # 转矩常数 (N⋅m/A)
        self.R = 9.5      # 电机电阻 (Ω)
        
        # 仿真参数
        self.dt = 0.01     # 时间步长 (s)
        
        # 定义状态空间 [角度α, 角速度α_dot]
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.pi, -15*np.pi], dtype=np.float32),
            high=np.array([np.pi, 15*np.pi], dtype=np.float32),
            dtype=np.float32
        )
        
        # 定义动作空间 (电压u)
        if self.is_discrete:
            self.n_actions = 11
            self.action_space = gym.spaces.Discrete(self.n_actions)
            self.discrete_actions = np.linspace(-3.0, 3.0, self.n_actions)
        else:
            self.action_space = gym.spaces.Box(
                low=-3.0,
                high=3.0,
                shape=(1,),
                dtype=np.float32
            )
        
        self.state = None
        
        # 添加渲染相关的属性
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.screen_width = 600
        self.screen_height = 400
        self.scale = 500  # 像素/米的比例
        
    def reset(self):
        # 初始状态设置为最低点 [π, 0]
        self.state = np.array([np.pi, 0.0], dtype=np.float32)
        return self.state, {}
    
    def step(self, action):
        alpha = self.state[0]
        alpha_dot = self.state[1]
        
        if self.is_discrete:
            u = self.discrete_actions[action]
        else:
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
        
        self.state[0] = alpha_new
        self.state[1] = alpha_dot_new
        
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
    
    def render(self):
        if self.screen is None:
            pygame.init()
            if self.render_mode == 'test':
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                self.clock = pygame.time.Clock()
        
        if self.render_mode == 'test':
            surface = self.screen
        elif self.render_mode == 'train':
            surface = pygame.Surface((self.screen_width, self.screen_height))
        
        surface.fill((255, 255, 255))  # 白色背景
        
        # 绘制支点（固定轴）
        pivot_x = self.screen_width // 2
        pivot_y = self.screen_height // 2
        pygame.draw.circle(surface, (0, 0, 0), (pivot_x, pivot_y), 5)
        
        # 计算摆杆端点位置
        angle = self.state[0]  # 当前角度
        pendulum_length = self.l * self.scale  # 缩放摆杆长度
        end_x = pivot_x + pendulum_length * math.sin(angle)
        end_y = pivot_y - pendulum_length * math.cos(angle)
        
        # 绘制摆杆
        pygame.draw.line(surface, (0, 0, 255), (pivot_x, pivot_y), (end_x, end_y), 3)
        # 绘制摆杆末端质点
        pygame.draw.circle(surface, (255, 0, 0), (int(end_x), int(end_y)), 8)
        
        # 添加文字信息
        font = pygame.font.Font(None, 36)
        angle_text = font.render(f'Angle: {math.degrees(angle):.1f}°', True, (0, 0, 0))
        speed_text = font.render(f'Speed: {math.degrees(self.state[1]):.1f}°/s', True, (0, 0, 0))
        surface.blit(angle_text, (10, 10))
        surface.blit(speed_text, (10, 50))
        
        if self.render_mode == 'test':
            pygame.display.flip()
            self.clock.tick(60)
        elif self.render_mode == 'train':
            # 将pygame surface转换为numpy数组
            image_array = pygame.surfarray.array3d(surface)
            image_array = np.rot90(image_array, k=3)  # k=1表示逆时针旋转90度
            image_array = np.transpose(image_array, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            image_array = np.flip(image_array, axis=2)  # 水平翻转
            image_array = np.ascontiguousarray(image_array)  # 确保数据连续
            return image_array
        
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
