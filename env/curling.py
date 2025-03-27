from typing import Optional
import gymnasium as gym
import numpy as np
import pygame
from pygame import gfxdraw

"""
v_0
"""
class CurlingEnv(gym.Env):
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(self, 
                 max_episode_steps: int = 200,
                 render_mode: Optional[str] = 'human'):
        super(CurlingEnv, self).__init__()
        
        self.max_episode_steps = max_episode_steps
        self.steps = 0
        self.radius = 1.0   # curling radius
        self.m = 1       # curling mass
        self.e = 0.9    # curling restitution coefficient
        
        self.render_mode = render_mode

        self.last_u = 0
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True
        
        self.state_bounds = {
        }
        
        low = np.array([0, 0, 0, 0], dtype=np.float32)
        high = np.array([100, 100, 100, 100], dtype=np.float32)
        # 定义状态空间 [x, y, goal_x, goal_y]
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32
        )
    
        # 定义动作空间 (电压u)
        self.action_space = gym.spaces.Discrete(n=4)
    
    def step(self, action):
        self.steps += 1
        
        # 计算更新后的速度
        speed_x, speed_y = (0, 0)
        # 仿真10个时间步，步长为0.01s
        for _ in range(10):
            # 碰撞检测
            pass
        
        terminated = self.steps >= self.max_episode_steps  # 任务自然终止
        truncated = self.steps >= self.max_episode_steps  # 到达最大步数限制
        
        return self._get_obs(), reward, terminated, truncated, {}
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.steps = 0
        if options is None:
            alpha = self.np_random.uniform(*self.state_bounds['alpha'])
            alpha_dot = self.np_random.uniform(*self.state_bounds['alpha_dot'])
            self.state = np.array([alpha, alpha_dot], dtype=np.float32)
        else:
            alpha = options.get("alpha")
            alpha_dot = options.get("alpha_dot")
            self.state = np.array([alpha, alpha_dot], dtype=np.float32)
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}
    
    def _normalize_state(self, state):
        alpha, alpha_dot = state
        norm_alpha = normalize(alpha, *self.state_bounds['alpha'])
        norm_alpha_dot = normalize(alpha_dot, *self.state_bounds['alpha_dot'])
        return np.array([norm_alpha, norm_alpha_dot], dtype=np.float32)
    
    def _get_obs(self):
        if self.normalize_state:
            return self._normalize_state(self.state)
        else:
            alpha, alpha_dot = self.state
            return np.array([alpha, alpha_dot], dtype=np.float32)
    
    def render(self):
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # 绘制curling
        
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
        
