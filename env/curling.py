from typing import Optional
import gymnasium as gym
import numpy as np
import pygame

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
        # curling radius
        self.radius = 1.0
        # curling mass
        self.m = 1.0
        # curling restitution coefficient     
        self.e = 0.9
        # friction coefficient
        self.mu = 0.005
        # time step
        self.t = 0.01
        # max speed
        self.v_max = 10.0
        # v = [v_x, v_y]
        self.v = np.zeros(2, dtype=np.float32)
        # f-mapping
        self.f_mapping = [
                np.array([5, 0], dtype=np.float32), 
                np.array([-5, 0], dtype=np.float32), 
                np.array([0, 5], dtype=np.float32), 
                np.array([0, -5], dtype=np.float32)
            ]
        # pos = [x, y]
        self.curling_pos = np.zeros(2, dtype=np.float32)
        self.curling_pos_history = []
        self.goal_pos = np.zeros(2, dtype=np.float32)
        
        self.render_mode = render_mode

        self.last_u = 0
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True
        
        self.state_bounds = {
        }
        
        low = np.array([0, 0, 0, 0, -self.v_max, -self.v_max], dtype=np.float32)
        high = np.array([100, 100, 100, 100, self.v_max, self.v_max], dtype=np.float32)
        # 定义状态空间 [x, y, goal_x, goal_y, v_x, v_y]
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32
        )
    
        # 定义动作空间 (+x, -x, +y, -y)
        self.action_space = gym.spaces.Discrete(n=4)
    
    def step(self, action):
        action = int(action)
        self.steps += 1
        # simulate 10 timesteps
        for _ in range(10):
            # cal force
            f = -self.mu * self.v * self.v + self.f_mapping[action]
            a = f / self.m
            
            # update position, x = x0 + v0 * t + 0.5 * a * t^2
            self.curling_pos += self.v * self.t + 0.5 * a * self.t * self.t 
            # update velocity, v = v0 + a * t
            self.v += a * self.t
            
            # collision detection
            self.collision_detection()
            
            # update history
            self.curling_pos_history.append(self.curling_pos)
        
        terminated = self.steps >= self.max_episode_steps  # 任务自然终止
        truncated = self.steps >= self.max_episode_steps  # 到达最大步数限制
        
        return self._get_obs(), self._get_reward(), terminated, truncated, {}
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.steps = 0
        # reset speed
        self.v = np.random.uniform(-10, 10, size=2)
        # reset position
        self.curling_pos = np.random.uniform(0, 100, size=2)
        self.goal_pos = np.random.uniform(0, 100, size=2)
        
        # clear history
        self.curling_pos_history = []
        self.curling_pos_history.append(self.curling_pos)
        
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}
    
    def _get_obs(self):
        return np.concatenate([self.curling_pos, self.goal_pos, self.v], dtype=np.float32)
    
    def _get_reward(self):
        # euclidean distance
        dist = np.linalg.norm(self.curling_pos - self.goal_pos)
        return -dist
    
    def collision_detection(self):
        if self.curling_pos[0] < 0 or self.curling_pos[0] > 100:
            self.v[0] *= -self.e
        if self.curling_pos[1] < 0 or self.curling_pos[1] > 100:
            self.v[1] *= -self.e
    
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

        # 清空屏幕
        self.screen.fill((255, 255, 255))
        
        # 坐标转换系数 (游戏世界坐标到屏幕坐标)
        scale = 3  # 将100*100的游戏区域缩放到300*300
        offset = (self.screen_dim - 100 * scale) // 2  # 计算居中偏移量
        
        # 绘制边框
        pygame.draw.rect(
            self.screen, 
            (0, 0, 0), 
            (offset, offset, 100 * scale, 100 * scale), 
            2
        )
        
        # 绘制历史轨迹点
        for pos in self.curling_pos_history[:-1]:
            screen_pos = (int(pos[0] * scale) + offset, int(pos[1] * scale) + offset)
            pygame.draw.circle(self.screen, (200, 200, 200), screen_pos, 2)
        
        # 绘制目标点
        goal_screen_pos = (int(self.goal_pos[0] * scale) + offset, int(self.goal_pos[1] * scale) + offset)
        pygame.draw.circle(self.screen, (255, 0, 0), goal_screen_pos, 5)
        
        # 绘制当前冰壶位置
        if len(self.curling_pos_history) > 0:
            current_pos = self.curling_pos_history[-1]
            current_screen_pos = (int(current_pos[0] * scale) + offset, int(current_pos[1] * scale) + offset)
            pygame.draw.circle(self.screen, (0, 0, 255), current_screen_pos, int(self.radius * scale))
        
        # 翻转坐标系（pygame的坐标系原点在左上角，我们需要将其转换为左下角）
        self.screen.blit(pygame.transform.flip(self.screen, False, True), (0, 0))
        
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
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
