from typing import Optional
import gymnasium as gym
import numpy as np
import pygame

DISCRETE_POS = 10
DISCRETE_V = 5

class CurlingEnv(gym.Env):
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(self, 
                 max_episode_steps: int = 300,
                 render_mode: Optional[str] = 'human',
                 discrete_state: bool = False):
        super(CurlingEnv, self).__init__()
        
        self.max_episode_steps = max_episode_steps
        self.discrete_state = discrete_state
        
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
        self.v_max = 30     
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

        self.last_action = 0
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True
        
        self.state_bounds = {
        }
        
        low = np.array([self.radius, self.radius, self.radius, self.radius, -self.v_max, -self.v_max], dtype=np.float32)
        high = np.array([100 - self.radius, 100 - self.radius, 100 - self.radius, 100 - self.radius, self.v_max, self.v_max], dtype=np.float32)
        # 定义状态空间 [x, y, goal_x, goal_y, v_x, v_y]
        if self.discrete_state:
            self.observation_space = gym.spaces.MultiDiscrete(
                [DISCRETE_POS, 
                 DISCRETE_POS, 
                 DISCRETE_POS, 
                 DISCRETE_POS, 
                 DISCRETE_V, 
                 DISCRETE_V]
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=low, high=high, dtype=np.float32
            )
    
        # 定义动作空间 (+x, -x, +y, -y)
        self.action_space = gym.spaces.Discrete(n=4)
    
    def step(self, action):
        action = int(action)
        self.last_action = action
        
        self.steps += 1
        f0 = self.f_mapping[action]
        # simulate 10 timesteps
        for _ in range(10):
            # cal force
            f = -self.mu * self.v * np.abs(self.v) + f0
            a = f / self.m
            
            # update position, x = x0 + v0 * t + 0.5 * a * t^2
            self.curling_pos += self.v * self.t + 0.5 * a * self.t * self.t 
            # update velocity, v = v0 + a * t
            self.v += a * self.t
            
            # collision detection
            self.collision_detection()
        
        # update history
        self.curling_pos_history.append(self.curling_pos.copy())
        
        terminated = self.steps >= self.max_episode_steps  # 任务自然终止
        truncated = self.steps >= self.max_episode_steps  # 到达最大步数限制
        
        return self._get_obs(), self._get_reward(), terminated, truncated, {}
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.steps = 0
        # reset speed
        self.v = np.random.uniform(-10, 10, size=2)
        # reset position
        self.curling_pos = np.random.uniform(self.radius, 100 - self.radius, size=2)
        self.goal_pos = np.random.uniform(self.radius, 100 - self.radius, size=2)
        
        # clear history
        self.curling_pos_history.clear()
        self.curling_pos_history.append(self.curling_pos.copy())
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        if self.discrete_state:
            # 将连续状态转换为离散状态
            pos_bins = np.linspace(self.radius, 100 - self.radius, DISCRETE_POS)
            vel_bins = np.linspace(-self.v_max, self.v_max, DISCRETE_V)
            
            discrete_state = np.array([
                np.digitize(self.curling_pos[0], pos_bins) - 1,
                np.digitize(self.curling_pos[1], pos_bins) - 1,
                np.digitize(self.goal_pos[0], pos_bins) - 1,
                np.digitize(self.goal_pos[1], pos_bins) - 1,
                np.digitize(self.v[0], vel_bins) - 1,
                np.digitize(self.v[1], vel_bins) - 1
            ], dtype=np.int32)
            return discrete_state
        else:
            return np.concatenate([self.curling_pos, self.goal_pos, self.v], dtype=np.float32)
    
    def _get_tables(self):
        return np.zeros((4,
                         DISCRETE_POS, 
                         DISCRETE_POS, 
                         DISCRETE_POS,
                         DISCRETE_POS,
                         DISCRETE_V,
                         DISCRETE_V), dtype=np.float32)
        
    def _get_reward(self):
        # euclidean distance
        dist = np.linalg.norm(self.curling_pos - self.goal_pos)
        return -dist
    
    def collision_detection(self):
        if self.curling_pos[0] < self.radius or self.curling_pos[0] > 100 - self.radius:
            self.v[0] *= -self.e
        if self.curling_pos[1] < self.radius or self.curling_pos[1] > 100 - self.radius:
            self.v[1] *= -self.e
        self.curling_pos.clip(self.radius, 100 - self.radius)
        
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
        scale = self.screen_dim * 0.92 / 100
        offset = (self.screen_dim - 100 * scale) // 2  # 计算居中偏移量
        
        # 绘制边框
        pygame.draw.rect(
            self.screen, 
            (0, 0, 0), 
            (offset, offset, 100 * scale, 100 * scale), 
            5
        )
        
        # 绘制历史轨迹点
        for pos in self.curling_pos_history[:-1]:
            screen_pos = (int(pos[0] * scale) + offset, int(pos[1] * scale) + offset)
            pygame.draw.circle(self.screen, (200, 200, 200), screen_pos, 2)
        
        # 绘制目标点
        goal_screen_pos = (int(self.goal_pos[0] * scale) + offset, int(self.goal_pos[1] * scale) + offset)
        pygame.draw.circle(self.screen, (255, 0, 0), goal_screen_pos, 5)
        
        # 绘制当前冰壶位置
        current_pos = self.curling_pos_history[-1]
        current_screen_pos = (int(current_pos[0] * scale) + offset, int(current_pos[1] * scale) + offset)
        pygame.draw.circle(self.screen, (0, 0, 255), current_screen_pos, 8)
        
        # 在冰壶位置绘制力和速度方向箭头
        # 绘制力的方向箭头（绿色）
        arrow_length = 20  # 箭头长度
        if hasattr(self, 'last_action'):
            if self.last_action == 0:  # x轴正向
                end_pos = (current_screen_pos[0] + arrow_length, current_screen_pos[1])
            elif self.last_action == 1:  # x轴反向
                end_pos = (current_screen_pos[0] - arrow_length, current_screen_pos[1])
            elif self.last_action == 2:  # y轴正向
                end_pos = (current_screen_pos[0], current_screen_pos[1] + arrow_length)
            else:  # y轴反向
                end_pos = (current_screen_pos[0], current_screen_pos[1] - arrow_length)
            
            # 绘制力的方向线
            pygame.draw.line(self.screen, (0, 255, 0), current_screen_pos, end_pos, 2)
            # 绘制箭头头部
            arrow_head_length = 8
            if self.last_action in [0, 1]:  # 水平方向
                pygame.draw.line(self.screen, (0, 255, 0), 
                    end_pos, 
                    (end_pos[0] - arrow_head_length * (1 if self.last_action == 0 else -1), 
                        end_pos[1] - arrow_head_length), 2)
                pygame.draw.line(self.screen, (0, 255, 0), 
                    end_pos, 
                    (end_pos[0] - arrow_head_length * (1 if self.last_action == 0 else -1), 
                        end_pos[1] + arrow_head_length), 2)
            else:  # 垂直方向
                pygame.draw.line(self.screen, (0, 255, 0), 
                    end_pos, 
                    (end_pos[0] - arrow_head_length, 
                        end_pos[1] - arrow_head_length * (1 if self.last_action == 2 else -1)), 2)
                pygame.draw.line(self.screen, (0, 255, 0), 
                    end_pos, 
                    (end_pos[0] + arrow_head_length, 
                        end_pos[1] - arrow_head_length * (1 if self.last_action == 2 else -1)), 2)
        
        # 绘制速度方向箭头（黄色）
        v_magnitude = np.linalg.norm(self.v)
        if v_magnitude > 0.1:  # 只在速度大于阈值时绘制
            v_normalized = self.v / v_magnitude
            arrow_length = min(30, v_magnitude * 3)  # 箭头长度随速度变化，但有上限
            end_pos = (
                current_screen_pos[0] + v_normalized[0] * arrow_length,
                current_screen_pos[1] + v_normalized[1] * arrow_length
            )
            
            # 绘制速度方向线
            pygame.draw.line(self.screen, (255, 255, 0), current_screen_pos, end_pos, 2)
            
            # 绘制箭头头部
            arrow_head_length = 8
            angle = np.arctan2(v_normalized[1], v_normalized[0])
            pygame.draw.line(self.screen, (255, 255, 0),
                end_pos,
                (end_pos[0] - arrow_head_length * np.cos(angle + np.pi/6),
                    end_pos[1] - arrow_head_length * np.sin(angle + np.pi/6)), 2)
            pygame.draw.line(self.screen, (255, 255, 0),
                end_pos,
                (end_pos[0] - arrow_head_length * np.cos(angle - np.pi/6),
                    end_pos[1] - arrow_head_length * np.sin(angle - np.pi/6)), 2)

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
