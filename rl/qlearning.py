import time
import sys
import os
import numpy as np
import random
import pygame
import wandb
from env import make_envs
from config import QLearningConfig

conf = QLearningConfig()

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

class QLearningAgent:
    def __init__(self, 
                 project_name: str = None,
                 algorithm: str = "qlearning"):
        self.project_name = project_name if project_name else conf.env_id
        self.algorithm = algorithm
    
    def train(self):
        # wandb init
        run = wandb.init(
            project=f"{self.project_name}-{self.algorithm}",
            config=conf.get_params_dict(),
            monitor_gym=True
        )
        
        # 创建环境
        env = make_envs(conf.env_id, conf.n_envs, discrete_action=True, discrete_state=True)
        
        # Q_table, shape=(action_dim, len_s1, len_s2...)
        self.Q_table = env.get_attr("q_table")[0]
        print(f"Q_table shape=", self.Q_table.shape)
        
        # 训练循环
        obs, _ = env.reset()
        
        for global_step in range(conf.total_timesteps):
            # 计算epsilon
            epsilon = linear_schedule(
                start_e=conf.start_epsilon,
                end_e=conf.end_epsilon,
                duration=conf.exploration_fraction * conf.total_timesteps,
                t=global_step
            )
            
            # epsilon-greedy选择动作
            if random.random() < epsilon:
                action = np.array([env.single_action_space.sample()])
            else:
                # 将obs转换为Q表的索引
                full_index = [slice(None)] + list(obs[0].astype(np.int32))
                action = np.array([np.argmax(self.Q_table[tuple(full_index)])])
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Q-learning更新
            current_q_indices = [int(action[0])] + list(obs[0].astype(np.int32))
            next_state_indices = [slice(None)] + list(next_obs[0].astype(np.int32))
            
            # print("current_q_indices = ", current_q_indices)
            # 获取当前Q值
            current_q = self.Q_table[tuple(current_q_indices)]
            
            # 获取下一状态的最大Q值
            next_max_q = np.max(self.Q_table[tuple(next_state_indices)])
            
            # Q-learning更新公式
            self.Q_table[tuple(current_q_indices)] = current_q + \
                conf.learning_rate * (reward[0] + conf.gamma * next_max_q * (1 - terminated) - current_q)
            
            # 记录数据
            if global_step % 100 == 0:
                wandb.log({
                    "training/avg_reward": reward.mean(),
                    "training/epsilon": epsilon,
                    "training/q_value": current_q
                }, step=global_step)
            
            # 更新状态
            obs = next_obs
            
            if terminated or truncated:
                obs, _ = env.reset()
        
        # save model
        dir_path = f"models/{self.project_name}/{self.algorithm}"
        os.makedirs(dir_path, exist_ok=True)
        np.save(f"{dir_path}/{run.id}.npy", self.Q_table)
        
        wandb.finish()
    
    def test(self, model_path: str = None):
        env = make_envs(conf.env_id, num_envs=1, render_mode="human")
        if model_path:
            self.Q_table = np.load(model_path)
        
        obs, _ = env.reset()
        
        while True:
            # 根据Q表选择动作
            full_index = [slice(None)] + list(obs[0].astype(np.int32))
            action = np.array([np.argmax(self.Q_table[tuple(full_index)])])
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            print(f"obs: {obs}, action: {action}, reward: {reward}")
            
            obs = next_obs
            
            env.render()[0]
            time.sleep(0.02)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            if terminated or truncated:
                break
