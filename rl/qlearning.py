import time
import sys
import os
import math
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

def exponential_schedule(start_e: float, end_e: float, duration: int, t: int):
    lambda_ = math.log(start_e / end_e) / duration
    return max(end_e, end_e + (start_e - end_e) * math.exp(-lambda_ * t))

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
        eval_env = make_envs(conf.env_id, 1, discrete_action=True, discrete_state=True)
        
        best_reward = -float('inf')
        best_model = None
        
        # Q_table, shape=(action_dim, len_s1, len_s2...)
        self.Q_table = env.get_attr("q_table")[0]
        print(f"Q_table shape=", self.Q_table.shape)
        
        obs, _ = env.reset()
        
        for global_step in range(conf.total_timesteps):
            # epsilon = linear_schedule(
            #     start_e=conf.start_epsilon,
            #     end_e=conf.end_epsilon,
            #     duration=conf.exploration_fraction * conf.total_timesteps,
            #     t=global_step
            # )
            epsilon = exponential_schedule(
                start_e=conf.start_epsilon,
                end_e=conf.end_epsilon,
                duration=conf.exploration_fraction * conf.total_timesteps,
                t=global_step
            )
            
            if random.random() < epsilon:
                action = np.array([env.single_action_space.sample()])
            else:
                full_index = [slice(None)] + list(obs[0].astype(np.int32))
                action = np.array([np.argmax(self.Q_table[tuple(full_index)])])
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            if "episode" in info:
                episode_length = info["episode"]["l"].mean()
                episode_return = info["episode"]["r"].mean()
                episode_time = info["episode"]["t"].mean()
                print(f"global_step={global_step}, episodic_return={episode_return}, episodic_length={episode_length}, episodic_time={episode_time}")
                # record episode information to wandb
                wandb.log({
                    "charts/episodic_return": episode_return,
                    "charts/episodic_length": episode_length,
                    "charts/episodic_time": episode_time,
                }, step=global_step)
                
            current_q_indices = [int(action[0])] + list(obs[0].astype(np.int32))
            next_state_indices = [slice(None)] + list(next_obs[0].astype(np.int32))
            
            current_q = self.Q_table[tuple(current_q_indices)]
            
            next_max_q = np.max(self.Q_table[tuple(next_state_indices)])
            
            # update q value
            self.Q_table[tuple(current_q_indices)] = current_q + \
                conf.learning_rate * (reward[0] + conf.gamma * next_max_q * (1 - terminated) - current_q)
            
            # record
            if global_step % 100 == 0:
                wandb.log({
                    "training/avg_reward": reward.mean(),
                    "training/epsilon": epsilon,
                    "training/q_value": current_q
                }, step=global_step)
            
            # update obs
            obs = next_obs
            
            # eval model
            eval_reward = self.eval(global_step, eval_env)
            if eval_reward and eval_reward > best_reward:
                best_reward = eval_reward
                best_model = self.Q_table.copy()
            
        
        # save model
        dir_path = f"models/{self.project_name}/{self.algorithm}"
        os.makedirs(dir_path, exist_ok=True)
        np.save(f"{dir_path}/{run.id}.npy", self.Q_table)
        if best_model is not None:
            np.save(f"{dir_path}/{run.id}_best.npy", best_model)
        wandb.finish()
    
    def test(self, model_path: str = None):
        env = make_envs(conf.env_id, num_envs=1, render_mode="human")
        if model_path:
            self.Q_table = np.load(model_path)
        
        obs, _ = env.reset()
        
        while True:
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

    def eval(self, global_step, env):
        if global_step % conf.eval_frequency == 0:
            obs, _ = env.reset(options={"alpha": np.pi, "alpha_dot": 0})
            done = False
            frames = []
            cumulative_reward = 0
            
            while not done:
                full_index = [slice(None)] + list(obs[0].astype(np.int32))
                action = np.array([np.argmax(self.Q_table[tuple(full_index)])])
                
                next_obs, reward, terminated, truncated, _ = env.step(action)
                # print(f"obs: {obs}, action: {action}, reward: {reward}")
                
                obs = next_obs
                
                done = terminated
                cumulative_reward += reward
                frame = env.render()[0]
                # (H, W, C) -> (C, H, W)
                frames.append(frame.transpose(2, 0, 1))
            
            video_frames = np.array(frames)
            wandb.log({
                "eval/episodic_return": cumulative_reward,
                "eval/video": wandb.Video(
                    video_frames, 
                    fps=30, 
                    format="gif"
                ),
            }, step=global_step)
            
            return float(cumulative_reward)
        
        else:
            return None