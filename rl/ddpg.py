import time
import sys
import os
import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pygame
import wandb
from stable_baselines3.common.buffers import ReplayBuffer
from env import make_envs
from config import DDPGConfig

conf = DDPGConfig()
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(
                np.array(env.single_observation_space.shape).prod() +
                np.array(env.single_action_space.shape).prod(),
                256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x, a):
        xa = torch.cat([x, a], dim=1)
        return self.network(xa)

class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, np.array(envs.single_action_space.shape).prod()),
            nn.Tanh(),
        )
        self.register_buffer(
            "action_scale",
            torch.tensor((envs.single_action_space.high - envs.single_action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((envs.single_action_space.high + envs.single_action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = self.network(x)
        return self.action_scale * x + self.action_bias

class DDPGAgent:
    def __init__(self, project_name: str = None,
                   algorithm: str = "ddpg"):
        self.project_name = project_name if project_name else conf.env_id
        self.algorithm = algorithm

    def create_model(self, envs, type: str = "actor"):
        if type == "actor":
            return Actor(envs).to(conf.device)
        else:
            return QNetwork(envs).to(conf.device)
    
    def load_model(self, envs, model_path: str, type: str = "actor"):
        model = self.create_model(envs, type=type)
        model_params = torch.load(model_path, map_location=conf.device)
        model.load_state_dict(model_params)
        return model

    def train(self):
        # wandb init
        run = wandb.init(
            project=f"{self.project_name}-{self.algorithm}",
            config=conf.get_params_dict(),
            monitor_gym=True
        )
        
        best_reward = -float('inf')
        best_model = None
        
        # create envs
        envs = make_envs(conf.env_id, num_envs=conf.n_envs, discrete_action=False)

        eval_env = make_envs(conf.env_id, num_envs=1, discrete_action=False)
        
        # create networks
        actor = self.create_model(envs, type="actor")
        actor_target = self.create_model(envs, type="actor")
        actor_target.load_state_dict(actor.state_dict())
        
        q_network = self.create_model(envs, type="critic")
        q_target = self.create_model(envs, type="critic")
        q_target.load_state_dict(q_network.state_dict())
        
        # optimizers
        actor_optimizer = optim.Adam(actor.parameters(), lr=conf.learning_rate)
        q_optimizer = optim.Adam(q_network.parameters(), lr=conf.learning_rate)
        
        # replay buffer
        rb = ReplayBuffer(
            conf.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            conf.device,
            n_envs=envs.num_envs,
            handle_timeout_termination=False,
        )
        
        # (n_envs, state_dim)
        obs, _ = envs.reset()
        
        for global_step in range(conf.total_timesteps):
            # select action
            if global_step < conf.learning_starts:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                with torch.no_grad():
                    actions = actor(torch.Tensor(obs).to(conf.device))
                    actions += torch.normal(0, actor.action_scale * conf.exploration_noise)
                    actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)
            
            # execute action
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            wandb.log({
                "charts/avg_reward": rewards.mean()
            }, step=global_step)
            
            if "episode" in infos:
                episode_length = infos["episode"]["l"].mean()
                episode_return = infos["episode"]["r"].mean() / episode_length
                episode_time = infos["episode"]["t"].mean()
                print(f"global_step={global_step}, episodic_return={episode_return}, episodic_length={episode_length}, episodic_time={episode_time}")
                # record episode information to wandb
                wandb.log({
                    "charts/episodic_return": episode_return,
                    "charts/episodic_length": episode_length,
                    "charts/episodic_time": episode_time,
                }, step=global_step)
                    
            # record replay buffer
            real_next_obs = next_obs.copy()

            rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
            
            # update obs
            obs = next_obs
            
            # training
            if global_step > conf.learning_starts:
                data = rb.sample(conf.batch_size)
                with torch.no_grad():
                    next_state_actions = actor_target(data.next_observations)
                    next_q_target = q_target(data.next_observations, next_state_actions)
                    next_q_value = data.rewards.flatten() + conf.gamma * (1 - data.dones.flatten()) * (next_q_target).view(-1)
                
                q_a_values = q_network(data.observations, data.actions).view(-1)
                q_loss = F.mse_loss(next_q_value, q_a_values)
                
                q_optimizer.zero_grad()
                q_loss.backward()
                nn.utils.clip_grad_norm_(q_network.parameters(), conf.max_grad_norm)
                q_optimizer.step()
                
                if global_step % conf.policy_frequency == 0:
                    actor_loss = -q_network(data.observations, actor(data.observations)).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), conf.max_grad_norm)
                    actor_optimizer.step()
                    
                    # soft update target networks
                    for target_param, param in zip(actor_target.parameters(), actor.parameters()):
                        target_param.data.copy_(conf.tau * param.data + (1.0 - conf.tau) * target_param.data)
                    for target_param, param in zip(q_target.parameters(), q_network.parameters()):
                        target_param.data.copy_(conf.tau * param.data + (1.0 - conf.tau) * target_param.data)
                        
                # record losses
                if global_step % 100 == 0:
                    wandb.log({
                        "losses/q_loss": q_loss.item(),
                        "losses/q_values": q_a_values.mean().item(),
                        "losses/actor_loss": actor_loss.item(),
                    }, step=global_step)
            
            # eval
            eval_reward = self.eval(global_step, eval_env, actor)
            if eval_reward and eval_reward > best_reward:
                best_reward = eval_reward
                best_model = actor.state_dict()
            
        # save model
        dir_path = f"models/{self.project_name}/{self.algorithm}"
        os.makedirs(dir_path, exist_ok=True)
        torch.save(actor.state_dict(), f"{dir_path}/{run.id}.pth")
        if best_model:
            torch.save(best_model, f"{dir_path}/{run.id}_best.pth")
            
        wandb.finish()

    def test(self, model_path: str = None):
        env = make_envs(conf.env_id, num_envs=1, render_mode="human")

        actor = self.load_model(env, model_path, type="actor")
        # obs, _ = env.reset()
        obs, _ = env.reset(options={"alpha": np.pi, "alpha_dot": 0})
        iters = 0
        
        while True:
            iters += 1
            if model_path:
                with torch.no_grad():
                    action = actor(torch.FloatTensor(obs).to(conf.device))
                    action += torch.normal(0, actor.action_scale * conf.exploration_noise)
                    action = action.cpu().numpy().clip(env.single_action_space.low, env.single_action_space.high)
            else:
                action = np.array([env.single_action_space.sample() for _ in range(env.num_envs)])
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            print(f"iter: {iters}, obs: {obs}, action: {action}, reward: {reward}")
            
            obs = next_obs
            
            env.render()[0]
            time.sleep(0.02)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            if terminated or truncated:
                break
    
    def eval(self, global_step, env, model):
        if global_step % conf.eval_frequency == 0:
            alpha_dot = np.random.uniform(-np.pi, np.pi)
            obs, _ = env.reset(options={"alpha": np.pi, "alpha_dot": alpha_dot})
            done = False
            frames = []
            cumulative_reward = 0
            
            while not done:
                # action = render_env.action_space.sample()
                with torch.no_grad():
                    actions = model(torch.FloatTensor(obs).to(conf.device))
                    actions += torch.normal(0, model.action_scale * conf.exploration_noise)
                    actions = actions.cpu().numpy().clip(env.single_action_space.low, env.single_action_space.high)
                
                # execute action and record frame
                obs, reward, terminated, truncated, _ = env.step(actions)
                
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