from dataclasses import dataclass, asdict

@dataclass
class DQNConfig:
    env_id: str = "curling"
    """choice from ['inverted-pendulum', 'curling']"""
    n_envs: int = 4
    total_timesteps: int = 500000
    learning_rate: float = 1e-4
    buffer_size: int = 2000000
    batch_size: int = 128
    gamma: float = 0.98
    target_network_frequency: int = 1000
    tau: float = 0.4
    learning_starts: int = 5000
    max_grad_norm: float = 10
    train_frequency: int = 4
    eval_frequency: int = 10000
    start_epsilon: float = 1.0
    end_epsilon: float = 0.05
    exploration_fraction: float = 0.2
    device: str = "cuda:0"
    
    def get_params_dict(self):
        return asdict(self)

@dataclass
class DDPGConfig:
    env_id: str = "inverted-pendulum"
    """choice from ['inverted-pendulum', 'curling']"""
    n_envs: int = 8
    total_timesteps: int = 1000000
    learning_rate: float = 1e-4
    buffer_size: int = int(1e6)
    batch_size: int = 128
    gamma: float = 0.98
    tau: float = 0.4
    learning_starts: int = 20000
    max_grad_norm: float = 10
    policy_frequency: int = 2
    eval_frequency: int = 10000
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    device: str = "cuda:0"
    
    def get_params_dict(self):
        return asdict(self)

@dataclass
class QLearningConfig:
    env_id: str = "curling"
    """choice from ['inverted-pendulum', 'curling']"""
    n_envs: int = 4
    total_timesteps: int = 500000
    learning_rate: float = 1e-4
    
