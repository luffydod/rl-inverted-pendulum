from dataclasses import dataclass, asdict

@dataclass
class DQNConfig:
    env_id: str = "inverted-pendulum"
    """choice from ['inverted-pendulum', 'curling']"""
    n_envs: int = 1
    total_timesteps: int = 500000
    learning_rate: float = 3e-4
    buffer_size: int = 1000000
    batch_size: int = 128
    gamma: float = 0.98
    target_network_frequency: int = 10000
    tau: float = 1
    learning_starts: int = 100
    max_grad_norm: float = 10
    train_frequency: int = 4
    eval_frequency: int = 5000
    start_epsilon: float = 1.0
    end_epsilon: float = 0.05
    exploration_fraction: float = 0.2
    device: str = "cuda:0"
    buffer_type: str = "uniform"
    """choice from ['per', 'uniform']"""
    
    def get_params_dict(self):
        return asdict(self)

@dataclass
class PPOConfig:
    env_id: str = "inverted-pendulum"
    """choice from ['inverted-pendulum',]"""
    n_envs: int = 2
    total_timesteps: int = 500000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float = None
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 10
    target_kl: float = 0.03
    normalize_advantage: bool = True
    device: str = "cuda:0"
    eval_frequency: int = 10000
    log_interval: int = 1
    reset_num_timesteps: bool = True
    
    def get_params_dict(self):
        return asdict(self)

@dataclass
class DDPGConfig:
    env_id: str = "inverted-pendulum"
    """choice from ['inverted-pendulum',]"""
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
    eval_frequency: int = 4
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
    n_envs: int = 1
    gamma: float = 0.9
    total_timesteps: int = 500000
    eval_frequency: int = 10000
    learning_rate: float = 0.1
    start_epsilon: float = 1
    end_epsilon: float = 0.01
    exploration_fraction: float = 0.4
    
    def get_params_dict(self):
        return asdict(self)
    
