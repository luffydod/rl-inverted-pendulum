from env.inverted_pendulum import InvertedPendulumEnv
from env.curling import CurlingEnv
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium.vector import SyncVectorEnv

def make_env(
    env_id: str, 
    render_mode: str = 'rgb_array',
    discrete_action: bool = True,
    discrete_state: bool = False
):
    def thunk():
        if env_id == "inverted-pendulum":
            env = InvertedPendulumEnv(
                discrete_action=discrete_action,
                render_mode=render_mode
            )
        elif env_id == "curling":
            env = CurlingEnv(
                render_mode=render_mode
            )
        else:
            raise ValueError(f"Invalid environment ID: {env_id}")
        env = RecordEpisodeStatistics(env)
        return env
    return thunk

def make_envs(env_id: str = "inverted-pendulum",
            num_envs: int = 1,
            render_mode: str = 'rgb_array',
            discrete_action: bool = True,
            discrete_state: bool = False):
    return SyncVectorEnv([make_env(env_id, render_mode, discrete_action, discrete_state) for _ in range(num_envs)])