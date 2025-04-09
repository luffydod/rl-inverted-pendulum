from env.inverted_pendulum import InvertedPendulumEnv
from env.curling import CurlingEnv
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium.wrappers import NormalizeObservation, NormalizeReward
from gymnasium.vector import SyncVectorEnv

def make_env(
    env_id: str, 
    render_mode: str = 'rgb_array',
    discrete_action: bool = True,
    discrete_state: bool = False,
    normalize_obs: bool = False,
    normalize_reward: bool = False,
    reset_option: str = None
):
    def thunk():
        if env_id == "inverted-pendulum":
            env = InvertedPendulumEnv(
                render_mode=render_mode,
                discrete_action=discrete_action,
                discrete_state=discrete_state,
                reset_option=reset_option
            )
        elif env_id == "curling":
            env = CurlingEnv(
                render_mode=render_mode,
                discrete_state=discrete_state
            )
        else:
            raise ValueError(f"Invalid environment ID: {env_id}")
        env = RecordEpisodeStatistics(env)
        if normalize_obs:
            env = NormalizeObservation(env)
        if normalize_reward:
            env = NormalizeReward(env)
        return env
    return thunk

def make_envs(env_id: str = "inverted-pendulum",
            num_envs: int = 1,
            render_mode: str = 'rgb_array',
            discrete_action: bool = True,
            discrete_state: bool = False,
            normalize_obs: bool = False,
            normalize_reward: bool = False,
            reset_option: str = None):
    return SyncVectorEnv([make_env(env_id, 
                                   render_mode, 
                                   discrete_action, 
                                   discrete_state, 
                                   normalize_obs,
                                   normalize_reward,
                                   reset_option) for _ in range(num_envs)])