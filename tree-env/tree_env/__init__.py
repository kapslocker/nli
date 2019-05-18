from gym.envs.registration import register

register(
    id='tree-v0',
    entry_point='tree_env.envs:TreeEnv',
)