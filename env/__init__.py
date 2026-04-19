from gym.envs.registration import register

register(
    id='MultiRobotEnv-v0',
    entry_point='env.gym_env:GymEnv',
)