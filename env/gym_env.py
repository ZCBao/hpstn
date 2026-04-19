import gym
from env.env_base import EnvBase

class GymEnv(gym.Env, EnvBase):
    def __init__(self, **kwargs):
        EnvBase.__init__(self, **kwargs)
        
    def reset(self, safe_dist=0.0):
        self.components['robots'].robots_reset(safe_dist)
        self.components['obstacles'].obstacles_reset()
        obs_list, _, _, _, _, _, _ = self.env_observation()
        return obs_list

    def env_step(self, vel_list, stop=True):
        self.obstacles_circle_step(stop=stop)
        self.robots_step(vel_list, stop=stop)
        obs_list, lo_dist_list, lo_ctime_list, lo_flag_list, vo_dist_list, vo_ctime_list, vo_flag_list = self.env_observation()
        reward_list, done_list, info_list = self.cal_reward_done_info_list(lo_dist_list, lo_ctime_list, lo_flag_list, vo_dist_list, vo_ctime_list, vo_flag_list)
        
        return obs_list, reward_list, done_list, info_list

    def render(self, mode='human', save=False, save_step=None, step_time=None, **kwargs):
        self.env_plot.clear_dynamic_components(save_step==0)
        self.env_plot.draw_dynamic_components(save_step, step_time, **kwargs)

        if not save:
            self.env_plot.pause()