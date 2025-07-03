
# from envs.half_cheetah_cost import HalfCheetahEnv_cost
from gym.envs.registration import register


register(
    id='DeskoCartpole-v0',
    entry_point='envs.half_cheetah_cost:HalfCheetahEnv_cost'
)
