import numpy as np
from make_env import make_env

env = make_env('simple_adversary') # simple_tag
# print(env.__dict__)
print("Number of agents : ", env.n)
print("Obs space : ", env.observation_space)
print("Action space : ", env.action_space)

observation = env.reset()
print(observation)
