import numpy as np
import os

class MultiAgentReplayBuffer():
    def __init__(self, max_size, critic_dims, actor_dims, n_actions,n_agents, batch_size):
        self.mem_size = max_size
        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)

        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []

        for i in range(self.n_agents):
            self.actor_state_memory.append(np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_new_state_memory.append(np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_action_memory.append(np.zeros((self.mem_size, self.n_actions)))


    def store_transition(self, raw_obs, state, action, reward, raw_obs_, state_, done):
        # raw_obs is the observation from all agents, state is the flattened version for the action memory.

        index = self.mem_cntr % self.mem_size

        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = action[agent_idx]

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace = False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        terminals = self.terminal_memory[batch]


        actor_states = []
        actor_new_states = []
        actions = []

        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])

        return actor_states, states, actions, rewards, actor_new_states, states_, terminals

    def ready(self):
        if self.mem_cntr >= self.batch_size:
            return True


# Path for checkpoints
def get_path(chkpt_dir, name):
    if not os.path.isdir(chkpt_dir):
        os.makedirs(chkpt_dir)
    checkpoint_file = os.path.join(chkpt_dir, name)
    return checkpoint_file


def obs_list_2_state_vector(observation):

    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


def plot_curve(scores, x):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):i+1])
    plt.plot(x, running_avg)
    plt.show()
