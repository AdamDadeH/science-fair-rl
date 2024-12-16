import numpy as np


    
class GeneralQLearningAgent:
    def __init__(self,
                 actions_func,
                 learning_rate=0.1,
                 discount_factor=0.9,
                 exploration_start=1.0,
                 exploration_end=0.01,
                 num_episodes=100):
        self.actions_func = actions_func  # Function that returns available actions for a given state
        self.q_table = {}  # Use a dictionary to handle arbitrary states and actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_start = exploration_start
        self.exploration_end = exploration_end
        self.num_episodes = num_episodes
        self.current_episode = 0

    def new_episode(self):
        self.current_episode += 1

    def terminate(self):
        return self.current_episode >= self.num_episodes

    def get_exploration_rate(self):
        # Calculate the current exploration rate using the given formula
        exploration_rate = self.exploration_start * (self.exploration_end / self.exploration_start) ** (self.current_episode / self.num_episodes)
        return exploration_rate


    def make_hashable(self, state):
        if isinstance(state, dict):
            return tuple((key, self.make_hashable(value)) for key, value in sorted(state.items()))
        elif isinstance(state, list):
            return tuple(self.make_hashable(item) for item in state)
        else:
            return state  #

    def get_action(self, state):
        exploration_rate = self.get_exploration_rate()
        actions = self.actions_func()
        if np.random.rand() < exploration_rate:
            return np.random.choice(actions)  # Random action selection
        else:
            hashable_state = self.make_hashable(state)
            q_values = self.q_table.get(hashable_state, {})  # Default to empty dict if state not in table
            if not q_values:  # If there are no Q-values for this state, return a random action
                return np.random.choice(actions)
            best_actions = [action for action in q_values if q_values[action] == max(q_values.values())]
            return np.random.choice(best_actions)  # Randomly choose from the best actions

    def update_q_table(self, state, action, next_state, reward):
        hashable_state = self.make_hashable(state)
        hashable_next_state = self.make_hashable(next_state)

        current_q_value = self.q_table.get(hashable_state, {}).get(action, 0)

        if hashable_next_state not in self.q_table:
            self.q_table[hashable_next_state] = {}

        if len(self.q_table[hashable_next_state]) > 0:
            best_next_action_q_value = max(self.q_table[hashable_next_state].values())
        else:
            best_next_action_q_value = 0 

        new_q_value = current_q_value + self.learning_rate * (reward + self.discount_factor * best_next_action_q_value - current_q_value)

        if hashable_state not in self.q_table:
            self.q_table[hashable_state] = {}
        self.q_table[hashable_state][action] = new_q_value

class QLearningAgent:

    """
    The code below is for the agent. 
    The agent can move in four directions: up, down, left, and right. 
    You do not need to change this code for your engineering project. 
    As a variation to the project, you can (with a little bit of python knowledge) 
    try making changes to the learning and exploration rate work as a variation to the project.
    """

    def __init__(self,
                 maze,
                 actions,
                 learning_rate=0.1,
                 discount_factor=0.9,
                 exploration_start=1.0,
                 exploration_end=0.01,
                 num_episodes=100):
        # Initialize the Q-learning agent with a Q-table containing all zeros
        # where the rows represent states, columns represent actions, and the third dimension is for each action
        self.actions = actions  # Store available actions
        self.actions_map = {action: ix for ix, action in enumerate(actions)}
        self.num_actions = len(actions)  # Number of possible actions
        self.q_table = np.zeros((maze.maze_height, maze.maze_width, self.num_actions))
        self.learning_rate = learning_rate          # Learning rate controls how much the agent updates its Q-values after each action
        self.discount_factor = discount_factor      # Discount factor determines the importance of future rewards in the agent's decisions
        self.exploration_start = exploration_start  # Exploration rate determines the likelihood of the agent taking a random action
        self.exploration_end = exploration_end
        self.num_episodes = num_episodes
        self.current_episode = 0

    def new_episode(self):
        self.current_episode += 1

    def terminate(self):
        return self.current_episode >= self.num_episodes

    def get_exploration_rate(self):
        # Calculate the current exploration rate using the given formula
        exploration_rate = self.exploration_start * (self.exploration_end / self.exploration_start) ** (self.current_episode / self.num_episodes)
        return exploration_rate

    def get_action(self, state): # State is tuple representing where agent is in maze (x, y)
        position = state["position"]
        exploration_rate = self.get_exploration_rate()
        # Select an action for the given state either randomly (exploration) or using the Q-table (exploitation)
        if np.random.rand() < exploration_rate:
            return self.actions[np.random.randint(self.num_actions)] # Choose a random action based on number of available actions
        else:
            return self.actions[np.argmax(self.q_table[position[0], position[1], :])] # Choose the action with the highest Q-value for the given state

    def update_q_table(self, state, action, next_state, reward):
        next_state = next_state["position"]
        state = state["position"]
        action_ix = self.actions_map[action]
        # Find the best next action by selecting the action that maximizes the Q-value for the next state
        best_next_action = np.argmax(self.q_table[next_state[0], next_state[1], :])

        # Get the current Q-value for the current state and action
        current_q_value = self.q_table[state[0], state[1], action_ix]

        # Q-value update using Q-learning formula
        new_q_value = current_q_value + self.learning_rate * (reward + self.discount_factor * self.q_table[next_state[0], next_state[1], best_next_action] - current_q_value)

        # Update the Q-table with the new Q-value for the current state and action
        self.q_table[state[0], state[1], action_ix] = new_q_value

print("This code block has been run and the QLearningAgent class is now available for use.")
