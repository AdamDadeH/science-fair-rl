import matplotlib.pyplot as plt

def run_single_simulation(agent, 
                   maze, 
                   train=True,
                   reward_map = {
                    "wall": -10, "step": -1, "goal": 100
                   },
                   max_steps=500000, 
                   debug=False):
    # Initialize the agent's current state to the maze's start position
    maze.reset()

    is_done = False
    episode_reward = 0
    episode_step = 0
    path = [maze.current_position]

    # Continue until the episode is done 
    while not is_done and episode_step < max_steps:
        current_state = maze.get_current_state()

        # Get the agent's action for the current state using its Q-table
        action = agent.get_action(current_state)

        if debug:
            print(f"current state: {current_state}")
            print(f"q_table: {agent.q_table[current_state['position'][0], current_state['position'][1]]}")
            print(f"action : {action}")

        next_state, reward_signal, is_done = maze.interact(action)

        # Give the agent a reward based on the reward signal. 
        # Penalties for hitting walls and taking steps.
        # Rewards for reaching the goal and other positive actions!
        if reward_signal in reward_map:
            reward = reward_map[reward_signal]
        else:
            reward = 0.0

        # Add the current position to the path if the agent has reached the goal or taken a step.
        if reward_signal != "wall":
            path.append(maze.current_position)
            #is_done = True
        #if reward_signal == "step":
            #path.append(maze.current_position)
        
        # Update the cumulative reward and step count for the episode
        episode_reward += reward
        episode_step += 1

        # Update the agent's Q-table if training is enabled
        if train == True:
            agent.update_q_table(current_state, action, next_state, reward)

    # Return the cumulative episode reward, total number of steps, and the agent's path during the simulation
    print(f"episode reward: {episode_reward}, episode step: {episode_step}")
    return episode_reward, episode_step, path


# This function evaluates an agent's performance in the maze. The function simulates the agent's movements in the maze,
# updating its state, accumulating the rewards, and determining the end of the episode when the agent reaches the goal position.
# The agent's learned path is then printed along with the total number of steps taken and the total reward obtained during the
# simulation. The function also visualizes the maze with the agent's path marked in blue for better visualization of the
# agent's trajectory.

def test_agent(agent, 
               maze, 
               plot=True, 
               debug=False):
    # Simulate the agent's behavior in the maze for the specified number of episodes
    episode_reward, episode_step, path = run_single_simulation(agent, maze, train=False, debug=debug)

    # Print the learned path of the agent
    print("Learned Path:")
    for row, col in path:
        print(f"({row}, {col})-> ", end='')
    print("Goal!")

    print("Number of steps:", episode_step)
    print("Total reward:", episode_reward)

    if plot:

        # Clear the existing plot if any
        if plt.gcf().get_axes():
            plt.cla()

        # Visualize the maze using matplotlib
        plt.figure(figsize=(maze.maze_width,maze.maze_height))
        plt.imshow(maze.maze, cmap='gray')

        # Mark the start position (red 'S') and goal position (green 'G') in the maze
        plt.text(maze.start_position[0], maze.start_position[1], 'S', ha='center', va='center', color='red', fontsize=20)
        plt.text(maze.goal_position[0], maze.goal_position[1], 'G', ha='center', va='center', color='green', fontsize=20)

        # Mark the agent's path with blue '#' symbols
        for position in path:
            plt.text(position[0], position[1], "#", va='center', color='blue', fontsize=20)

        # Remove axis ticks and grid lines for a cleaner visualization
        plt.xticks([]), plt.yticks([])
        plt.grid(color='black', linewidth=2)
        plt.show()

    return episode_step, episode_reward


"""
Below is the code for Q-learning, a basic reinforcement learning algorithm. This is used to train the agent. This code updates the Q-values based on the rewards it receives during exploration.  You do not need to change this code for your engineering project.
"""
def train_agent(agent, maze, reward_map):
    # Lists to store the data for plotting
    episode_rewards = []
    episode_steps = []

    # Loop over the specified number of episodes
    while not agent.terminate():
        episode_reward, episode_step, path = run_single_simulation(agent, maze, train=True, reward_map=reward_map)

        # Store the episode's cumulative reward and the number of steps taken in their respective lists
        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step)
        agent.new_episode()

        #print("testing agent : ")
        #test_agent(agent, maze, num_episodes=episode, plot=False)

    # Plotting the data after training is completed
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Reward per Episode')

    average_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"The average reward is: {average_reward}")

    plt.subplot(1, 2, 2)
    plt.plot(episode_steps)
    plt.xlabel('Episode')
    plt.ylabel('Steps Taken')
    #plt.ylim(0, 100)
    plt.title('Steps per Episode')

    average_steps = sum(episode_steps) / len(episode_steps)
    print(f"The average steps is: {average_steps}")

    plt.tight_layout()
    plt.show()