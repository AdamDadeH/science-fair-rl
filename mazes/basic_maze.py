from mazes.env import Environment


import matplotlib.pyplot as plt
import numpy as np


class Maze(Environment):
    """
    This represents the maze and the current position within the maze.

    This enables
     - get_current_state() -> Where are we now?
     - get_actions() -> what can we do? (go up, down, right, left)   
     - interact(action) -> 
        try to take the given action, either moving in the desired direction or running into a wall 
        lets us know new position and whether we hit a wall / took a step / reached the goal
    
    
    """

    def __init__(self,
                 maze_specification,
                 start_position,
                 goal_position):
        
        # Initialize Maze object with the provided maze, start_position, and goal position
        self.maze = maze_specification
        self.maze_height = self.maze.shape[0] # Get the height of the maze (number of rows)
        self.maze_width = self.maze.shape[1]  # Get the width of the maze (number of columns)
        
        self.start_position = start_position    # Set the start position in the maze as a tuple (x, y)
        self.goal_position = goal_position      # Set the goal position in the maze as a tuple (x, y)
        self.current_position = start_position # Set the current position in the maze as a tuple (x, y)
        self.action_step_map = {"down": (0, 1), "up": (0, -1), "left": (-1, 0), "right": (1, 0)}
        # check that there is a path from start to end
        self.__validate_maze()

    def reset(self):
        """
        Reset the envirnoment (the maze) by resetting the characters position back to the start_position.
        """
        self.current_position = self.start_position

    def get_current_state(self):
        """
        Return the current position of the character.
        """
        return {"position" : self.current_position}

    def get_actions(self):
        """
        Return the actions that are possible for the current state.

        For this maze we let the agent attempt to go 
        up, down, left, right even if there is a wall in the way.
        """
        return ["down", "up", "left", "right"]

    def wall_check(self, next_position):
        """
        Is the position we are going to a wall?
        """
        return (next_position[0] < 0 or next_position[0] >= self.maze_height or
            next_position[1] < 0 or next_position[1] >= self.maze_width or
            self.maze[next_position[1]][next_position[0]] == 1)

    def goal_check(self, next_position):
        """
        Did we reacht the goal!!
        """
        return next_position[0] == self.goal_position[0] and next_position[1] == self.goal_position[1]

    def interact(self, action):
        """
        Interact with the maze environment by taking an action and returning the next state, reward, and whether the episode is done.

        Args:
            action: Name of the action to take.

        Returns:
            next_state: The resulting position after taking the action
            reward: The reward received for taking the action 
            is_done: Whether the episode is complete (reached goal)
        """
        action_step = self.action_step_map[action]
        next_position = [self.current_position[0] + action_step[0], self.current_position[1] + action_step[1]]

        # Check if the next state is out of bounds or hitting a wall
        if self.wall_check(next_position):
            reward_signal = "wall" # Wall penalty
            is_done = False
        # Check if the agent reached the goal
        elif self.goal_check(next_position):
            reward_signal = "goal" # Goal reward
            self.current_position = next_position
            is_done = True
        # The agent takes a valid step but hasn't reached the goal yet
        else:
            reward_signal = "step" # Small penalty for each step
            self.current_position = next_position
            is_done = False

        return {"position" : self.current_position}, reward_signal, is_done


    def __validate_maze(self):
        # Validate the maze is a 2D array
        if not isinstance(self.maze, np.ndarray) or self.maze.ndim != 2:
            raise ValueError("Maze must be a 2D array")
        # Validate the start and goal positions are within the maze bounds
        if not (0 <= self.start_position[0] < self.maze_height and 0 <= self.start_position[1] < self.maze_width):
            raise ValueError("Start position is out of bounds")
        if not (0 <= self.goal_position[0] < self.maze_height and 0 <= self.goal_position[1] < self.maze_width):
            raise ValueError("Goal position is out of bounds")
        # Validate the start and goal positions are not on a wall
        if self.maze[self.start_position[1]][self.start_position[0]] == 1:
            raise ValueError("Start position is on a wall")
        if self.maze[self.goal_position[1]][self.goal_position[0]] == 1:
            raise ValueError("Goal position is on a wall")
        # Validate the maze has a path from the start to the goal
        if not self.__has_path(self.start_position, self.goal_position):
            raise ValueError("Maze does not have a path from start to goal")

    def __has_path(self, start, goal):
        # Implement a pathfinding algorithm to check if there is a path from start to goal
        # For simplicity, we'll use a simple breadth-first search (BFS)
        # Create a queue for BFS and a set to track visited positions
        queue = [(start[0], start[1])]
        visited = {(start[0], start[1])}

        while queue:
            current = queue.pop(0)

            # If we reached the goal, a path exists
            if current == (goal[0], goal[1]):
                return True

            # Check all 4 adjacent positions
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                next_x = current[0] + dx
                next_y = current[1] + dy
                next_pos = (next_x, next_y)

                # Check if position is valid and unvisited
                if (0 <= next_x < self.maze_width and
                    0 <= next_y < self.maze_height and
                    self.maze[next_y][next_x] == 0 and
                    next_pos not in visited):

                    queue.append(next_pos)
                    visited.add(next_pos)

        # No path found
        return False


    def show_maze(self):
        # Visualize the maze using Matplotlib
        plt.figure(figsize=(self.maze_width,self.maze_height))

        # Display the maze as an image in grayscale ('gray' colormap)
        plt.imshow(self.maze, cmap='gray')

        # Add start and goal positions as 'S' and 'G'
        plt.text(self.start_position[0], self.start_position[1], 'Start', ha='center', va='center', color='red', fontsize=20)
        plt.text(self.goal_position[0], self.goal_position[1], 'Goal', ha='center', va='center', color='green', fontsize=20)

        # Remove ticks and labels from the axes
        plt.xticks([]), plt.yticks([])

        # Show the plot
        plt.show()