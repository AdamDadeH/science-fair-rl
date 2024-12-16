
from mazes.basic_maze import Maze



class KeyMaze(Maze):
    def __init__(self, 
                 maze_specification, 
                 start_position, 
                 goal_position, 
                 key_position):
        super().__init__(maze_specification, start_position, goal_position)
        self.key_position = key_position  # Set the key position in the maze as a tuple (x, y)
        self.has_key = False  # Initialize key status

    def reset(self):
        self.current_position = self.start_position
        self.has_key = False

    def get_current_state(self):
        return {
            "position": self.current_position,
            "has_key": self.has_key
        }

    def interact(self, action):
        action_step = self.action_step_map[action]
        next_position = [self.current_position[0] + action_step[0], self.current_position[1] + action_step[1]]

        if self.wall_check(next_position):            
            return self.get_current_state(), "wall", False
        
        # Check if the agent reached the goal
        if self.has_key and self.goal_check(next_position):
            self.current_position = next_position
            return self.get_current_state(), "goal", True
        
        if not self.has_key and self.goal_check(next_position):
            self.current_position = next_position
            return self.get_current_state(), "goal_locked", False

        if next_position == self.key_position:
            self.current_position = next_position
            if not self.has_key:
                print(f"GOT THE KEY! at {self.current_position} - matching {self.key_position}")
            self.has_key = True  # Collect the key
            reward_signal = "got_key"
            return self.get_current_state(), "got_key", False

        self.current_position = next_position
        return { "position" : next_position, "has_key" : self.has_key }, "step", False


        
