from mazes.Maze_1 import Maze


import numpy as np


def get_first_example_maze():
    maze_layout = np.array([
        [0, 1, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ])

    # Create an instance of the maze and set the starting and ending positions
    maze = Maze(maze_layout, (0, 0), (4, 4))
    return maze


def prims_maze(size):
    maze = np.ones((size, size))
    start = (0, 0)
    maze[start[1]][start[0]] = 0

    # List of walls to consider
    walls = [(0,1), (1,0)]

    while walls:
        wall = walls.pop(np.random.randint(len(walls)))
        x, y = wall

        # Count number of visited neighbors
        neighbors = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
        visited = sum(1 for nx, ny in neighbors
                     if 0 <= nx < size and 0 <= ny < size
                     and maze[ny][nx] == 0)

        if visited == 1:  # Only one visited neighbor
            maze[y][x] = 0
            for nx, ny in neighbors:
                if (0 <= nx < size and 0 <= ny < size
                    and maze[ny][nx] == 1):
                    walls.append((nx, ny))

    maze[size-1][size-1] = 0
    return Maze(maze, (0,0), (size-1,size-1))


def random_maze(size):
    k = 0
    max_attempts = 10000  # Prevent infinite loop
    while k < max_attempts:
        k += 1
        maze = np.random.randint(2, size=(size, size))
        # Ensure start and goal positions are open
        maze[0,0] = 0
        maze[size-1,size-1] = 0

        # Create maze instance
        start_position = (0, 0)
        goal_position = (size-1, size-1)
        try:
            maze_instance = Maze(maze, start_position, goal_position)
            return maze_instance
        except ValueError:
            continue