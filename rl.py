import sys

import q_learning
import numpy as np
import random
import time

BOARD_HEIGHT = 10
BOARD_WIDTH = 15
GOAL_COORDINATES = (7, 12)

NUMBER_OF_ACTIONS = 5  # Hard coded in q_learning module - DO NOT CHANGE
ITERATIONS = 10000  # Number of random starting positions sampled
LIFESPAN = 100  # Maximum number of actions a single iteration will take

UPSILON = 0.9  # Discount factor of future rewards
EPSILON = 0.5  # Rate of random exploration

GOAL_REWARD = 100
EMPTY_REWARD = -1


def main():
    world: np.ndarray = np.full((BOARD_HEIGHT, BOARD_WIDTH), EMPTY_REWARD, dtype=np.dtype("i"), order="C")
    q_table: np.ndarray = np.zeros((BOARD_HEIGHT, BOARD_WIDTH, NUMBER_OF_ACTIONS), dtype=np.dtype("f"), order="C")
    world[GOAL_COORDINATES[0]][GOAL_COORDINATES[1]] = GOAL_REWARD

    print("Learning...")
    start_time = time.time()
    for i in range(ITERATIONS):
        # Generate random starting position for this iteration
        x = random.randrange(BOARD_HEIGHT)
        y = random.randrange(BOARD_WIDTH)

        # Conduct learning and update q_table for this iteration
        q_table = q_learning.learn(world, q_table, x, y, GOAL_REWARD, LIFESPAN, UPSILON, EPSILON)

        # Loading bar
        loading = "["
        percent_done = int((i / ITERATIONS) * 100)
        portion = int(percent_done / 5)
        for j in range(portion):
            loading += "|"
        for j in range(20 - portion):
            loading += "="
        loading += "]"
        sys.stdout.write(f"\r {loading} {percent_done}% done")
        sys.stdout.flush()
    print()
    finish_time = time.time()
    print(f"Learning Finished! {ITERATIONS} iterations completed in {format(finish_time - start_time, '.4f')} seconds.")
    q_learning.print_table(q_table)


if __name__ == "__main__":
    main()
