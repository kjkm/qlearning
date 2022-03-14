# cython: boundscheck=False

import numpy as np
cimport numpy as np
import random

np.import_array()

# Prints a given q_table. Note that each cell of the printed table contains the sum of the calculated reward values
# for all possible actions in the given location. That is, the printed q_table is the point-wise sum of five separate
# q_tables -- one for each action. This leads to numbers that look larger than they should be, but in fact the actual
# values within each individual q_table falls within expected bounds.
cpdef print_table(np.ndarray[np.float32_t, ndim=3] table):
    cdef str line = ""
    cdef int i, j
    for i in range(table.shape[0]):
        line = ""
        for j in range(table.shape[1]):
            line += '{:4}'.format(int(np.sum(table[i][j]))) + " "
        print(line)


# Generates a list of all possible legal moves from a given location.
cdef generate_moves(int x, int y, int height, int width):
    cdef np.ndarray moves = np.full((5, 2), -1, dtype=np.dtype("i"), order="C")
    moves[0][0] = x
    moves[0][1] = y
    if x - 1 >= 0:
        moves[1][0] = x - 1
        moves[1][1] = y
    if x + 1 < height:
        moves[2][0] = x + 1
        moves[2][1] = y
    if y - 1 >= 0:
        moves[3][0] = x
        moves[3][1] = y - 1
    if y + 1 < width:
        moves[4][0] = x
        moves[4][1] = y + 1
    return moves


# Using currently known information, calculates the best move to take. Occasionally takes a random action instead,
# proportional to epsilon.
cdef int next_move(np.ndarray[np.float32_t, ndim=3] q_table, int x, int y, np.ndarray[np.int32_t, ndim=2] moves, float epsilon):
    cdef float p = random.uniform(0, 1)
    cdef int i
    cdef float max_val = -100
    cdef int max_index = 0
    cdef int rand_move

    if p >= epsilon:
        for i in range(5):
            if moves[i][0] >= 0:
                if max_val < q_table[x][y][i]:
                    max_val = q_table[x][y][i]
                    max_index = i
        return max_index
    else:
        rand_move = random.randrange(5)
        while moves[rand_move][0] <= -1:
            rand_move = random.randrange(5)
        return rand_move


# Recursively traverses across the board, making decisions informed by the Q_table, game world/reward table, and the epsilon-greedy algorithm
cpdef np.ndarray learn(np.ndarray[np.int32_t, ndim=2] world, np.ndarray[np.float32_t, ndim=3] q_table, int x, int y, int goal_value, int lifespan, float upsilon, float epsilon):
    # stop if random location is the goal
    if world[x][y] == goal_value or lifespan == 0:
        return q_table

    # Choose a move
    cdef np.ndarray moves = generate_moves(x, y, world.shape[0], world.shape[1])
    cdef int chosen_move = next_move(q_table, x, y, moves, epsilon)

    # Value of just that move
    cdef float value = world[moves[chosen_move][0]][moves[chosen_move][1]]

    # Value of future move
    learn(world, q_table, moves[chosen_move][0], moves[chosen_move][1], goal_value, lifespan - 1, upsilon, epsilon)
    cdef int future_index = np.argmax(q_table[moves[chosen_move][0]][moves[chosen_move][1]])
    cdef float future_value = q_table[moves[chosen_move][0]][moves[chosen_move][1]][future_index]

    # Set value of current move
    q_table[x][y][chosen_move] = value + (upsilon * future_value)

    # return
    return q_table


# Given a board, q_table, and starting position, determine the single best move to take. Returns a coordinate pair.
cpdef move(np.ndarray[np.int32_t, ndim=2] world, np.ndarray[np.float32_t, ndim=3] q_table, int x, int y):
    cdef np.ndarray moves = generate_moves(x, y, world.shape[0], world.shape[1])
    cdef int best_index = np.argmax(q_table[x][y])
    return moves[best_index][0], moves[best_index][1]
