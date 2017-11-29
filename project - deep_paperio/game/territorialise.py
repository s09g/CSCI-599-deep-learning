import numpy as np
from collections import deque

DIR = [0, 1, 0, -1, 0]

TEST_BOARD = [
    [2, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 1, 0],
    [0, 0, 1, 1, 0, 1, 0],
    [0, 0, 1, 1, 0, 1, 0],
    [0, 0, 1, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0]
]

TEST_BOARD_1 = [
    [0, 0, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 0],
    [1, 1, 0, 0, 1, 1],
    [1, 1, 0, 0, 1, 1],
    [0, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 0, 0],
]

def search_boundary(board, id):
    N, M = len(board), len(board[0])
    for i in range(N):
        if board[i][0] != id:
            return i, 0
        if board[i][M-1] != id:
            return i, M-1
    for j in range(M):
        if board[0][j] != id:
            return 0, j
        if board[N-1][j] != id:
            return N-1, j
    return None


def dfs(board, i, j, num, place_holder=-1):
    if not (0 <= i < len(board) and 0 <= j < len(board[0])) or board[i][j] in {num, place_holder}:
        return
    board[i][j] = place_holder
    for d in range(4):
        dfs(board, i+DIR[d], j+DIR[d+1], num, place_holder)


def bfs(board, i, j, num, place_holder=-1):
    sett = {num, place_holder}
    if not (0 <= i < len(board) and 0 <= j < len(board[0])) or board[i][j] in sett:
        return
    N, M = board.shape
    queue = deque()
    board[i][j] = place_holder
    queue.append(i * M + j)
    while queue:
        top = queue.popleft()
        i, j = top // M, top % M
        for d in range(4):
            ii, jj = i+DIR[d], j+DIR[d+1]
            if 0 <= ii < N and 0 <= jj < M and board[ii][jj] not in sett:
                board[ii][jj] = place_holder
                queue.append(ii * M + jj)


def territorialise(board, num):
    boolean_matrix = (np.array(board) == num)
    sum0 = np.cumsum(boolean_matrix, axis=0, dtype=bool)
    sum1 = np.cumsum(boolean_matrix, axis=1, dtype=bool)
    sum0_1 = np.flipud(np.cumsum(np.flipud(boolean_matrix), axis=0, dtype=bool))
    sum1_1 = np.fliplr(np.cumsum(np.fliplr(boolean_matrix), axis=1, dtype=bool))
    return np.where(sum0 & sum1 & sum0_1 & sum1_1, num, board)


if __name__ == '__main__':
    board = np.array(TEST_BOARD)
    board = territorialise(board, 1)
    print(board)
    board = np.array(TEST_BOARD_1)
    board = territorialise(board, 1)
    print(board)
