import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from board import *
import player
from player import *

area_colors = {0: (255, 255, 255), 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255),
               4: (255, 255, 0), 5: (255, 0, 255), 6: (0, 255, 255)}
path_colors = {0: (255, 255, 255), 1: (128, 0, 0), 2: (0, 128, 0), 3: (0, 0, 128),
               4: (128, 128, 0), 5: (128, 0, 128), 6: (0, 128, 128)}

def show(area, path, players, size):
    width = size * 10
    img = np.zeros((width, width, 3), np.uint8)
    for i in range(size):
        for j in range(size):
            img = cv.rectangle(img, (j * 10, i * 10), (j * 10 + 9, i * 10 + 9), area_colors[area[i][j]], -1)
    for i in range(size):
        for j in range(size):
            if path[i][j] != 0:
                img = cv.rectangle(img, (j * 10, i * 10), (j * 10 + 9, i * 10 + 9), path_colors[path[i][j]], -1)
    for val in players.values():
        x, y = val.head
        img = cv.rectangle(img, (y * 10, x * 10), (y * 10 + 9, x * 10 + 9), (0, 0, 0), 1)
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    area = [
                [2, 2, 2, 0, 0],
                [2, 2, 2, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1],
            ]
    path = [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [2, 2, 0, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
    p1 = Player(id=1, head=(2, 4), direction=player.UP)
    p2 = Player(id=2, head=(2, 1), direction=player.DOWN)
    board = Board(area, path, {p1, p2})
    show(board.areas, board.paths, board.players, 5)
    board.step()
    show(board.areas, board.paths, board.players, 5)
