import numpy as np

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

DIRECTIONS = {
    UP: [-1, 0],
    DOWN: [1, 0],
    LEFT: [0, -1],
    RIGHT: [0, 1]
}

DIR_STR = {
    UP: "up",
    DOWN: "down",
    LEFT: "left",
    RIGHT: "right"
}

VERTICAL = {UP, DOWN}
HORIZONTAL = {LEFT, RIGHT}


def get_rand_dir():
    return np.random.randint(0, 4)


def get_valid_move(prev, next):
    if next not in DIRECTIONS:
        # This will include the case when next is None.
        return prev
    elif next != prev and ((next in VERTICAL and prev in VERTICAL) or (next in HORIZONTAL and prev in HORIZONTAL)):
        # prev and next are in the opposite direction
        return prev
    else:
        return next


def get_opposite_dir(d):
    if d in VERTICAL:
        return UP if d == DOWN else DOWN
    elif d in HORIZONTAL:
        return LEFT if d == RIGHT else RIGHT
    return None


class Player(object):
    # direction = {UP, DOWN, LEFT, RIGHT}
    def __init__(self, id, head, direction):
        self.id = id
        self.head = head
        self.dir = direction

    def move_to(self, new_dir):
        self.dir = get_valid_move(self.dir, new_dir)
        self.head = np.add(self.head, DIRECTIONS[self.dir])
        return self.head

    def __repr__(self):
        return "{}: {}".format(self.head, DIR_STR[self.dir])

    def __str__(self):
        return "{}: {}".format(self.head, DIR_STR[self.dir])
