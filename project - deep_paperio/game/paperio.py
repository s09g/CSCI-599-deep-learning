from board import Board
from player import Player
import numpy as np
from Visualization import show


'''Some basic abstraction of PaperIO game. Able to add player individually and initialize game 
'''
class PaperIO(object):
    def __init__(self, N=50):
        self.N = N
        self.area = np.zeros((N, N))
        self.path = np.zeros((N, N))
        self.players = dict()
        self.board = None

    def add_player(self, p_id, head, init_dir, pad=1):
        self.players[p_id] = Player(id=p_id, head=head, direction=init_dir)
        i, j = head
        N = self.N
        self.area[max(0, i - pad): min(N, i + pad + 1), max(0, j - pad): min(N, j + pad + 1)] = p_id
        self.path[i][j] = p_id

    def init_board(self):
        self.board = Board(self.area, self.path, set(self.players.values()))

    def step(self, dirs):
        if self.board:
            self.board.step(dirs)

    def is_in_area(self, p):
        if p not in self.board.players:
            return False
        i, j = self.players[p].head
        return self.board.areas[i][j] == p

    def is_dead(self, p):
        return p not in self.board.players

    def get_views(self, p):
        return self.board.get_view(p)

    def show_board(self):
        show(self.board.areas, self.board.paths, self.board.players, self.N)
