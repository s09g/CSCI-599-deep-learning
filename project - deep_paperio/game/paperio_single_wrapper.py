import numpy as np

import player
from paperio import PaperIO


"""A wrapper only deal with single player. 
When there is only one player, we can easily keep track of many things
Also provide reset method. 
"""
class PaperIOSingleWrapper(object):
    def __init__(self, N=50, init_head=None, init_pad=5, id=1):
        self.player_id = id
        self.game = PaperIO(N)
        i, j = init_head if init_head is not None else np.random.randint(2, N - 2), np.random.randint(2, N - 2)
        self.game.add_player(p_id=self.player_id, head=(i, j), init_dir=player.UP, pad=init_pad)
        self.game.init_board()
        self.prev_dis = self.cur_dis = self.game.board.get_distance(self.player_id)
        self.prev_area = np.sum(self.game.board.areas == self.player_id)

    def step(self, d):
        self.prev_area = self.game.board.current_areas[self.player_id]
        self.game.step(d)
        self.prev_dis = self.cur_dis
        self.cur_dis = self.game.board.get_distance(self.player_id)

    def reset(self, init_pad=5, init_head=None):
        N = self.game.N
        self.game = PaperIO(N)
        i, j = init_head if init_head is not None else np.random.randint(2, N - 2), np.random.randint(2, N - 2)
        direction = np.random.randint(0, 4)
        self.game.add_player(self.player_id, head=(i, j), init_dir=direction, pad=init_pad)
        self.game.init_board()
        self.prev_dis = 0
        self.cur_dis = 0
        self.prev_area = self.game.board.current_areas[self.player_id]

    def is_in_area(self):
        return self.game.is_in_area(self.player_id)

    def delta_dis(self):
        return self.cur_dis - self.prev_dis

    def get_delta_area(self):
        return self.game.board.current_areas[self.player_id] - self.prev_area

    def get_delta_d_reward(self):
        delta_d = self.cur_dis - self.prev_dis
        if self.is_in_area():
            return delta_d
        else:
            return -delta_d * 2

    def is_dead(self):
        return self.game.is_dead(self.player_id)

    def get_views(self):
        return self.game.board.get_view(self.player_id)
