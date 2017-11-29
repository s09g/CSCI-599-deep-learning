import numpy as np

import player
from paperio import PaperIO

def get_swap(m, a, b):
    idx_a = (m == a)
    res = np.where(m == b, a, m)
    res[idx_a] = b
    return res

class PaperIOTwoWrapper(object):
    def __init__(self, N=50, id1=1, id2=2, h1=None, h2=None, init_pad=1, is_two_player=True):
        self.N = N
        self.id1 = id1
        self.id2 = id2
        self.is_two_player_mode = is_two_player
        self.reset(h1, h2, init_pad)

    def step(self, dirs):
        self.prev_area = dict(self.game.board.current_areas)
        self.game.step(dirs)
        self.prev_dis = self.cur_dis
        self.cur_dis = {p: self.game.board.get_distance(p) for p in self.players}

    def reset(self, h1=None, h2=None, init_pad=1):
        i1, i2, j1, j2 = 0, 0, 0, 0
        while abs(i1 - i2) < 2 * init_pad + 3 or abs(j1 - j2) < 2 * init_pad + 3:
            # avoid two players are too close or at the same position
            i1, i2 = np.random.choice(range(init_pad, self.N - init_pad), 2, replace=False)
            j1, j2 = np.random.choice(range(init_pad, self.N - init_pad), 2, replace=False)

        h1 = np.array([i1, j1]) if h1 is None else np.array(h1)
        h2 = np.array([i2, j2]) if h2 is None else np.array(h2)
        assert h1.shape == h2.shape == (2,)
        self.game = PaperIO(self.N)
        self.game.add_player(p_id=self.id1, head=h1, init_dir=player.get_rand_dir(), pad=init_pad)
        if self.is_two_player_mode:
            self.game.add_player(p_id=self.id2, head=h2, init_dir=player.get_rand_dir(), pad=init_pad)
        self.game.init_board()
        # shadow copy
        self.players = self.game.board.players
        self.prev_dis = dict()
        self.cur_dis = dict()
        for p in self.game.board.players:
            self.prev_dis[p] = self.cur_dis[p] = self.game.board.get_distance(p)
        self.prev_area = dict(self.game.board.current_areas)

    def is_in_area(self, p_id):
        return self.game.is_in_area(p_id)

    def delta_dis(self, p_id):
        return self.cur_dis[p_id] - self.prev_dis[p_id]

    def get_delta_area(self, p_id):
        return self.game.board.current_areas[p_id] - self.prev_area[p_id]

    def get_delta_d_reward(self, p_id):
        delta_d = self.cur_dis[p_id] - self.prev_dis[p_id]
        return delta_d if self.is_in_area(p_id) else -2 * delta_d

    def is_dead(self, p_id):
        return self.game.is_dead(p_id)

    def get_views(self, p_id):
        area, path = self.game.board.get_view(p_id)
        another_p = self.id2 if p_id == self.id1 else self.id1
        if p_id == 1 and another_p == 2:
            return area, path
        elif p_id == 2 and another_p == 1:
            return get_swap(area, 1, 2), get_swap(path, 1, 2)
        else:
            res = []
            for m in area, path:
                mask0 = (m == p_id)
                mask1 = (m == another_p)
                ans = np.where(mask0, 1, m)
                ans[mask1] = 2
                res.append(ans)
            return res[0], res[1]
