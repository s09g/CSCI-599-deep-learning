import numpy as np
from territorialise import territorialise
from player import Player
from player import *


class Board(object):
    def __init__(self, areas, paths, players, step_penalty=-0.5, termination_reward=1):
        assert [isinstance(p, Player) for p in players]
        self.areas = np.array(areas)
        self.paths = np.array(paths)
        assert self.areas.shape == self.paths.shape
        assert isinstance(players, set)
        self.players = {}
        self.prev_area_cnt = {}
        self.step_penalty = step_penalty
        self.termination_reward = termination_reward
        for player in players:
            i, j = player.head
            if self.paths[i][j] == player.id:
                self.players[player.id] = player
                self.prev_area_cnt[player.id] = np.sum(self.areas == player.id)
        self.suicide = set()
        self.centers = {p: self.get_center(p) for p in self.players}
        self.current_areas = {p: np.sum(self.areas == p) for p in self.players}

    # directions : {<player_id>: {UP, DOWN, LEFT, RIGHT}}
    def step(self, directions={}):
        N, M = self.areas.shape
        prev_heads = {p: self.players[p].head for p in self.players}
        kills = set()
        temp = {}
        new_pos = {}
        for p in self.players:
            pos = tuple(self.players[p].move_to(directions.get(p, None)))
            if pos in temp:
                kills |= temp[pos]
                kills.add(p)
            new_pos[p] = pos
            temp.setdefault(pos, set()).add(p)
        # check all the new positions, see whether new pos belongs to other's path
        # if yes, kill the other (or itself)
        for p in new_pos:
            i, j = new_pos[p]
            if not (0 <= i < N and 0 <= j < M):
                kills.add(p)
                self.suicide.add(p)
            elif self.paths[i][j] != 0:
                kills.add(self.paths[i][j])
                if self.paths[i][j] == p:
                    self.suicide.add(p)
        for p in kills:
            self.kill(p)
        kills.clear()
        # try to move to new positions
        need_re_cal = False
        for p in self.players:
            i0, j0 = prev_heads[p]
            if not np.any(self.areas == p):
                # This is probably because other player make p's head into its territory
                # i.e. p is being surrounded, and die
                kills.add(p)
                continue
            i, j = new_pos[p]
            assert self.paths[i][j] == 0
            if self.areas[i0][j0] == p:  # previous head is inside area
                self.paths[i0][j0] = 0
                self.paths[i][j] = p
            elif self.areas[i][j] != p:  # from outside to outside
                self.paths[i][j] = p
            else:  # from outside to inside -> making territory!!
                assert self.areas[i0][j0] != p and self.areas[i][j] == p
                all_path = self.paths == p
                self.paths[all_path] = 0
                self.areas[all_path] = p
                self.areas = territorialise(self.areas, p)
                # This will potentially kill all players within new territory
                self.paths[i][j] = p
                need_re_cal = True
        for p in self.players:
            if p not in kills and not np.any(self.areas == p):
                kills.add(p)
        for p in kills:
            self.kill(p)
        if need_re_cal:
            for p in self.players:
                self.centers[p] = self.get_center(p)
                self.current_areas[p] = np.sum(np.array(self.areas) == p)

    def kill(self, player_id):
        self.areas[self.areas == player_id] = 0
        self.paths[self.paths == player_id] = 0
        del self.players[player_id]
        del self.current_areas[player_id]
        del self.centers[player_id]

    def frame_step(self, actions={}):
        '''
        :param actions: a dictionary represent actions of each players
                        {<player_id> : {UP, DOWN, LEFT, RIGHT}}
        :return:
            areas: all areas of next timestamp
            paths: all paths of next timestamp
            step_penalty
            delta_area: a dictionary represent each player's current area - previous area
            termination_rewards: a dictionary represent each player's termination state;
                                 defined by self.termination_reward
        '''
        self.step(actions)
        cur_area_cnt = {}
        delta_area = {}
        dead = set()
        for p in actions:
            prev = self.prev_area_cnt.get(p, 0)
            cur_area_cnt[p] = np.sum(self.areas == p)
            delta_area[p] = cur_area_cnt[p] - prev
            if prev == 0 or cur_area_cnt[p] == 0:
                dead.add(p)
        termination_rewards = {p: 0 for p in actions}
        for p in dead:
            termination_rewards[p] = -self.termination_reward
            del cur_area_cnt[p]
        if len(cur_area_cnt) == 1:
            termination_rewards[list(cur_area_cnt.keys())[0]] = self.termination_reward
        self.prev_area_cnt = cur_area_cnt
        return self.areas, self.paths, self.step_penalty, delta_area, termination_rewards

    def get_center(self, player_id):
        idx = np.array(np.where(self.areas == player_id))
        return np.mean(idx, axis=1) if idx.shape[0] != 0 else None
    
    def get_distance(self, player_id):
        if player_id not in self.players:
            return None
        head = np.array(self.players[player_id].head)
        return np.linalg.norm(np.array(self.centers[player_id] - head))

    def get_view(self, player_id, pad=7, wall=100):
        if player_id not in self.players:
            return np.zeros((pad * 2 + 1, pad * 2 + 1)), np.zeros((pad * 2 + 1, pad * 2 + 1))
        i, j = self.players[player_id].head
        N, M = self.areas.shape
        if 0 <= i - pad and i + pad + 1 <= N and 0 <= j - pad and j + pad + 1 <= M:
            return self.areas[i - pad: i + pad + 1, j - pad: j + pad + 1], \
                   self.paths[i - pad: i + pad + 1, j - pad: j + pad + 1]
        temp_a = np.pad(self.areas, pad_width=pad, mode='constant', constant_values=wall)
        temp_p = np.pad(self.paths, pad_width=pad, mode='constant', constant_values=wall)
        return temp_a[i: i + 2 * pad + 1, j: j + 2 * pad + 1], temp_p[i: i + 2 * pad + 1, j: j + 2 * pad + 1]

if __name__ == '__main__':
    area = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 2, 0],
        [0, 0, 0, 0, 0, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    path = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    p1 = Player(id=1, head=(2, 2), direction=LEFT)
    p2 = Player(id=2, head=(5, 5), direction=RIGHT)
    board = Board(area, path, {p1, p2})
    move1 = [LEFT, DOWN, RIGHT, RIGHT, UP, UP, LEFT]
    move2 = [RIGHT, UP, LEFT, LEFT, DOWN, DOWN, RIGHT]

    area = board.areas
    path = board.paths
    out = np.where(path != 0, path, area * 10)
    print(out)
    print()

    for move in zip(move1, move2):
        board.step({1: move[0], 2: move[1]})
        print(board.players)
        area = board.areas
        path = board.paths
        out = np.where(path != 0, path, area * 10)
        print(out)
        print()
