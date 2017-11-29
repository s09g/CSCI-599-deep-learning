from board import Board
from player import Player
import player
import numpy as np
from Visualization import show

def init_head_area(areas, paths, p, padding=1):
    assert isinstance(p, Player)
    i, j = p.head
    areas[i - padding:i + padding + 1, j - padding:j + padding + 1] = p.id
    paths[i][j] = p.id


def get_swap(m, a, b):
    idx_a = (m == a)
    res = np.where(m == b, a, m)
    res[idx_a] = b
    return res


class PaperIO2Wrapper(object):
    def __init__(self, N=50, step_penalty=0.1, termination_reward=50):
        self.p1 = self.p2 = self.board = None
        self.reset(N, step_penalty, termination_reward)

    def reset(self, N=50, step_penalty=0.1, termination_reward=50):
        areas = np.zeros((N, N))
        paths = np.zeros((N, N))
        self.p1 = Player(id=1, head=(10, 10), direction=player.DOWN)
        self.p2 = Player(id=2, head=(N-11, N-11), direction=player.UP)
        init_head_area(areas, paths, self.p1)
        init_head_area(areas, paths, self.p2)
        self.board = Board(areas=areas,
                           paths=paths,
                           players={self.p1, self.p2},
                           step_penalty=step_penalty,
                           termination_reward=termination_reward)

    def get_reversed_board(self):
        """
        :return: return a reversed board instance such that player 1 and 2 are
                 completely reversed in board.areas, board.paths and board.players
        """
        rev_p1 = Player(id=1, head=self.p2.head, direction=self.p2.dir)
        rev_p2 = Player(id=2, head=self.p1.head, direction=self.p1.dir)
        rev_areas = get_swap(self.board.areas, self.p1.id, self.p2.id)
        rev_paths = get_swap(self.board.paths, self.p1.id, self.p2.id)
        return Board(rev_areas, rev_paths, {rev_p1, rev_p2}, self.board.step_penalty, self.board.termination_reward)

    def frame_step(self, actions):
        res = self.board.frame_step(actions)
        termination_rewards = res[-1]
        if len(self.board.suicide) == 1:
            dead = list(self.board.suicide)[0]
            alive = 1 if dead == 2 else 2
            termination_rewards[alive] = 0
        # Assuming if only one player left, restart the game
        if len(self.board.players) < 2:
            self.reset(N=self.board.areas.shape[0],
                       step_penalty=self.board.step_penalty,
                       termination_reward=self.board.termination_reward)
        return res

    def get_player2_view(self):
        area = get_swap(self.board.areas, self.p1.id, self.p2.id)
        path = get_swap(self.board.paths, self.p1.id, self.p2.id)
        area = np.rot90(area, 2)
        path = np.rot90(path, 2)
        return area, path

    def show_board(self):
        show(self.board.areas, self.board.paths, self.board.players, 50)

if __name__ == '__main__':
    game = PaperIO2Wrapper(10)
    print(get_swap(game.board.areas, 1, 2))
    print(game.get_reversed_board().areas)
