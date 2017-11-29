from board import Board
import player
from player import Player
import unittest
import numpy as np


class TestBoard(unittest.TestCase):

    def test_make_territory(self):
        area = np.zeros((3, 3))
        path = np.zeros((3, 3))
        area[0][0] = path[0][1] = 1
        board = Board(area, path, {Player(1, (0, 1), player.RIGHT)})
        for d in [player.RIGHT, player.DOWN, player.DOWN, player.LEFT, player.LEFT, player.UP, player.UP]:
            board.step({1: d})
        expected_path = np.zeros((3, 3))
        expected_path[0][0] = 1
        self.assertTrue(np.allclose(board.paths, expected_path))
        self.assertTrue(np.allclose(board.areas, 1), "{}".format(board.areas))

    def test_invade_territory(self):
        area = [
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
        path = [
            [0, 0, 2, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        p1 = Player(id=1, head=(2, 4), direction=player.DOWN)
        p2 = Player(id=2, head=(0, 2), direction=player.RIGHT)
        board = Board(area, path, {p1, p2})
        board.step()
        expected_area = [
            [2, 2, 2, 2, 2],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
        expected_path = [
            [0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]
        self.assertTrue(np.allclose(board.areas, expected_area))
        self.assertTrue(np.allclose(board.paths, expected_path))

    def test_defend_territory(self):
        area = [
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
        path = [
            [0, 0, 2, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        p1 = Player(id=1, head=(2, 4), direction=player.DOWN)
        p2 = Player(id=2, head=(0, 2), direction=player.DOWN)
        board = Board(area, path, {p1, p2})
        board.step()
        expected_area = [
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        expected_path = [
            [0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        self.assertTrue(np.allclose(board.areas, expected_area))
        self.assertTrue(np.allclose(board.paths, expected_path))

    def test_separate_boundary(self):
        area = [
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 0],
        ]
        path = [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
        p1 = Player(id=1, head=(1, 1), direction=player.RIGHT)
        board = Board(area, path, {p1})
        board.step()
        expected_area = [
            [0, 0, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 0],
        ]
        expected_path = [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
        self.assertTrue(np.allclose(board.areas, expected_area))
        self.assertTrue(np.allclose(board.paths, expected_path))

    def test_surround_other_player(self):
        area = [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 2, 2, 0, 0],
            [0, 0, 2, 2, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
        ]
        path = [
            [1, 1, 1, 1, 1, 1],
            [1, 2, 2, 2, 2, 1],
            [1, 2, 0, 0, 2, 1],
            [1, 2, 0, 0, 2, 1],
            [0, 0, 2, 2, 2, 1],
            [0, 0, 1, 1, 1, 1],
        ]
        p1 = Player(id=1, head=(3, 0), direction=player.DOWN)
        p2 = Player(id=2, head=(3, 1), direction=player.RIGHT)
        board = Board(area, path, {p1, p2})
        board.step()
        expected_area = np.ones((6, 6), dtype=int)
        expected_path = np.zeros((6, 6), dtype=int)
        expected_path[4][0] = 1
        self.assertTrue(np.allclose(board.areas, expected_area))
        self.assertTrue(np.allclose(board.paths, expected_path))

    def test_head_collision(self):
        area = np.zeros((3, 3))
        path = np.zeros((3 ,3))
        area[1][0] = path[0][0] = 1
        area[1][2] = path[0][2] = 2
        area[2][1] = path[1][1] = 3
        p1 = Player(id=1, head=(0, 0), direction=player.RIGHT)
        p2 = Player(id=2, head=(0, 2), direction=player.LEFT)
        p3 = Player(id=3, head=(1, 1), direction=player.UP)
        board = Board(area, path, {p1, p2, p3})
        board.step()
        self.assertTrue(np.alltrue(board.areas == 0))
        self.assertTrue(np.alltrue(board.paths == 0))
        self.assertFalse(board.players)

    def test_out_of_boundary(self):
        for d in player.DIRECTIONS:
            area = np.ones((1, 1))
            path = np.ones((1, 1))
            board = Board(area, path, {Player(1, (0, 0), d)})
            board.step({})
            self.assertTrue(np.allclose(board.areas, 0))
            self.assertTrue(np.allclose(board.paths, 0))
            self.assertEqual(len(board.players), 0)

    def test_separate_area(self):
        area = [
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
        ]
        path = [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
        p = Player(id=1, head=(3, 3), direction=player.RIGHT)
        board = Board(area, path, {p})
        board.step()
        expected_area = [
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
        ]
        expected_path = [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
        self.assertTrue(np.allclose(board.areas, expected_area))
        self.assertTrue(np.allclose(board.paths, expected_path))

    def test_kill_itself(self):
        area = np.zeros((3, 3))
        path = np.zeros((3, 3))
        expected_path = np.zeros((3, 3))
        expected_area = np.zeros((3, 3))
        expected_area[0][0] = area[0][0] = expected_path[1][1] = path[1][1] = 1
        p = Player(id=1, head=(1, 1), direction=player.RIGHT)
        board = Board(area, path, {p})
        board.step({1: player.RIGHT})
        expected_path[1][2] = 1
        self.assertTrue(np.allclose(board.areas, expected_area))
        self.assertTrue(np.allclose(board.paths, expected_path))
        self.assertTrue(np.allclose(board.players[1].head, (1, 2)))
        board.step({1: player.DOWN})
        expected_path[2][2] = 1
        self.assertTrue(np.allclose(board.areas, expected_area))
        self.assertTrue(np.allclose(board.paths, expected_path))
        self.assertTrue(np.allclose(board.players[1].head, (2, 2)))
        board.step({1: player.LEFT})
        expected_path[2][1] = 1
        self.assertTrue(np.allclose(board.areas, expected_area))
        self.assertTrue(np.allclose(board.paths, expected_path))
        self.assertTrue(np.allclose(board.players[1].head, (2, 1)))
        board.step({1: player.UP})
        expected_path *= 0
        expected_area *= 0
        self.assertTrue(np.allclose(board.areas, expected_area))
        self.assertTrue(np.allclose(board.paths, expected_path))
        self.assertEqual(len(board.players), 0)

    def test_large_board(self):
        """Mainly for test whether it is able to handle large board"""
        N = 500
        area = np.zeros((N, N))
        path = np.zeros((N, N))
        area[0][0] = 1
        path[0][1] = path[1][1] = path[1][0] = 1
        p1 = Player(id=1, head=(1, 0), direction=player.UP)
        board = Board(area, path, {p1})
        board.frame_step()
        expected_area = np.zeros((N, N))
        expected_area[0:2, 0:2] = 1
        expected_path = np.zeros((N, N))
        expected_path[0][0] = 1
        self.assertTrue(np.allclose(board.areas, expected_area))
        self.assertTrue(np.allclose(board.paths, expected_path))


if __name__ == '__main__':
    unittest.main()