import unittest

import numpy as np

from FieldMarkings.run import blend
from match_analytics import MatchAnalytics


def frame(ball_xy=None, player_team=1, player_bbox=(80, 40, 120, 140)):
    ball = {}
    if ball_xy is not None:
        x, y = ball_xy
        ball = {1: {"bbox": [x - 3, y - 3, x + 3, y + 3]}}
    return {
        "players": {7: {"bbox": list(player_bbox), "team": player_team}},
        "ball": ball,
    }


class PossessionTests(unittest.TestCase):
    def setUp(self):
        colors = {1: (0, 255, 0), 2: (0, 0, 255)}
        self.analytics = MatchAnalytics(colors, 25)

    def test_uses_player_feet_for_control(self):
        self.analytics.update_possession(frame(ball_xy=(100, 138)))

        self.assertEqual(self.analytics.possession_frames, {1: 1, 2: 0})

    def test_loose_ball_stops_counting_after_tolerance(self):
        self.analytics.update_possession(frame(ball_xy=(100, 138)))
        for _ in range(self.analytics._POSS_MISS_TOLERANCE + 5):
            self.analytics.update_possession(frame(ball_xy=(400, 400)))

        self.assertEqual(
            self.analytics.total_possession_frames,
            1 + self.analytics._POSS_MISS_TOLERANCE,
        )

    def test_opponent_must_control_for_hold_frames(self):
        self.analytics.update_possession(frame(ball_xy=(100, 138), player_team=1))
        opponent = frame(
            ball_xy=(300, 138),
            player_team=2,
            player_bbox=(280, 40, 320, 140),
        )

        for _ in range(self.analytics._POSS_HOLD_FRAMES - 1):
            self.analytics.update_possession(opponent)
        self.assertEqual(self.analytics._last_poss_team, 1)

        self.analytics.update_possession(opponent)
        self.assertEqual(self.analytics._last_poss_team, 2)


class PassCountTests(unittest.TestCase):
    def setUp(self):
        colors = {1: (0, 255, 0), 2: (0, 0, 255)}
        self.analytics = MatchAnalytics(colors, 25)

    def _multi_player_frame(self, ball_xy, player_dict):
        # player_dict: {pid: (team, bbox)}
        players = {}
        for pid, (team, bbox) in player_dict.items():
            players[pid] = {"team": team, "bbox": list(bbox)}
        bx, by = ball_xy
        ball = {1: {"bbox": [bx - 3, by - 3, bx + 3, by + 3]}}
        return {"players": players, "ball": ball}

    def test_pass_counted_only_between_same_team_different_players(self):
        p_setup = {
            1: (1, (80, 40, 120, 140)),     # feet at (100, 140)
            2: (1, (280, 40, 320, 140)),    # feet at (300, 140)
            3: (2, (480, 40, 520, 140)),    # feet at (500, 140)
        }

        # Player 1 (Team 1) controls ball for 3 frames (0, 1, 2)
        for f_idx in range(3):
            f = self._multi_player_frame((100, 138), p_setup)
            self.analytics.update_pass_count(f_idx, f)

        self.assertEqual(self.analytics.pass_counts, {1: 0, 2: 0})
        self.assertEqual(self.analytics._passer_id, 1)

        # Player 2 (Team 1, SAME TEAM) receives and controls ball for 3 frames (10, 11, 12)
        for f_idx in range(10, 13):
            f = self._multi_player_frame((300, 138), p_setup)
            self.analytics.update_pass_count(f_idx, f)

        self.assertEqual(self.analytics.pass_counts, {1: 1, 2: 0})

        # Player 3 (Team 2, OPPONENT) controls ball for 3 frames (20, 21, 22) -> Turnover, NO pass for Team 2
        for f_idx in range(20, 23):
            f = self._multi_player_frame((500, 138), p_setup)
            self.analytics.update_pass_count(f_idx, f)

        self.assertEqual(self.analytics.pass_counts, {1: 1, 2: 0})

    def test_side_by_side_players_do_not_trigger_pass(self):
        # Two players right next to each other (< 35px distance)
        p_setup = {
            1: (1, (80, 40, 120, 140)),     # feet at (100, 140)
            2: (1, (90, 40, 130, 140)),     # feet at (110, 140) - only 10px apart!
        }
        for f_idx in range(3):
            self.analytics.update_pass_count(f_idx, self._multi_player_frame((100, 138), p_setup))
        for f_idx in range(5, 8):
            self.analytics.update_pass_count(f_idx, self._multi_player_frame((110, 138), p_setup))

        self.assertEqual(self.analytics.pass_counts, {1: 0, 2: 0})


class MinimapLayoutTests(unittest.TestCase):
    def test_minimap_does_not_touch_reserved_bottom_area(self):
        video = np.zeros((720, 1280, 3), dtype=np.uint8)
        minimap = np.full((540, 960, 3), 255, dtype=np.uint8)

        result = blend(video, minimap, reserved_bottom=60)

        self.assertFalse(result[-60:].any())
        self.assertTrue(result[:-60].any())


if __name__ == "__main__":
    unittest.main()
