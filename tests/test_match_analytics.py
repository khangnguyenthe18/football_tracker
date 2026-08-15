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


class MinimapLayoutTests(unittest.TestCase):
    def test_minimap_does_not_touch_reserved_bottom_area(self):
        video = np.zeros((720, 1280, 3), dtype=np.uint8)
        minimap = np.full((540, 960, 3), 255, dtype=np.uint8)

        result = blend(video, minimap, reserved_bottom=60)

        self.assertFalse(result[-60:].any())
        self.assertTrue(result[:-60].any())


if __name__ == "__main__":
    unittest.main()
