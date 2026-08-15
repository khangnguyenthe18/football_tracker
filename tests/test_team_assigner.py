import unittest

import cv2
import numpy as np

from team_assigner.team_assigner import TeamAssigner


class TeamAssignerTests(unittest.TestCase):
    @staticmethod
    def _synthetic_frame():
        frame = np.full((200, 220, 3), (45, 125, 45), dtype=np.uint8)
        boxes = {
            1: {"bbox": [15, 10, 95, 190]},
            2: {"bbox": [125, 10, 205, 190]},
        }

        # Keep grass at bbox edges and paint the upper-body center only.
        cv2.rectangle(frame, (35, 30), (75, 105), (145, 250, 180), -1)
        cv2.rectangle(frame, (145, 30), (185, 105), (240, 240, 240), -1)
        return frame, boxes

    def test_extracts_kit_instead_of_pitch_color(self):
        frame, boxes = self._synthetic_frame()
        assigner = TeamAssigner()

        green_kit = assigner.get_player_color(frame, boxes[1]["bbox"])
        white_kit = assigner.get_player_color(frame, boxes[2]["bbox"])

        self.assertGreater(green_kit[1], green_kit[0] + 50)
        self.assertLess(np.ptp(white_kit), 20)

    def test_assigns_two_kits_to_different_teams(self):
        frame, boxes = self._synthetic_frame()
        assigner = TeamAssigner()
        assigner.assign_team_color(frame, boxes)

        team_1 = assigner.get_player_team(frame, boxes[1]["bbox"], 1)
        team_2 = assigner.get_player_team(frame, boxes[2]["bbox"], 2)

        self.assertNotEqual(team_1, team_2)

    def test_display_color_keeps_white_kit_neutral(self):
        boosted = TeamAssigner._boost_color((240, 240, 240))
        hsv = cv2.cvtColor(np.uint8([[boosted]]), cv2.COLOR_BGR2HSV)[0, 0]

        self.assertLess(hsv[1], 40)
        self.assertGreaterEqual(hsv[2], 210)

    def test_waits_when_only_one_kit_color_is_visible(self):
        frame, boxes = self._synthetic_frame()
        cv2.rectangle(frame, (145, 30), (185, 105), (145, 250, 180), -1)
        assigner = TeamAssigner()

        assigner.assign_team_color(frame, boxes)

        self.assertIsNone(assigner.kmeans)
        self.assertEqual(assigner.team_colors, {})


if __name__ == "__main__":
    unittest.main()
