from BallAction.src.ball_action import constants
from BallAction.src.utils import post_processing
from BallAction.scripts.ball_action.visualize import draw_graph
from collections import defaultdict
import numpy as np
import cv2

import os as _os
def load_video_predictions():
    raw_predictions_path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), 'pipeline_models', 'data', 'models', 'ball_action', 'test_raw_predictions.npz')
    raw_predictions_npz = np.load(str(raw_predictions_path))
    frame_indexes = raw_predictions_npz["frame_indexes"]
    raw_predictions = raw_predictions_npz["raw_predictions"]
    video_prediction = defaultdict(lambda: np.zeros(2, dtype=np.float32))

    for frame_index, prediction in zip(frame_indexes, raw_predictions):
        video_prediction[frame_index] = prediction

    video_pred_actions = defaultdict(lambda: np.zeros(2, dtype=np.float32))

    for cls, cls_index in constants.class2target.items():
        action_frame_indexes, _ = post_processing(
            frame_indexes, raw_predictions[:, cls_index], **constants.postprocess_params
        )
        for frame_index in action_frame_indexes:
            video_pred_actions[frame_index][cls_index] = 1.0
    return video_prediction, video_pred_actions

class BallActionSpot:
    def __init__(self):
        self.targets = {cls: [] for cls in constants.classes}
        self.predictions = {cls: [] for cls in constants.classes}
        self.pred_actions = {cls: [] for cls in constants.classes}
        self.video_prediction, self.video_pred_actions = load_video_predictions()

    def visualize_frame(self, frame, frame_idx, tracks_frame=None):
        prediction = self.video_prediction[frame_idx]
        pred_action = self.video_pred_actions[frame_idx]

        for cls, cls_index in constants.class2target.items():
            self.predictions[cls].append(prediction[cls_index])
            self.targets[cls].append(0)
            self.pred_actions[cls].append(pred_action[cls_index])

        h, w = frame.shape[:2]
        graph_margin = 10
        label_w = 55  # space for class label text

        for i, cls in enumerate(constants.classes):
            pass_graph = draw_graph(
                self.targets[cls], self.predictions[cls],
                self.pred_actions[cls], upscale=3
            )
            gh, gw = pass_graph.shape[:2]

            # Position: top-right corner, stacked vertically
            gx = w - gw - graph_margin - label_w
            gy = graph_margin + i * (gh + graph_margin + 5)

            # Bounds check
            if gx < 0 or gy < 0 or gx + gw > w or gy + gh > h:
                continue

            # Semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (gx - 2, gy - 2), (gx + gw + label_w + 2, gy + gh + 2),
                          (0, 0, 0), -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

            # Draw graph
            crop = frame[gy: gy + gh, gx: gx + gw]
            if crop.shape[0] == gh and crop.shape[1] == gw:
                cv2.addWeighted(pass_graph, 0.9, crop, 0.1, 0.0, crop)

            # Class label
            label_x = gx + gw + 4
            label_y = gy + gh // 2 + 5
            cv2.putText(frame, cls, (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

        # Frame counter — small text bottom-left
        cv2.putText(frame, f"F:{frame_idx}", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
        
        return frame