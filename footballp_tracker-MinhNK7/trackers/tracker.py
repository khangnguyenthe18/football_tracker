from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

# Import BoT-SORT
sys.path.append("BoT-SORT")
from tracker.bot_sort import BoTSORT


def _remove_outliers(points, max_dist=200):
    """
    Loại bỏ các điểm (cx, cy) có khoảng cách > max_dist với điểm trước đó.
    """
    if len(points) < 3:
        return points
    clean = [points[0]]
    for i in range(1, len(points)):
        prev = np.array(clean[-1][1], dtype=float)
        curr = np.array(points[i][1], dtype=float)
        dist = np.linalg.norm(curr - prev)
        if dist < max_dist:
            clean.append(points[i])
    return clean


class Tracker:
    def __init__(self, model_path, use_boost=False):
        print("[DEBUG] Tracker init started")

        self.model = YOLO(model_path)
        self.use_boost = use_boost

        # Quỹ đạo bóng
        self.ball_trajectory = []
        self.smoothed_trajectory = None

        if self.use_boost:
            # ⚙️ Tạo args có đủ 2 thuộc tính BoT-SORT yêu cầu
            args = SimpleNamespace(
                track_high_thresh=0.55,
                track_low_thresh=0.1,
                new_track_thresh=0.8,
                track_buffer=240,
                match_thresh=0.7,
                proximity_thresh=0.5,
                appearance_thresh=0.25,
                with_reid=False,
                fast_reid_config=None,
                fast_reid_weights=None,
                device="cuda",
                cmc_method="orb",
                mot20=True,
                # 🔧 Bắt buộc thêm hai dòng này
                name="default",
                ablation=False
            )

            print("[DEBUG] Args content:", vars(args))
            print("[DEBUG] BoTSORT about to initialize...")

            self.tracker = BoTSORT(args, frame_rate=30)

            print("[DEBUG] BoTSORT initialized successfully.")
        else:
            print("[DEBUG] Using ByteTrack instead of BoTSORT.")
            self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(
                frames[i:i + batch_size],
                imgsz=960,
                conf=0.3,
                iou=0.45
            )
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        self.ball_trajectory = []
        self.smoothed_trajectory = None
        detections = self.detect_frames(frames)
        tracks = {"players": [], "refs": [], "ball": []}

        for frame_num, (frame, detection) in enumerate(zip(frames, detections)):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            det_sv = sv.Detections.from_ultralytics(detection)

            # Chuyển goalkeeper thành player
            if "player" in cls_names_inv:
                for i, class_id in enumerate(det_sv.class_id):
                    if cls_names[class_id] == "goalkeeper":
                        det_sv.class_id[i] = cls_names_inv["player"]

            # Tracker
            if self.use_boost:
                det_input = []
                for (xyxy, score, cls_id) in zip(det_sv.xyxy, det_sv.confidence, det_sv.class_id):
                    x1, y1, x2, y2 = xyxy.tolist()
                    det_input.append([x1, y1, x2, y2, float(score), int(cls_id)])
                det_input = np.array(det_input) if len(det_input) > 0 else np.empty((0, 6))
                tracks_active = self.tracker.update(det_input, frame)
                detection_with_tracks = []
                for track in tracks_active:
                    if not getattr(track, "is_activated", True):
                        continue
                    tlwh = track.tlwh
                    x1, y1, w, h = tlwh
                    x2, y2 = x1 + w, y1 + h
                    bbox = [int(x1), int(y1), int(x2), int(y2)]
                    track_id = track.track_id if track.track_id is not None else -1
                    cls_id = getattr(track, "cls", None) or cls_names_inv.get("player", -1)
                    detection_with_tracks.append((bbox, int(cls_id), int(track_id)))
            else:
                detection_with_tracks = []
                for det in self.tracker.update_with_detections(det_sv):
                    bbox = det[0]
                    cls_id = int(det[3])
                    track_id = int(det[4]) if det[4] is not None else -1
                    detection_with_tracks.append((bbox, cls_id, track_id))

            tracks["players"].append({})
            tracks["refs"].append({})
            tracks["ball"].append({})

            player_id = cls_names_inv.get('player', None)
            ref_id = cls_names_inv.get('ref', None)
            ball_id = cls_names_inv.get('ball', None)

            for bbox, cls_id, track_id in detection_with_tracks:
                if cls_id == player_id:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                elif cls_id == ref_id:
                    tracks["refs"][frame_num][track_id] = {"bbox": bbox}

            # Bóng
            if ball_id is not None:
                for bbox, cls_id in zip(det_sv.xyxy, det_sv.class_id):
                    if int(cls_id) == ball_id:
                        bb = bbox.tolist()
                        tracks["ball"][frame_num][1] = {"bbox": bb}
                        x1, y1, x2, y2 = map(float, bb)
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        self.ball_trajectory.append((frame_num, (cx, cy)))
                        break

        self.interpolate_ball_trajectory()

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    # === Nội suy & làm mượt quỹ đạo bóng ===
    def interpolate_ball_trajectory(self, max_dist=200, smooth=True):
        pts = self.ball_trajectory
        if len(pts) < 3:
            self.smoothed_trajectory = pts
            return

        clean = _remove_outliers(pts, max_dist=max_dist)
        if len(clean) < 3:
            self.smoothed_trajectory = clean
            return

        frames = [f for f, _ in clean]
        xs = [p[0] for _, p in clean]
        ys = [p[1] for _, p in clean]
        full_frames = list(range(frames[0], frames[-1] + 1))
        x_interp = np.interp(full_frames, frames, xs)
        y_interp = np.interp(full_frames, frames, ys)

        if smooth:
            kernel = np.ones(5) / 5
            x_interp = np.convolve(x_interp, kernel, mode='same')
            y_interp = np.convolve(y_interp, kernel, mode='same')

        self.smoothed_trajectory = list(zip(full_frames, zip(x_interp, y_interp)))

    def _draw_trajectory(self, frame, max_jump_px=150, max_gap_frames=5, color=(0, 255, 0), thickness=2):
        traj = self.smoothed_trajectory or self.ball_trajectory
        if len(traj) < 2:
            return frame

        for i in range(1, len(traj)):
            f1, (x1, y1) = traj[i - 1]
            f2, (x2, y2) = traj[i]
            if (f2 - f1) <= max_gap_frames and np.hypot(x2 - x1, y2 - y1) < max_jump_px:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        return frame

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(0.8 * width), int(0.3 * width)),
            angle=0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2
        )
        if track_id is not None:
            cv2.putText(frame, str(track_id), (x_center, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def draw_bbox_with_id(self, frame, bbox, color, track_id=None):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if track_id:
            cv2.putText(frame, str(track_id), (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    def draw_annotations_frame(self, frame, tracks_frame):
        out = frame.copy()
        for track_id, player in (tracks_frame.get("players") or {}).items():
            out = self.draw_ellipse(out, player["bbox"], (0, 0, 255), track_id)
        for _, ref in (tracks_frame.get("refs") or {}).items():
            out = self.draw_ellipse(out, ref["bbox"], (0, 255, 255))
        for _, ball in (tracks_frame.get("ball") or {}).items():
            out = self.draw_bbox_with_id(out, ball["bbox"], (0, 255, 0))
        out = self._draw_trajectory(out)
        return out
