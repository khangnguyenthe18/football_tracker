from ultralytics import YOLO
import torch
import supervision as sv
import pickle
import os
import sys
import cv2
import numpy as np
from types import SimpleNamespace
from collections import deque 


try:
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
except NameError:
    pass


from utils import get_center_of_bbox, get_bbox_width, get_foot_position

# Import BoT-SORT
try:
    
    sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), "BoT-SORT"))
    from tracker.bot_sort import BoTSORT
except ImportError:
    
    print("Do not have BoT-SORT")
    class BoTSORT:
        def __init__(self, args, frame_rate):
            print("BoTSORT Mocked: Tracking")
        def update(self, det_input, frame):
            return []

def _remove_outliers(points, max_dist=150):
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
    def __init__(self, model_path, use_boost=False, trail_length=30):
        self.model = YOLO(model_path)
        self.use_boost = use_boost
        
        self.ball_trajectory = [] 
        self.smoothed_trajectory = None  
        self.ball_trajectory_pixel = deque(maxlen=trail_length) 
        self.trail_length = trail_length 

        self._miss_count = 0             
        self._last_vel = None            
        self._decay = 0.85               
        self._miss_tolerance = 8         

        if self.use_boost:
            args = SimpleNamespace(
                track_high_thresh=0.55, track_low_thresh=0.1, new_track_thresh=0.8,
                track_buffer=240, match_thresh=0.7, proximity_thresh=0.5,
                appearance_thresh=0.25, with_reid=False, device="cuda" if torch.cuda.is_available() else "cpu",
                cmc_method="orb", name="default", ablation=False, mot20=True
            )
            
            self.tracker = BoTSORT(args, frame_rate=30)
        else:
            self.tracker = sv.ByteTrack()


    def _clamp_to_frame(self, x, y, frame_shape):
        h, w = frame_shape[:2]  
        x_clamped = np.clip(x, 0, w - 1)  
        y_clamped = np.clip(y, 0, h - 1)  
        return x_clamped, y_clamped

    def _update_velocity(self, frame_num, cx, cy):
        if len(self.ball_trajectory) > 0:
            prev_frame, (prev_cx, prev_cy) = self.ball_trajectory[-1]
            if frame_num > prev_frame:  
                delta_x = cx - prev_cx
                delta_y = cy - prev_cy
                velocity = np.hypot(delta_x, delta_y)  
                self._last_vel = (delta_x, delta_y)  
        else:
            self._last_vel = (0, 0)  

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

    # Set up slicer (Not use)
    def _inference_callback(self, patch: np.ndarray) -> sv.Detections:
        result = self.model.infer(patch, confidence=0.3)[0]
        return sv.Detections.from_inference(result)

    def _setup_inference_slicer(self, frame):
        self.h, self.w, _ = frame.shape  
        self._slicer = sv.InferenceSlicer(
            callback=self._inference_callback,
            overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
            slice_wh=(self.w // 2 + 100, self.h // 2 + 100),  
            overlap_wh=(100, 100),  
            iou_threshold=0.1  
        )

    # Interpolate
    def interpolate_ball_trajectory(self, smooth=True):
        pts = self.ball_trajectory
        if len(pts) < 3:
            self.smoothed_trajectory = pts
            return

        frames = [f for f, _ in pts]
        xs = [p[0] for _, p in pts]
        ys = [p[1] for _, p in pts]
        full_frames = list(range(frames[0], frames[-1] + 1))
        x_interp = np.interp(full_frames, frames, xs)
        y_interp = np.interp(full_frames, frames, ys)

        if smooth:
            window = 5
            kernel = np.ones(window) / window
            x_interp = np.convolve(x_interp, kernel, mode='same')
            y_interp = np.convolve(y_interp, kernel, mode='same')

        self.smoothed_trajectory = list(zip(full_frames, zip(x_interp, y_interp)))

    # Track object
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None, demo_seconds=5, fps=30):
        
        if hasattr(self, "_cached_detections") and self._cached_detections is not None:
            detections = self._cached_detections
        else:
            detections = self.detect_frames(frames)
            self._cached_detections = detections
        

        tracks = {"players": [], "refs": [], "ball": []}
        max_frames = int(demo_seconds * fps)

       
        self.ball_trajectory = []
        self.ball_trajectory_pixel.clear() 
        self.smoothed_trajectory = None
        self._miss_count = 0
        self._last_vel = None

        for frame_num, (frame, detection) in enumerate(zip(frames, detections)):
            if frame_num >= max_frames:
                break

            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            det_sv = sv.Detections.from_ultralytics(detection)

            # goalkeeper = player
            for i_obj, class_id in enumerate(det_sv.class_id):
                if cls_names[class_id] == "goalkeeper" and "player" in cls_names_inv:
                    det_sv.class_id[i_obj] = cls_names_inv["player"]

            # 
            if self.use_boost:
                
                det_input = []
                for (xyxy, score, cls_id) in zip(det_sv.xyxy, det_sv.confidence, det_sv.class_id):
                    x1, y1, x2, y2 = xyxy.tolist()
                    det_input.append([x1, y1, x2, y2, float(score), int(cls_id)])
                det_input = np.array(det_input) if len(det_input) > 0 else np.empty((0, 6))
                tracks_active = self.tracker.update(det_input, frame)

                detection_with_tracks = []
                for track in tracks_active:
                    if not getattr(track, "is_activated", True): continue
                    tlwh = track.tlwh
                    x1, y1, w, h = tlwh
                    x2, y2 = x1 + w, y1 + h
                    bbox = [int(x1), int(y1), int(x2), int(y2)]
                    track_id = track.track_id or -1
                    cls_id = getattr(track, "cls", None) or cls_names_inv.get("player", 0)
                    detection_with_tracks.append((bbox, int(cls_id), int(track_id)))
            else:
                detection_with_tracks = []
                for det in self.tracker.update_with_detections(det_sv):
                    bbox = det[0]
                    cls_id = int(det[3])
                    track_id = int(det[4]) if det[4] else -1
                    detection_with_tracks.append((bbox, cls_id, track_id))

            
            tracks["players"].append({})
            tracks["refs"].append({})
            tracks["ball"].append({})

            player_cls_id = cls_names_inv.get("player")
            ball_cls_id = cls_names_inv.get("ball")

            for bbox, cls_id, track_id in detection_with_tracks:
                if cls_id == player_cls_id:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

            
            if ball_cls_id is not None:
                for bbox, cls_id in zip(det_sv.xyxy, det_sv.class_id):
                    if cls_id == ball_cls_id:
                        bb = bbox.tolist()
                        x1, y1, x2, y2 = map(float, bb)

                       
                        if (x2 - x1) > 0.05 * frame.shape[1] or (y2 - y1) > 0.1 * frame.shape[0]:
                            continue

                        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                        
                        
                        self.ball_trajectory.append((frame_num, (cx, cy)))
                        
                        
                        self.ball_trajectory_pixel.append(np.array([cx, cy]))

                        tracks["ball"][frame_num][1] = {
                            "bbox": bb,
                            "center": (cx, cy) 
                        }

                        self._miss_count = 0
                        self._update_velocity(frame_num, cx, cy)
                        break

        
        self.ball_trajectory = _remove_outliers(self.ball_trajectory, max_dist=140)
        self.interpolate_ball_trajectory()

        # Fill interpolated ball positions back into tracks for missed frames
        if self.smoothed_trajectory:
            for f_idx, (cx, cy) in self.smoothed_trajectory:
                if f_idx < len(tracks["ball"]) and not tracks["ball"][f_idx]:
                    # Create synthetic ball entry with small bbox
                    r = 8  # approximate ball radius in pixels
                    tracks["ball"][f_idx][1] = {
                        "bbox": [cx - r, cy - r, cx + r, cy + r],
                        "center": (cx, cy),
                        "interpolated": True
                    }

        return tracks


    def draw_annotations_frame(self, frame, tracks_frame, frame_num):
        
        out = frame.copy()

        # Ball trail disabled
        # out = self._draw_ball_trail(out)

        # Draw players 
        for track_id, player in (tracks_frame.get("players") or {}).items():
            color = player.get("team_color", (0, 0, 255))
            # Convert numpy color to tuple of ints
            if hasattr(color, '__iter__') and not isinstance(color, tuple):
                color = tuple(int(c) for c in color)
            out = self.draw_ellipse(out, player["bbox"], color, track_id)

        # Draw ball 
        for _, ball in (tracks_frame.get("ball") or {}).items():
            out = self.draw_ball_marker(out, ball["bbox"])

        return out

   

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        rx = int(0.8 * width)
        ry = int(0.3 * width)
        # Black outline for contrast
        cv2.ellipse(frame, (x_center, y2), (rx, ry), 0, -45, 235, (0, 0, 0), 4, cv2.LINE_AA)
        # Colored fill
        cv2.ellipse(frame, (x_center, y2), (rx, ry), 0, -45, 235, color, 2, cv2.LINE_AA)

        # Draw track ID
        if track_id is not None:
            label = str(track_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            tx = x_center - tw // 2
            ty = y2 + ry + th + 4
            # Background pill
            cv2.rectangle(frame, (tx - 3, ty - th - 3), (tx + tw + 3, ty + 3), (0, 0, 0), -1, cv2.LINE_AA)
            cv2.putText(frame, label, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return frame

    def draw_ball_marker(self, frame, bbox):
        """Draw a circular ball marker with glow effect."""
        x1, y1, x2, y2 = map(int, bbox)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        r = max(6, int(0.5 * max(x2 - x1, y2 - y1)))
        # Outer glow
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), r + 6, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        # Main circle
        cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2, cv2.LINE_AA)
        # Center dot
        cv2.circle(frame, (cx, cy), 2, (255, 255, 255), -1, cv2.LINE_AA)
        return frame

    
    def _draw_ball_trail(self, frame, base_color=(0, 255, 255), max_thickness=3):
        """Draw ball trail with gradient fade and distance check."""
        traj_points = np.array(self.ball_trajectory_pixel).astype(np.int32)

        if len(traj_points) < 2:
            return frame

        n = len(traj_points)
        max_jump = 150  # skip segments longer than this (outlier jumps)

        for i in range(1, n):
            pt1 = tuple(traj_points[i - 1])
            pt2 = tuple(traj_points[i])

            # Skip outlier jumps
            dist = np.linalg.norm(traj_points[i].astype(float) - traj_points[i - 1].astype(float))
            if dist > max_jump:
                continue

            # Gradient: older segments are dimmer
            alpha = (i / n)  # 0.0 (oldest) → 1.0 (newest)
            color = tuple(int(c * (0.25 + 0.75 * alpha)) for c in base_color)
            thickness = max(1, int(max_thickness * alpha))

            cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

        return frame