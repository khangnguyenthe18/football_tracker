from trackers import Tracker
from team_assigner.team_assigner import TeamAssigner
from utils.video_utils import open_video, iter_frames, create_writer, write_frame, close_video, close_writer
from gsr_adapter import GameStateAdapter
from camera_movement_estimator import CameraMovementEstimator
from match_analytics import MatchAnalytics

from FieldMarkings.run import CamCalib, MODEL_PATH, LINES_FILE, blend
from BallAction.Test_Visual import BallActionSpot
from tqdm import tqdm
import sys
sys.stdout.reconfigure(line_buffering=True)


def main():
    in_path  = '08fd33_4.mp4'
    out_path = 'output_full.avi'

    cap, fps, w, h = open_video(in_path)

    tracker = Tracker('best.pt', use_boost=True)
    team_assigner = TeamAssigner()
    gsr = GameStateAdapter(out_jsonl="game_state.jsonl")

    writer = create_writer(out_path, fps, w, h)
    team_fit_done = False

    cm_est = None
    frame_idx = 0
    cam_calib = CamCalib(MODEL_PATH, LINES_FILE)
    ball_action = BallActionSpot()
    prev_field = None
    analytics = None  # initialized after team colors are assigned

    # ======== ĐỌC TOÀN BỘ FRAMES ========
    frames = []
    print("[INFO] Loading video frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    total_frames = len(frames)
    print(f"[INFO] Loaded {total_frames} frames (~{total_frames/fps:.1f}s @ {fps:.0f}fps).")

    # ======== TRACK TOÀN BỘ ========
    print("[INFO] Running tracker for all frames...")
    tracks_all = tracker.get_object_tracks(frames, read_from_stub=False, stub_path=None,
                                           demo_seconds=total_frames/fps + 1, fps=fps)

    # ======== XỬ LÝ TỪNG FRAME ========
    print("[INFO] Drawing annotations & exporting...")
    for frame_num in tqdm(range(total_frames), desc="Processing", unit="frame"):
        frame = frames[frame_num]
        cur_tracks = {k: v[frame_num] if frame_num < len(v) else {} for k, v in tracks_all.items()}

        # Fit team 1 lần
        if not team_fit_done and cur_tracks['players']:
            team_assigner.assign_team_color(frame, cur_tracks['players'])
            team_fit_done = getattr(team_assigner, 'kmeans', None) is not None

        # Initialize analytics after team colors are known
        if analytics is None and team_fit_done:
            analytics = MatchAnalytics(team_assigner.team_colors, fps,
                                       minimap_w=cam_calib.IMG_W,
                                       minimap_h=cam_calib.IMG_H)

        # Update homography before projecting player positions for this frame.
        try:
            cam_calib(frame)
        except Exception:
            pass  # retain the last valid calibration when HRNet misses

        # Gán team & màu + calibrate feet
        feets, colors = [], []
        for pid, info in cur_tracks['players'].items():
            x1, y1, x2, y2 = info["bbox"]
            x1_n, y1_n, x2_n, y2_n = x1 / w, y1 / h, x2 / w, y2 / h
            world_pos = cam_calib.get_world_feet((x1_n, y1_n, x2_n, y2_n))
            team = team_assigner.get_player_team(frame, info['bbox'], pid, world_pos=world_pos)
            info['team'] = team
            info['team_color'] = team_assigner.team_colors.get(team, (0, 0, 255))
            try:
                feet = cam_calib.calibrate_player_feet((x1_n, y1_n, x2_n, y2_n), player_id=pid)
                if feet is not None:
                    feets.append(feet)
                    colors.append(info['team_color'])
            except Exception:
                pass  # skip if calibration fails for this player

        # ======== CAMERA MOVEMENT (compute before analytics so dx/dy is available) ========
        if cm_est is None:
            cm_est = CameraMovementEstimator(frame)
            dx, dy = 0.0, 0.0
        else:
            dx, dy = cm_est.update(frame)

        # ======== ANALYTICS UPDATE ========
        if analytics is not None:
            analytics.update_possession(cur_tracks)
            analytics.update_pass_count(frame_idx, cur_tracks, ball_action)
            analytics.update_speed(cur_tracks, cam_calib, w, h,
                                   cam_dx=dx, cam_dy=dy)
            analytics.update_heatmap(cur_tracks, cam_calib, w, h)
            analytics.tick()

        # ======== DRAW ========
        drawn = tracker.draw_annotations_frame(frame, cur_tracks, frame_num)
        drawn = cm_est.draw_overlay(drawn, dx, dy)

        # Field markings (with error guard)
        try:
            heatmap_ov = analytics.get_heatmap_overlay(cam_calib.IMG_H, cam_calib.IMG_W) if analytics else None
            if cam_calib.H is not None:
                prev_field = cam_calib.draw(drawn, colors, feets, heatmap_overlay=heatmap_ov)
        except Exception:
            pass  # HRNet can fail on some frames

        if prev_field is not None:
            drawn = blend(drawn, prev_field)

        # Ball action
        drawn = ball_action.visualize_frame(drawn, frame_idx, cur_tracks)

        # ======== ANALYTICS HUD ========
        if analytics is not None:
            drawn = analytics.draw_possession_bar(drawn)
            drawn = analytics.draw_pass_count(drawn)
            drawn = analytics.draw_speed_labels(drawn, cur_tracks)

        gsr.emit(frame_idx, cur_tracks)
        write_frame(writer, drawn)
        frame_idx += 1

    # ======== MATCH SUMMARY FRAME (3 seconds) ========
    if analytics is not None:
        print("[INFO] Writing match summary frame...")
        summary = analytics.draw_summary_frame(w, h)
        for _ in range(int(fps * 3)):
            write_frame(writer, summary)

    close_video(cap)
    close_writer(writer)
    print(f"[DONE] Full video wrote: {out_path} ({frame_idx} frames + summary)")


if __name__ == '__main__':
    main()
