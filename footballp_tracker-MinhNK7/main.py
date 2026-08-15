from trackers import Tracker
from team_assigner.team_assigner import TeamAssigner
from utils.video_utils import open_video, iter_frames, create_writer, write_frame, close_video, close_writer
from gsr_adapter import GameStateAdapter
from camera_movement_estimator import CameraMovementEstimator

from FieldMarkings.run import CamCalib, MODEL_PATH, LINES_FILE, blend

from BallAction.Test_Visual import BallActionSpot

def main():
    in_path  = 'data/inputs/Testvideo.mp4'
    out_path = 'data/outputs/TestVideo.avi'

    cap, fps, w, h = open_video(in_path)
    tracker = Tracker('data/models/detect/best_ylv8_ep50.pt', use_boost=True)
    team_assigner = TeamAssigner()
    gsr = GameStateAdapter(out_jsonl="data/outputs/game_state.jsonl")

    writer = create_writer(out_path, fps, w, h)
    team_fit_done = False

    cm_est = None
    frame_idx = 0
    cam_calib = CamCalib(MODEL_PATH, LINES_FILE)
    ball_action = BallActionSpot()
    prev_field = None

    for frame in iter_frames(cap):
        # Track 1 frame
        tracks_one = tracker.get_object_tracks([frame], read_from_stub=False, stub_path=None)
        cur_tracks = {k: v[0] for k, v in tracks_one.items()}

        # Fit team 1 lần khi có player
        if not team_fit_done and cur_tracks['players']:
            team_assigner.assign_team_color(frame, cur_tracks['players'])
            team_fit_done = getattr(team_assigner, 'kmeans', None) is not None

        # Gán team & màu vào tracks
        feets = []
        colors = []
        for pid, info in cur_tracks['players'].items():
            team = team_assigner.get_player_team(frame, info['bbox'], pid)
            info['team'] = team
            info['team_color'] = team_assigner.team_colors.get(team, (0, 0, 255))
            x1, y1, x2, y2 = info["bbox"]
            x1_n, y1_n, x2_n, y2_n = x1 / w, y1 / h, x2 / w, y2 / h
            feets.append(cam_calib.calibrate_player_feet((x1_n, y1_n, x2_n, y2_n)))
            colors.append(info['team_color'])

        # Draw
        drawn = tracker.draw_annotations_frame(frame, cur_tracks)
        
        # camera movement
        if cm_est is None:                            
            cm_est = CameraMovementEstimator(frame)  
            dx, dy = 0.0, 0.0                        
        else:
            dx, dy = cm_est.update(frame)
        
        drawn = cm_est.draw_overlay(drawn, dx, dy)  

        # Field Markings
        cam_calib(drawn)
        if feets and colors:
            prev_field = cam_calib.draw(drawn, colors, feets)
        if prev_field is not None:
            drawn = blend(drawn, prev_field, scale=0.5, alpha=0.4)
            
        # Action Spotting
        drawn = ball_action.visualize_frame(drawn, frame_idx)

        gsr.emit(frame_idx, cur_tracks)
        write_frame(writer, drawn)
        frame_idx += 1
        
        if frame_idx >= fps * 300: 
            break

    close_video(cap)
    close_writer(writer)
    print(f"[DONE] Wrote: {out_path}")


if __name__ == '__main__':
    main()
