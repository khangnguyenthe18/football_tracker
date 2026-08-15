# Input videos

Put videos into folders input_videos

# Training

Follow file training/football_training_yolo_v5.ipynb, the ouput weight save at training/runs/detect/train6/weights

# Inference and detection testing

```
python yolo_inference.py
```

The output is file .avi at runs/detect

# input/ouput (Read n write vid)

utils/video_utils.py

# tracker

# Clustering players into 2 teams

team_assigner/team_assigner.py

```
# main.py
# Assign Player teams

team_assigner = TeamAssigner()
team_assigner.assign_team_color(video_frames[0],
                                tracks['players'][0])

for frame_num, player_track in enumerate(tracks['players']):
    for player_id, track in player_track.items():
        team = team_assigner.get_player_team(video_frames[frame_num],
                                                track['bbox'],
                                                player_id)
        tracks['players'][frame_num][player_id]['team'] = team
        tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
```

# Interpolate ball position (pending)

tracker.py

```
def interpolate_ball_positions(self, ball_positions):
    ...
```

ball_tracker/ball_tracker.py

# Tính bù trừ camera (camera estimator)

\camera_movement_estimator\camera_movement_estimator.py
