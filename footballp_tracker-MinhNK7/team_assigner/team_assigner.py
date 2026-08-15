# team_assigner/team_assigner.py
from sklearn.cluster import KMeans
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}          
        self.player_team_dict = {}    
        self.kmeans = None

    # helper: clamp bbox vào trong frame (tránh crash)
    def _clip_bbox(self, frame, bbox):
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = map(float, bbox)
        x1 = max(0, min(W - 1, x1))
        x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1))
        y2 = max(0, min(H - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return int(x1), int(y1), int(x2), int(y2)

    # add guard
    def get_clustering_model(self, image):
        if image is None or image.ndim != 3 or image.shape[0] < 1 or image.shape[1] < 1:
            return None
        X = image.reshape(-1, 3)
        if X.size == 0:
            return None
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init="auto", random_state=42)
        kmeans.fit(X)
        return kmeans

    def get_player_color(self, frame, bbox):
        clipped = self._clip_bbox(frame, bbox)
        if clipped is None:
            return None

        x1, y1, x2, y2 = clipped
        image = frame[y1:y2, x1:x2]
        if image.size == 0:
            return None

        h, w = image.shape[:2]
        if h < 2 or w < 2:
            return None

        y_top = max(0, int(h * 0.2))
        y_bot = min(h, max(y_top + 1, int(h * 0.7)))
        x_l   = max(0, int(w * 0.2))
        x_r   = min(w, max(x_l + 1, int(w * 0.8)))
        mid_image = image[y_top:y_bot, x_l:x_r]
        if mid_image.size == 0:
            return None

        kmeans = self.get_clustering_model(mid_image)
        if kmeans is None:
            return None

        labels = kmeans.labels_.reshape(mid_image.shape[0], mid_image.shape[1])

        corners = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
        non_player_cluster = max(set(corners), key=corners.count)
        player_cluster = 1 - non_player_cluster

        return kmeans.cluster_centers_[player_cluster]

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, det in player_detections.items():
            bbox = det.get("bbox")
            if bbox is None:
                continue
            color = self.get_player_color(frame, bbox)
            if color is not None:
                player_colors.append(color)

        if len(player_colors) < 2:
            self.kmeans = None
            self.team_colors.clear()
            return

        X = np.asarray(player_colors, dtype=float)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init="auto", random_state=42)
        kmeans.fit(X)

        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        if self.kmeans is None:
            return None

        color = self.get_player_color(frame, player_bbox)
        if color is None:
            return None

        team_id = int(self.kmeans.predict(np.asarray(color, dtype=float).reshape(1, -1))[0]) + 1
        self.player_team_dict[player_id] = team_id
        return team_id
