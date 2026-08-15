# team_assigner/team_assigner.py
from sklearn.cluster import KMeans
from collections import defaultdict, deque
import numpy as np
import cv2

class TeamAssigner:
    REEVAL_INTERVAL = 30   # re-check team color every N frames
    VOTE_HISTORY = 5       # keep last N observations for majority vote
    TORSO_TOP = 0.10
    TORSO_BOTTOM = 0.58
    TORSO_LEFT = 0.15
    TORSO_RIGHT = 0.85
    BACKGROUND_EDGE = 0.12
    FOREGROUND_PERCENTILE = 60
    MIN_TEAM_COLOR_DISTANCE = 12.0

    def __init__(self):
        self.team_colors = {}          
        self.player_team_dict = {}    
        self.kmeans = None
        self._frame_count = defaultdict(int)    # pid → frames since last eval
        self._team_votes = defaultdict(lambda: deque(maxlen=TeamAssigner.VOTE_HISTORY))

    @staticmethod
    def _boost_color(bgr):
        """Boost saturation & brightness while preserving hue."""
        pixel = np.uint8([[list(bgr)]])
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
        saturation = int(hsv[1])
        # Preserve neutral kits such as white/grey. Forcing their saturation
        # creates a false hue and made both team indicators look green.
        if saturation >= 40:
            hsv[1] = min(255, max(140, int(saturation * 1.2)))
        hsv[2] = min(255, max(int(hsv[2]), 210))
        boosted = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0][0]
        return tuple(int(c) for c in boosted)

    @staticmethod
    def _to_lab(bgr):
        pixel = np.asarray(bgr, dtype=np.uint8).reshape(1, 1, 3)
        return cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0, 0].astype(float)

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

        y_top = max(0, int(h * self.TORSO_TOP))
        y_bot = min(h, max(y_top + 1, int(h * self.TORSO_BOTTOM)))
        x_l = max(0, int(w * self.TORSO_LEFT))
        x_r = min(w, max(x_l + 1, int(w * self.TORSO_RIGHT)))
        torso = image[y_top:y_bot, x_l:x_r]
        if torso.size == 0:
            return None

        # Estimate the pitch/background from thin vertical strips along the
        # bbox. Select torso pixels most different from that background. This
        # works for bright, dark and neutral kits without assuming grass hue.
        edge_w = max(1, int(w * self.BACKGROUND_EDGE))
        edge_pixels = np.concatenate(
            (image[:, :edge_w].reshape(-1, 3),
             image[:, -edge_w:].reshape(-1, 3)),
            axis=0,
        )
        if edge_pixels.size == 0:
            return None

        edge_lab = cv2.cvtColor(
            edge_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB
        ).reshape(-1, 3).astype(float)
        background_lab = np.median(edge_lab, axis=0)

        torso_pixels = torso.reshape(-1, 3)
        torso_lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
        distances = np.linalg.norm(torso_lab - background_lab, axis=1)
        cutoff = np.percentile(distances, self.FOREGROUND_PERCENTILE)
        foreground = torso_pixels[distances >= cutoff]
        if len(foreground) < 5:
            foreground = torso_pixels

        return np.median(foreground, axis=0)

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

        colors_bgr = np.asarray(player_colors, dtype=np.uint8)
        X = np.asarray([self._to_lab(color) for color in colors_bgr])
        pairwise_distances = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        if pairwise_distances.max() < self.MIN_TEAM_COLOR_DISTANCE:
            self.kmeans = None
            self.team_colors.clear()
            return

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        kmeans.fit(X)

        labels_present = set(int(label) for label in kmeans.labels_)
        center_distance = np.linalg.norm(
            kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1]
        )
        if labels_present != {0, 1} or center_distance < self.MIN_TEAM_COLOR_DISTANCE:
            self.kmeans = None
            self.team_colors.clear()
            return

        self.kmeans = kmeans
        for label in (0, 1):
            cluster_colors = colors_bgr[kmeans.labels_ == label]
            representative = np.median(cluster_colors, axis=0)
            self.team_colors[label + 1] = self._boost_color(representative)

    def get_player_team(self, frame, player_bbox, player_id):
        if self.kmeans is None:
            return None

        # Check if we need to re-evaluate (periodic or first time)
        self._frame_count[player_id] += 1
        needs_eval = (player_id not in self.player_team_dict or
                      self._frame_count[player_id] >= self.REEVAL_INTERVAL)

        if needs_eval:
            self._frame_count[player_id] = 0
            color = self.get_player_color(frame, player_bbox)
            if color is not None:
                team_id = int(self.kmeans.predict(
                    self._to_lab(color).reshape(1, -1))[0]) + 1
                self._team_votes[player_id].append(team_id)

                # Majority vote from recent observations
                votes = list(self._team_votes[player_id])
                counts = {vote: votes.count(vote) for vote in set(votes)}
                max_count = max(counts.values())
                majority = next(
                    vote for vote in reversed(votes)
                    if counts[vote] == max_count
                )
                self.player_team_dict[player_id] = majority

        return self.player_team_dict.get(player_id)
