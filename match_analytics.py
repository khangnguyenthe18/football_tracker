# match_analytics.py
"""
Match Analysis engine — possession, pass count, speed, heatmap, HUD overlays.
All drawing uses a modern glassmorphism / broadcast-TV aesthetic.
"""
import cv2
import numpy as np
from collections import defaultdict, deque
from utils.bbox_utils import get_center_of_bbox, get_foot_position


# ─── Design tokens ──────────────────────────────────────────────────
_BG_ALPHA = 0.65            # glass panel opacity
_ACCENT   = (0, 230, 255)   # cyan accent  (BGR)
_WHITE    = (245, 245, 245)
_DIM      = (160, 160, 160)
_DARK_BG  = (20, 20, 25)    # near-black for panels
_FONT     = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SM  = cv2.FONT_HERSHEY_SIMPLEX

# Real pitch dimensions (FIFA standard, metres)
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M  = 68.0


def _rounded_rect(img, pt1, pt2, color, thickness, radius=8, alpha=None):
    """Draw a rounded rectangle. If alpha is set, blends with background."""
    x1, y1 = pt1
    x2, y2 = pt2
    overlay = img.copy() if alpha else img

    # Clamp radius
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if r < 1:
        cv2.rectangle(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)
    else:
        # Fill body
        if thickness == -1:
            cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
            cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1, cv2.LINE_AA)
            cv2.circle(overlay, (x1 + r, y1 + r), r, color, -1, cv2.LINE_AA)
            cv2.circle(overlay, (x2 - r, y1 + r), r, color, -1, cv2.LINE_AA)
            cv2.circle(overlay, (x1 + r, y2 - r), r, color, -1, cv2.LINE_AA)
            cv2.circle(overlay, (x2 - r, y2 - r), r, color, -1, cv2.LINE_AA)
        else:
            # outline only
            cv2.line(overlay, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
            cv2.line(overlay, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
            cv2.line(overlay, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
            cv2.line(overlay, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
            cv2.ellipse(overlay, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(overlay, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(overlay, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(overlay, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)

    if alpha is not None:
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img


def _glass_panel(img, x1, y1, x2, y2, alpha=_BG_ALPHA, border_color=(60, 60, 65)):
    """Draw a glassmorphism-style panel."""
    _rounded_rect(img, (x1, y1), (x2, y2), _DARK_BG, -1, radius=10, alpha=alpha)
    _rounded_rect(img, (x1, y1), (x2, y2), border_color, 1, radius=10)
    return img


def _put_text(img, text, org, scale=0.45, color=_WHITE, thickness=1):
    cv2.putText(img, text, org, _FONT, scale, color, thickness, cv2.LINE_AA)


def _gradient_bar(img, x, y, w, h, pct, color_left, color_right):
    """Draw a horizontal gradient progress bar."""
    split = max(1, int(w * pct))
    # Left portion (team 1)
    for i in range(split):
        alpha_fill = 0.6 + 0.4 * (i / max(1, w))
        c = tuple(int(v * alpha_fill) for v in color_left)
        cv2.line(img, (x + i, y), (x + i, y + h), c, 1, cv2.LINE_AA)
    # Right portion (team 2)
    for i in range(split, w):
        alpha_fill = 0.6 + 0.4 * ((w - i) / max(1, w))
        c = tuple(int(v * alpha_fill) for v in color_right)
        cv2.line(img, (x + i, y), (x + i, y + h), c, 1, cv2.LINE_AA)
    # Separator line
    cv2.line(img, (x + split, y - 1), (x + split, y + h + 1), _WHITE, 1, cv2.LINE_AA)
    # Border
    _rounded_rect(img, (x - 1, y - 1), (x + w + 1, y + h + 1), (80, 80, 85), 1, radius=4)


# ═══════════════════════════════════════════════════════════════════
class MatchAnalytics:
    """Central match analysis engine."""

    # ── Possession tuning ──
    _POSS_HOLD_FRAMES = 5         # frames a new team must lead before switching
    _POSS_MISS_TOLERANCE = 15     # keep last team for N frames when ball is lost
    _POSS_MIN_RADIUS = 30.0
    _POSS_MAX_RADIUS = 90.0
    _POSS_HEIGHT_FACTOR = 0.75

    def __init__(self, team_colors: dict, fps: float,
                 minimap_w: int = 960, minimap_h: int = 540):
        self.team_colors = team_colors  # {1: (b,g,r), 2: (b,g,r)}
        self.fps = fps

        # Minimap → real-world scale (direct ratio, no external calib_f needed)
        self.minimap_w = minimap_w
        self.minimap_h = minimap_h
        self.px_per_m_x = minimap_w / PITCH_LENGTH_M
        self.px_per_m_y = minimap_h / PITCH_WIDTH_M

        # ── Possession ──
        self.possession_frames = {1: 0, 2: 0}
        self.total_possession_frames = 0
        self._last_poss_team = None         # current possessing team
        self._poss_candidate = None         # candidate team for switch
        self._poss_candidate_streak = 0     # how many frames candidate leads
        self._poss_miss_count = 0           # frames since ball last seen

        # ── Pass count ──
        self.pass_counts = {1: 0, 2: 0}
        self._prev_pass_active = False  # rising-edge detection

        # ── Speed ──
        self._prev_feet = {}       # pid → (cal_x, cal_y)  calibrated
        self._speed_buf = defaultdict(lambda: deque(maxlen=8))  # pid → last N speeds
        self._cur_speeds = {}      # pid → smoothed km/h
        self._top_speed = {1: 0.0, 2: 0.0}
        self._total_dist = defaultdict(float)  # pid → metres

        # ── Heatmap ──
        self._heat_t1 = np.zeros((minimap_h, minimap_w), dtype=np.float32)
        self._heat_t2 = np.zeros((minimap_h, minimap_w), dtype=np.float32)

        # ── Accumulated stats for summary ──
        self._frame_count = 0
        self._player_teams = {}    # pid → team

    # ─── helpers ─────────────────────────────────────────────────
    @staticmethod
    def _ball_center(tracks_frame):
        ball = tracks_frame.get("ball") or {}
        for info in ball.values():
            if "bbox" not in info:
                continue
            bb = info["bbox"]
            return (float(bb[0] + bb[2]) / 2, float(bb[1] + bb[3]) / 2)
        return None

    def _nearest_player(self, tracks_frame, point, use_feet=True):
        """Return (player_id, team, distance) of the player closest to `point`."""
        best_pid, best_team, best_dist = None, None, float("inf")
        for pid, info in (tracks_frame.get("players") or {}).items():
            if "bbox" not in info:
                continue
            if use_feet:
                cx, cy = get_foot_position(info["bbox"])
            else:
                cx, cy = get_center_of_bbox(info["bbox"])
            d = ((cx - point[0]) ** 2 + (cy - point[1]) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_pid = pid
                best_team = info.get("team")
        return best_pid, best_team, best_dist

    def _control_radius(self, player_info):
        bbox = player_info["bbox"]
        height = max(0.0, float(bbox[3]) - float(bbox[1]))
        return min(
            self._POSS_MAX_RADIUS,
            max(self._POSS_MIN_RADIUS, height * self._POSS_HEIGHT_FACTOR),
        )

    def _record_uncontrolled_frame(self):
        """Keep brief dropouts without awarding long loose-ball periods."""
        self._poss_miss_count += 1
        self._poss_candidate = None
        self._poss_candidate_streak = 0
        if (
            self._last_poss_team in (1, 2)
            and self._poss_miss_count <= self._POSS_MISS_TOLERANCE
        ):
            self.possession_frames[self._last_poss_team] += 1
            self.total_possession_frames += 1

    # ═══════ 1. POSSESSION (with distance threshold + hysteresis) ═══
    def update_possession(self, tracks_frame):
        bc = self._ball_center(tracks_frame)

        if bc is None:
            self._record_uncontrolled_frame()
            return

        pid, team, dist = self._nearest_player(tracks_frame, bc, use_feet=True)
        players = tracks_frame.get("players") or {}
        player_info = players.get(pid)

        # A visible ball may still be loose or in flight.
        if (
            team not in (1, 2)
            or player_info is None
            or dist > self._control_radius(player_info)
        ):
            self._record_uncontrolled_frame()
            return

        self._poss_miss_count = 0

        # Assign the first controlled frame immediately.
        if self._last_poss_team is None:
            self._last_poss_team = team
            self._poss_candidate = None
            self._poss_candidate_streak = 0

        # Hysteresis: require N consecutive frames before switching
        if team == self._last_poss_team:
            self._poss_candidate = None
            self._poss_candidate_streak = 0
        else:
            if team == self._poss_candidate:
                self._poss_candidate_streak += 1
            else:
                self._poss_candidate = team
                self._poss_candidate_streak = 1

            if self._poss_candidate_streak >= self._POSS_HOLD_FRAMES:
                self._last_poss_team = team
                self._poss_candidate = None
                self._poss_candidate_streak = 0

        self.possession_frames[self._last_poss_team] += 1
        self.total_possession_frames += 1

    def get_possession_pct(self):
        total = max(1, self.total_possession_frames)
        return {t: self.possession_frames[t] / total for t in (1, 2)}

    # ═══════ 2. PASS COUNT ═══════════════════════════════════════
    def update_pass_count(self, frame_idx, tracks_frame, ball_action_spot):
        pred_action = ball_action_spot.video_pred_actions[frame_idx]
        pass_active = pred_action[0] > 0.5  # PASS class index = 0

        # Rising edge: was inactive → now active
        if pass_active and not self._prev_pass_active:
            bc = self._ball_center(tracks_frame)
            if bc is not None:
                _, team, _ = self._nearest_player(tracks_frame, bc)
                if team in (1, 2):
                    self.pass_counts[team] += 1

        self._prev_pass_active = pass_active

    # ═══════ 3. SPEED ════════════════════════════════════════════
    def update_speed(self, tracks_frame, cam_calib, frame_w, frame_h,
                     cam_dx: float = 0.0, cam_dy: float = 0.0):
        """Estimate speed from calibrated foot positions."""
        if cam_calib.H is None:
            return

        for pid, info in (tracks_frame.get("players") or {}).items():
            team = info.get("team")
            if team:
                self._player_teams[pid] = team

            x1, y1, x2, y2 = info["bbox"]
            x1_n, y1_n = x1 / frame_w, y1 / frame_h
            x2_n, y2_n = x2 / frame_w, y2 / frame_h
            try:
                feet = cam_calib.calibrate_player_feet((x1_n, y1_n, x2_n, y2_n))
            except Exception:
                continue
            if feet is None:
                continue

            cal_x, cal_y = float(feet[0]), float(feet[1])

            if pid in self._prev_feet:
                px, py = self._prev_feet[pid]
                dx_px = cal_x - px
                dy_px = cal_y - py

                # Compensate camera movement (scaled to minimap space)
                scale_x = cam_calib.IMG_W / frame_w if frame_w else 1.0
                scale_y = cam_calib.IMG_H / frame_h if frame_h else 1.0
                dx_px -= cam_dx * scale_x
                dy_px -= cam_dy * scale_y

                # Convert pixels → metres (rough)
                dx_m = dx_px / max(0.01, self.px_per_m_x)
                dy_m = dy_px / max(0.01, self.px_per_m_y)
                dist_m = (dx_m ** 2 + dy_m ** 2) ** 0.5

                speed_ms = dist_m * self.fps
                speed_kmh = speed_ms * 3.6

                # Reject outliers (teleportation artefacts)
                if speed_kmh < 45:
                    self._speed_buf[pid].append(speed_kmh)
                    self._total_dist[pid] += dist_m

                    # Update top speed
                    if team in (1, 2):
                        self._top_speed[team] = max(self._top_speed[team], speed_kmh)

            self._prev_feet[pid] = (cal_x, cal_y)

        # Compute smoothed speeds
        for pid, buf in self._speed_buf.items():
            if len(buf) > 0:
                self._cur_speeds[pid] = sum(buf) / len(buf)

    def get_player_speed(self, pid):
        return self._cur_speeds.get(pid, 0.0)

    # ═══════ 4. HEATMAP ══════════════════════════════════════════
    def update_heatmap(self, tracks_frame, cam_calib, frame_w, frame_h):
        if cam_calib.H is None:
            return
        for pid, info in (tracks_frame.get("players") or {}).items():
            team = info.get("team")
            if team not in (1, 2):
                continue
            x1, y1, x2, y2 = info["bbox"]
            try:
                feet = cam_calib.calibrate_player_feet(
                    (x1 / frame_w, y1 / frame_h, x2 / frame_w, y2 / frame_h))
            except Exception:
                continue
            if feet is None:
                continue
            ix, iy = int(feet[0]), int(feet[1])
            if 0 <= ix < self.minimap_w and 0 <= iy < self.minimap_h:
                heat = self._heat_t1 if team == 1 else self._heat_t2
                heat[iy, ix] += 1.0

    def get_heatmap_overlay(self, target_h, target_w):
        """Return a blendable BGR heatmap at (target_h, target_w)."""
        overlay = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        for team, heat_raw in [(1, self._heat_t1), (2, self._heat_t2)]:
            if heat_raw.max() < 1:
                continue
            # Blur & normalize
            blurred = cv2.GaussianBlur(heat_raw, (51, 51), 0)
            normed = (blurred / max(blurred.max(), 1e-6) * 255).astype(np.uint8)
            resized = cv2.resize(normed, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            # Apply team-colored colormap
            tc = self.team_colors.get(team, (128, 128, 128))
            colored = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            for ch in range(3):
                colored[:, :, ch] = (resized.astype(float) / 255 * tc[ch]).astype(np.uint8)

            overlay = cv2.add(overlay, colored)

        return overlay

    # ═══════════════════════════════════════════════════════════════
    # ███  DRAWING — Premium broadcast HUD  ████████████████████████
    # ═══════════════════════════════════════════════════════════════

    def draw_possession_bar(self, frame):
        """Draw a sleek possession bar at the bottom of the frame."""
        h, w = frame.shape[:2]
        bar_w = int(w * 0.50)
        bar_h = 14
        bar_x = (w - bar_w) // 2
        bar_y = h - 48

        pct = self.get_possession_pct()
        c1 = self.team_colors.get(1, (255, 150, 0))
        c2 = self.team_colors.get(2, (0, 100, 255))

        # Glass panel background
        panel_pad = 10
        _glass_panel(frame,
                     bar_x - panel_pad - 55, bar_y - 22,
                     bar_x + bar_w + panel_pad + 55, bar_y + bar_h + 10,
                     alpha=0.60)

        # "POSSESSION" label
        _put_text(frame, "POSSESSION", (bar_x + bar_w // 2 - 42, bar_y - 8),
                  scale=0.35, color=_ACCENT, thickness=1)

        # Gradient bar
        _gradient_bar(frame, bar_x, bar_y, bar_w, bar_h, pct[1], c1, c2)

        # Percentage labels
        p1_str = f"{pct[1]*100:.0f}%"
        p2_str = f"{pct[2]*100:.0f}%"
        _put_text(frame, p1_str, (bar_x - 50, bar_y + bar_h - 1), scale=0.50, color=c1, thickness=1)
        _put_text(frame, p2_str, (bar_x + bar_w + 10, bar_y + bar_h - 1), scale=0.50, color=c2, thickness=1)

        return frame

    def draw_pass_count(self, frame):
        """Draw pass count panel — top-left, below camera movement overlay."""
        c1 = self.team_colors.get(1, (255, 150, 0))
        c2 = self.team_colors.get(2, (0, 100, 255))

        px, py = 8, 60  # below cam overlay (ends at ~52)
        pw, ph = 155, 68

        _glass_panel(frame, px, py, px + pw, py + ph, alpha=0.60)

        # Title with accent bar
        cv2.line(frame, (px + 8, py + 6), (px + 8, py + 18), _ACCENT, 2, cv2.LINE_AA)
        _put_text(frame, "PASSES", (px + 14, py + 17), scale=0.38, color=_ACCENT)

        # Team 1
        cv2.circle(frame, (px + 16, py + 34), 5, c1, -1, cv2.LINE_AA)
        _put_text(frame, f"Team 1:  {self.pass_counts[1]}", (px + 26, py + 38),
                  scale=0.40, color=_WHITE)
        # Team 2
        cv2.circle(frame, (px + 16, py + 54), 5, c2, -1, cv2.LINE_AA)
        _put_text(frame, f"Team 2:  {self.pass_counts[2]}", (px + 26, py + 58),
                  scale=0.40, color=_WHITE)

        return frame

    def draw_speed_labels(self, frame, tracks_frame):
        """Draw speed (km/h) near each player."""
        for pid, info in (tracks_frame.get("players") or {}).items():
            speed = self.get_player_speed(pid)
            if speed < 0.5:
                continue
            x1, y1, x2, y2 = info["bbox"]
            # Position label above player
            lx = int((x1 + x2) / 2) - 18
            ly = int(y1) - 8
            if ly < 14:
                ly = int(y2) + 14

            label = f"{speed:.0f}"
            (tw, th), _ = cv2.getTextSize(label, _FONT, 0.32, 1)
            # Tiny glass pill
            cv2.rectangle(frame, (lx - 2, ly - th - 3), (lx + tw + 12, ly + 3),
                          (0, 0, 0), -1, cv2.LINE_AA)
            cv2.addWeighted(frame, 1.0, frame, 0.0, 0, frame)  # noop but keeps shape
            _put_text(frame, label, (lx, ly), scale=0.32, color=_WHITE)
            # "km/h" suffix — smaller & dimmer
            _put_text(frame, "km/h", (lx + tw + 1, ly), scale=0.22, color=_DIM)

        return frame

    def draw_heatmap_on_minimap(self, minimap):
        """Blend heatmap overlay onto the minimap image."""
        mh, mw = minimap.shape[:2]
        overlay = self.get_heatmap_overlay(mh, mw)
        if overlay.max() > 0:
            cv2.addWeighted(overlay, 0.45, minimap, 1.0, 0, minimap)
        return minimap

    # ═══════ SUMMARY FRAME ═══════════════════════════════════════
    def draw_summary_frame(self, w, h):
        """Generate a full-screen match summary with broadcast aesthetic."""
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Background gradient (dark to slightly lighter)
        for row in range(h):
            v = int(15 + 10 * (row / h))
            frame[row, :] = (v, v, v + 3)

        c1 = self.team_colors.get(1, (255, 150, 0))
        c2 = self.team_colors.get(2, (0, 100, 255))
        pct = self.get_possession_pct()
        cx = w // 2

        # ── Header ──
        # Accent line
        cv2.line(frame, (cx - 180, 30), (cx + 180, 30), _ACCENT, 2, cv2.LINE_AA)
        _put_text(frame, "MATCH ANALYSIS", (cx - 82, 58), scale=0.70, color=_WHITE, thickness=2)
        cv2.line(frame, (cx - 180, 68), (cx + 180, 68), _ACCENT, 2, cv2.LINE_AA)

        # ── Stats panel ──
        panel_x = cx - 260
        panel_w = 520
        panel_y = 95
        _glass_panel(frame, panel_x, panel_y, panel_x + panel_w, panel_y + 200, alpha=0.50)

        # Column headers
        col_l = panel_x + 30    # Team 1 column
        col_c = cx              # center (labels)
        col_r = panel_x + panel_w - 90  # Team 2 column

        # Team dots in header
        cv2.circle(frame, (col_l + 25, panel_y + 25), 8, c1, -1, cv2.LINE_AA)
        _put_text(frame, "TEAM 1", (col_l + 40, panel_y + 30), scale=0.45, color=c1)
        cv2.circle(frame, (col_r + 25, panel_y + 25), 8, c2, -1, cv2.LINE_AA)
        _put_text(frame, "TEAM 2", (col_r + 40, panel_y + 30), scale=0.45, color=c2)

        # Separator
        cv2.line(frame, (panel_x + 15, panel_y + 45), (panel_x + panel_w - 15, panel_y + 45),
                 (50, 50, 55), 1, cv2.LINE_AA)

        # ── Row: Possession ──
        row_y = panel_y + 72
        _put_text(frame, "POSSESSION", (col_c - 48, row_y), scale=0.40, color=_ACCENT)
        _put_text(frame, f"{pct[1]*100:.0f}%", (col_l + 20, row_y), scale=0.55, color=c1, thickness=2)
        _put_text(frame, f"{pct[2]*100:.0f}%", (col_r + 20, row_y), scale=0.55, color=c2, thickness=2)

        # Possession bar
        bar_y = row_y + 8
        _gradient_bar(frame, panel_x + 40, bar_y, panel_w - 80, 8, pct[1], c1, c2)

        # ── Row: Passes ──
        row_y = panel_y + 110
        _put_text(frame, "PASSES", (col_c - 30, row_y), scale=0.40, color=_ACCENT)
        _put_text(frame, str(self.pass_counts[1]), (col_l + 30, row_y), scale=0.55, color=c1, thickness=2)
        _put_text(frame, str(self.pass_counts[2]), (col_r + 30, row_y), scale=0.55, color=c2, thickness=2)

        # ── Row: Top Speed ──
        row_y = panel_y + 145
        _put_text(frame, "TOP SPEED", (col_c - 42, row_y), scale=0.40, color=_ACCENT)
        _put_text(frame, f"{self._top_speed[1]:.1f} km/h", (col_l + 5, row_y),
                  scale=0.45, color=c1)
        _put_text(frame, f"{self._top_speed[2]:.1f} km/h", (col_r + 5, row_y),
                  scale=0.45, color=c2)

        # ── Row: Total Distance ──
        row_y = panel_y + 180
        dist_t1 = sum(self._total_dist[pid] for pid, t in self._player_teams.items() if t == 1)
        dist_t2 = sum(self._total_dist[pid] for pid, t in self._player_teams.items() if t == 2)
        _put_text(frame, "TOTAL DIST", (col_c - 45, row_y), scale=0.40, color=_ACCENT)
        _put_text(frame, f"{dist_t1/1000:.1f} km", (col_l + 10, row_y), scale=0.45, color=c1)
        _put_text(frame, f"{dist_t2/1000:.1f} km", (col_r + 10, row_y), scale=0.45, color=c2)

        # ── Heatmaps side by side ──
        heat_y = panel_y + 225
        heat_w = int(w * 0.35)
        heat_h = int(heat_w * 68 / 105)

        for team_idx, (heat_raw, hx) in enumerate([
            (self._heat_t1, cx - heat_w - 20),
            (self._heat_t2, cx + 20),
        ], start=1):
            tc = c1 if team_idx == 1 else c2
            team_label = f"TEAM {team_idx} HEATMAP"

            # Label
            _put_text(frame, team_label, (hx + 5, heat_y - 5), scale=0.35, color=tc)

            # Dark pitch background
            pitch_bg = np.full((heat_h, heat_w, 3), (10, 40, 10), dtype=np.uint8)

            # Draw pitch lines (simplified)
            cv2.rectangle(pitch_bg, (5, 5), (heat_w - 5, heat_h - 5), (30, 80, 30), 1, cv2.LINE_AA)
            cv2.line(pitch_bg, (heat_w // 2, 5), (heat_w // 2, heat_h - 5), (30, 80, 30), 1, cv2.LINE_AA)
            cv2.circle(pitch_bg, (heat_w // 2, heat_h // 2), heat_h // 4, (30, 80, 30), 1, cv2.LINE_AA)

            # Overlay heatmap
            if heat_raw.max() > 0:
                blurred = cv2.GaussianBlur(heat_raw, (51, 51), 0)
                normed = (blurred / max(blurred.max(), 1e-6) * 255).astype(np.uint8)
                resized = cv2.resize(normed, (heat_w, heat_h))
                colored = np.zeros_like(pitch_bg)
                for ch in range(3):
                    colored[:, :, ch] = (resized.astype(float) / 255 * tc[ch]).astype(np.uint8)
                cv2.addWeighted(colored, 0.7, pitch_bg, 1.0, 0, pitch_bg)

            # Border
            _rounded_rect(pitch_bg, (0, 0), (heat_w - 1, heat_h - 1), (80, 80, 85), 1, radius=6)

            # Place on frame
            if heat_y + heat_h <= h and hx >= 0 and hx + heat_w <= w:
                frame[heat_y:heat_y + heat_h, hx:hx + heat_w] = pitch_bg

        # ── Footer ──
        footer_y = h - 30
        total_secs = self._frame_count / max(1, self.fps)
        _put_text(frame, f"Analyzed {self._frame_count} frames  |  {total_secs:.0f}s  |  {self.fps:.0f} fps",
                  (cx - 130, footer_y), scale=0.35, color=_DIM)

        # Bottom accent line
        cv2.line(frame, (cx - 180, footer_y + 10), (cx + 180, footer_y + 10), _ACCENT, 1, cv2.LINE_AA)

        return frame

    # ═══════ PER-FRAME TICK ══════════════════════════════════════
    def tick(self):
        """Call once per frame to advance internal counters."""
        self._frame_count += 1
