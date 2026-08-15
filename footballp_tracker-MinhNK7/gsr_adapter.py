import json
from typing import Dict, Any, List, Optional

class GameStateAdapter:

    def __init__(self, out_jsonl: Optional[str] = None):
        self.out_jsonl = out_jsonl
        if self.out_jsonl:
            # clear file cũ
            open(self.out_jsonl, "w", encoding="utf-8").close()

    @staticmethod
    def _bbox_xyxy_to_center(bbox):
        x1, y1, x2, y2 = map(float, bbox)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        return cx, cy, w, h

    def build_frame_state(
        self,
        frame_idx: int,
        tracks_frame: Dict[str, Dict[int, Dict[str, Any]]]
    ) -> Dict[str, Any]:

        players_out: List[Dict[str, Any]] = []
        for pid, info in (tracks_frame.get("players") or {}).items():
            # Kiểm tra nếu pid là None, nếu có thì xử lý và gán giá trị mặc định
            if pid is None:
                pid = -1  # Gán giá trị mặc định nếu pid là None
            cx, cy, w, h = self._bbox_xyxy_to_center(info["bbox"])
            players_out.append({
                "id": int(pid),  # Chuyển thành int sau khi đảm bảo không phải None
                "cx": cx, "cy": cy, "w": w, "h": h,
                "team": int(info.get("team", 0))
            })

        refs_out: List[Dict[str, Any]] = []
        for rid, info in (tracks_frame.get("refs") or {}).items():
            # Kiểm tra nếu rid là None, nếu có thì xử lý và gán giá trị mặc định
            if rid is None:
                rid = -1  # Gán giá trị mặc định nếu rid là None
            cx, cy, w, h = self._bbox_xyxy_to_center(info["bbox"])
            refs_out.append({
                "id": int(rid),  # Chuyển thành int sau khi đảm bảo không phải None
                "cx": cx, "cy": cy, "w": w, "h": h
            })

        ball_out = None
        ball_dict = tracks_frame.get("ball") or {}
        if 1 in ball_dict and "bbox" in ball_dict[1]:
            cx, cy, w, h = self._bbox_xyxy_to_center(ball_dict[1]["bbox"])
            ball_out = {"cx": cx, "cy": cy, "w": w, "h": h}

        return {
            "frame_index": int(frame_idx),
            "players": players_out,
            "refs": refs_out,
            "ball": ball_out
        }

    def emit(self, frame_idx: int, tracks_frame: Dict[str, Any]) -> Dict[str, Any]:
        state = self.build_frame_state(frame_idx, tracks_frame)
        if self.out_jsonl:
            with open(self.out_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(state) + "\n")
        return state
