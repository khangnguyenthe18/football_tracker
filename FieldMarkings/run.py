import os
import json
import concurrent.futures as cf
import cv2
import torch
import numpy as np
import sys

sys.path.append('FieldMarkings')
from tqdm import tqdm
from argus import load_model
from torchvision import transforms as T
from baseline.camera import unproject_image_point
from baseline.baseline_cameras import draw_pitch_homography
from src.datatools.ellipse import PITCH_POINTS
from src.models.hrnet.metamodel import HRNetMetaModel
from src.models.hrnet.prediction import CameraCreator


MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pipeline_models', 'data', 'models', 'HRNet_57_hrnet48x2_57_003', 'evalai-018-0.536880.pth')

LINES_FILE = None # '/workdir/data/result/line_model_result.pkl'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class CamCalib:
    # FIFA standard pitch dimensions (metres)
    PITCH_LENGTH = 105.0
    PITCH_WIDTH = 68.0

    def __init__(self, keypoints_path, lines_path):
        self.IMG_W = 960
        self.IMG_H = 540

        # Auto-calculate scale factor so the full pitch fits the minimap
        margin = 20  # px padding on each side
        f_x = (self.IMG_W - 2 * margin) / self.PITCH_LENGTH
        f_y = (self.IMG_H - 2 * margin) / self.PITCH_WIDTH
        self.f = min(f_x, f_y)  # keep aspect ratio → ~7.35
        self.model = load_model(keypoints_path, loss=None, optimizer=None, device=DEVICE)

        self.calibrator = CameraCreator(
            PITCH_POINTS, conf_thresh=0.5, conf_threshs=[0.5, 0.35, 0.2],
            algorithm='iterative_voter',
            lines_file=lines_path, max_rmse=55.0, max_rmse_rel=5.0,
            min_points=5, min_focal_length=10.0, min_points_per_plane=6,
            min_points_for_refinement=6, reliable_thresh=57
        )

        self.H = None

    def __call__(self, img):
        to_tensor = T.ToTensor()
        img = cv2.resize(img, (self.IMG_W, self.IMG_H))
        tensor = to_tensor(img).unsqueeze(0).to(DEVICE)
        pred = self.model.predict(tensor).cpu().numpy()[0]
        cam = self.calibrator(pred)
        
        if cam is not None:
            self.H = cam.calibration @ cam.rotation @ np.concatenate((np.eye(3)[:, :2], -cam.position.reshape(3, 1)), axis=1)

    def calibrate_player_feet(self, xyxyn):
        if self.H is None:
            return None

        x1, y1, x2, y2 = xyxyn
        x1 *= self.IMG_W
        y1 *= self.IMG_H
        x2 *= self.IMG_W
        y2 *= self.IMG_H
        point2D = np.array([x1 + (x2 - x1) / 2, y2, 1])

        top_view_h = np.array([[self.f, 0, self.IMG_W/2], [0, self.f, self.IMG_H/2], [0, 0, 1]])
        
        feet = unproject_image_point(self.H, point2D=point2D)
        imaged_feets = top_view_h @ np.array([feet[0], feet[1], 1])
        imaged_feets /= imaged_feets[2]

        return imaged_feets

    def draw(self, img, colors, feets, heatmap_overlay=None):
        if self.H is None:
            return None

        # Dark green background for pitch look
        bg = np.full((self.IMG_H, self.IMG_W, 3), (10, 40, 10), dtype=np.uint8)
        top_view_h = np.array([[self.f, 0, self.IMG_W/2], [0, self.f, self.IMG_H/2], [0, 0, 1]])
        drawn = draw_pitch_homography(bg, top_view_h)

        # Blend heatmap overlay if provided
        if heatmap_overlay is not None:
            hm = cv2.resize(heatmap_overlay, (drawn.shape[1], drawn.shape[0]))
            if hm.max() > 0:
                cv2.addWeighted(hm, 0.45, drawn, 1.0, 0, drawn)

        for color, feet in zip(colors, feets):
            if feet is not None:
                pt = (int(feet[0]), int(feet[1]))
                if not (0 <= pt[0] < self.IMG_W and 0 <= pt[1] < self.IMG_H):
                    continue
                # Convert numpy color to tuple of ints
                c = tuple(int(v) for v in color) if hasattr(color, '__iter__') else color
                cv2.circle(drawn, pt, 10, (255, 255, 255), 2, cv2.LINE_AA)  # white outline
                cv2.circle(drawn, pt, 8, c, -1, cv2.LINE_AA)               # filled dot
        
        return drawn

def blend(img1, img2, scale=None, alpha=0.82, reserved_bottom=None):
    """Blend a responsive minimap above the bottom HUD."""
    img1_h, img1_w = img1.shape[:2]
    source_h, source_w = img2.shape[:2]

    if reserved_bottom is None:
        reserved_bottom = max(64, int(img1_h * 0.07))

    max_w = max(1, int(img1_w * 0.28))
    max_h = max(1, int((img1_h - reserved_bottom) * 0.30))
    fit_scale = min(max_w / source_w, max_h / source_h, 1.0)
    if scale is not None:
        fit_scale = min(fit_scale, scale)

    target_w = max(1, int(round(source_w * fit_scale)))
    target_h = max(1, int(round(source_h * fit_scale)))
    img2 = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_AREA)
    img2_h, img2_w = img2.shape[:2]

    margin = 10
    x1 = img1_w - img2_w - margin
    y1 = img1_h - reserved_bottom - img2_h - margin
    x2 = x1 + img2_w
    y2 = y1 + img2_h

    # Clamp to frame bounds
    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    x2 = min(x2, img1_w)
    y2 = min(y2, img1_h)
    img2 = img2[:y2-y1, :x2-x1]

    roi = img1[y1:y2, x1:x2]
    img1[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, img2, alpha, 0)

    # White border
    cv2.rectangle(img1, (x1, y1), (x2 - 1, y2 - 1), (180, 180, 180), 1, cv2.LINE_AA)

    return img1
