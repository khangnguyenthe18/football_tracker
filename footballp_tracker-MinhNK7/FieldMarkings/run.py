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


MODEL_PATH = 'data/models/HRNet_57_hrnet48x2_57_003/evalai-018-0.536880.pth'
LINES_FILE = None # '/workdir/data/result/line_model_result.pkl'
DEVICE = 'cuda:0'  # Device for running inference


class CamCalib:
    def __init__(self, keypoints_path, lines_path):
        self.IMG_W = 960
        self.IMG_H = 540
        self.f = 6
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

    def draw(self, img, colors, feets):
        if self.H is None:
            return None

        black_img = np.zeros((self.IMG_H, self.IMG_W, 3), dtype=np.uint8)
        top_view_h = np.array([[self.f, 0, self.IMG_W/2], [0, self.f, self.IMG_H/2], [0, 0, 1]])
        drawn = draw_pitch_homography(black_img, top_view_h)

        for color, feet in zip(colors, feets):
            if feet is not None:
                cv2.circle(drawn, (int(feet[0]), int(feet[1])), 8, color, -1)
        
        return drawn

def blend(img1, img2, scale=0.5, alpha=0.5):
    img2 = cv2.resize(img2, (int(img2.shape[1]*scale), int(img2.shape[0]*scale)))
    img2_h, img2_w = img2.shape[:2]

    img1_h, img1_w = img1.shape[:2]
    x1, y1 = img1_w//2 - img2_w//2, img1_h - img2_h
    x2, y2 = img1_w//2 + img2_w//2, img1_h
    roi = img1[y1:y2, x1:x2]
    img1[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, img2, alpha, 0)

    return img1