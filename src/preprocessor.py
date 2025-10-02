# src/preprocessor.py
"""影像前處理模組"""
import os
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """影像前處理器"""
    
    def __init__(
        self,
        n_select: int = 10,
        detection_confidence: float = 0.8,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: int = 8,
        mirror_size: Tuple[int, int] = (512, 512),
        feather_px: int = 2
    ):
        self.n_select = n_select
        self.detection_confidence = detection_confidence
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.mirror_size = mirror_size
        self.feather_px = feather_px
        self.face_mesh = None
    
    def __enter__(self):
        """初始化 MediaPipe"""
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.detection_confidence
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """離開 context manager，清理資源"""
        if self.face_mesh:
            self.face_mesh.close()
            self.face_mesh = None
    
    def process_folder(
        self,
        input_dir: str,
        output_dir: str,
        steps: List[str] = None
    ) -> Dict[str, List[Path]]:
        """處理整個資料夾 - 流程：select -> mirror -> clahe -> align"""
        if steps is None:
            steps = ['select', 'mirror', 'clahe', 'align']
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        total_processed = 0
        
        # 掃描所有包含圖片的資料夾
        image_folders = []
        
        for root, dirs, files in os.walk(input_path):
            root_path = Path(root)
            images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if images:
                rel_path = root_path.relative_to(input_path)
                image_folders.append((root_path, rel_path))
        
        logger.info(f"找到 {len(image_folders)} 個包含圖片的資料夾")
        
        # 處理每個包含圖片的資料夾
        for folder_path, rel_path in image_folders:
            logger.info(f"處理資料夾: {rel_path}")
            
            output_subfolder = output_path / rel_path
            output_subfolder.mkdir(parents=True, exist_ok=True)
            
            all_images = list(folder_path.glob("*.jpg")) + \
                        list(folder_path.glob("*.jpeg")) + \
                        list(folder_path.glob("*.png"))
            
            if not all_images:
                continue
            
            # Step 0: 篩選最佳圖片
            if 'select' in steps:
                selected_images = self.select_best_images(folder_path)
                logger.info(f"  從 {len(all_images)} 張中選擇了 {len(selected_images)} 張照片")
            else:
                selected_images = all_images
                logger.info(f"  處理全部 {len(selected_images)} 張照片")
            
            folder_results = []
            
            # 處理每張照片
            for img_path in tqdm(selected_images, desc=f"處理 {folder_path.name}"):
                try:
                    # 讀取影像
                    image = cv2.imread(str(img_path))
                    if image is None:
                        logger.warning(f"無法讀取: {img_path.name}")
                        continue
                    
                    stem = img_path.stem
                    
                    if 'mirror' in steps:
                        # Step 1: 建立鏡射
                        left_mirror, right_mirror = self.create_mirror_images(image)
                        
                        # Step 2: CLAHE 處理
                        if 'clahe' in steps:
                            left_mirror = self.apply_clahe(left_mirror)
                            right_mirror = self.apply_clahe(right_mirror)
                        
                        # Step 3: 角度校正
                        if 'align' in steps:
                            left_mirror = self.correct_face_angle(left_mirror)
                            right_mirror = self.correct_face_angle(right_mirror)
                        
                        # 儲存結果
                        left_path = output_subfolder / f"{stem}_Lmirror_claheL.png"
                        right_path = output_subfolder / f"{stem}_Rmirror_claheL.png"
                        
                        cv2.imwrite(str(left_path), left_mirror)
                        cv2.imwrite(str(right_path), right_mirror)
                        
                        folder_results.extend([left_path, right_path])
                        total_processed += 2
                        
                    else:
                        # 沒有鏡射的處理流程
                        if 'clahe' in steps:
                            image = self.apply_clahe(image)
                        if 'align' in steps:
                            image = self.correct_face_angle(image)
                        
                        output_file = output_subfolder / f"{stem}_processed.png"
                        cv2.imwrite(str(output_file), image)
                        folder_results.append(output_file)
                        total_processed += 1
                        
                except Exception as e:
                    logger.warning(f"處理 {img_path.name} 失敗: {e}")
                    continue
            
            results[str(rel_path)] = folder_results
            logger.info(f"  完成 {folder_path.name}: 產生 {len(folder_results)} 個檔案")
        
        logger.info(f"\n總共處理完成: {len(results)} 個資料夾, {total_processed} 個檔案")
        logger.info(f"輸出位置: {output_path}")
        
        return results
    
    def select_best_images(self, folder_path: Path) -> List[Path]:
        """選擇最佳的 n 張照片（基於臉部角度）"""
        if not self.face_mesh:
            raise RuntimeError("請使用 with 語句或先呼叫 __enter__")
        
        all_images = list(folder_path.glob("*.jpg")) + \
                    list(folder_path.glob("*.jpeg")) + \
                    list(folder_path.glob("*.png"))
        
        if len(all_images) <= self.n_select:
            return all_images
        
        angles = []
        for img_path in all_images:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)
            
            if not results.multi_face_landmarks:
                angles.append((img_path, float('inf')))
                continue
            
            angle = self._calculate_face_angle(results, image.shape)
            angles.append((img_path, abs(angle)))
        
        angles.sort(key=lambda x: x[1])
        return [path for path, _ in angles[:self.n_select]]
    
    def correct_face_angle(self, image: np.ndarray) -> np.ndarray:
        """校正臉部角度使其垂直"""
        if not self.face_mesh:
            raise RuntimeError("請使用 with 語句或先呼叫 __enter__")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return image
        
        angle = self._calculate_face_angle(results, image.shape)
        
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), -angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        
        return rotated
    
    def create_mirror_images(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """建立左右臉鏡射影像"""
        if not self.face_mesh:
            raise RuntimeError("請使用 with 語句或先呼叫 __enter__")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return image.copy(), image.copy()
        
        face_landmarks = results.multi_face_landmarks[0].landmark
        pts_xy = self._landmarks_to_xy(face_landmarks, image.shape)
        
        mask = self._build_face_mask(image.shape, pts_xy)
        
        p0, n = self._estimate_midline(pts_xy)
        
        left_mirror = self._align_to_canvas_premul(
            image, mask, p0, n, 
            side='left',
            out_size=self.mirror_size,
            margin=0.08
        )
        
        right_mirror = self._align_to_canvas_premul(
            image, mask, p0, n,
            side='right', 
            out_size=self.mirror_size,
            margin=0.08
        )
        
        return left_mirror, right_mirror
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """應用 CLAHE 直方圖均衡"""
        # 確保影像格式正確
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        # 轉換到 Lab 色彩空間
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 對 L 通道應用 CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(self.clahe_tile_size, self.clahe_tile_size)
        )
        l_eq = clahe.apply(l)
        
        # 合併並轉換回 BGR
        lab_eq = cv2.merge([l_eq, a, b])
        result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        
        return result
    
    def _calculate_face_angle(self, results, image_shape) -> float:
        """計算臉部中軸角度"""
        h, w = image_shape[:2]
        
        FACEMESH_MID_LINE = [
            (10, 151), (151, 9), (9, 8), (8, 168),
            (168, 6), (6, 197), (197, 195), (195, 5), (5, 4),
            (4, 1), (1, 19), (19, 94), (94, 2)
        ]
        
        angles = []
        for pair in FACEMESH_MID_LINE:
            point1 = results.multi_face_landmarks[0].landmark[pair[0]]
            point2 = results.multi_face_landmarks[0].landmark[pair[1]]
            
            dot1 = np.array([point1.x * w, point1.y * h])
            dot2 = np.array([point2.x * w, point2.y * h])
            
            vector = dot2 - dot1
            if vector[1] != 0:
                angle = np.arctan(vector[0] / vector[1])
                angles.append(np.degrees(angle))
            else:
                angles.append(90.0 if vector[0] > 0 else -90.0)
        
        return sum(angles) / len(angles) if angles else 0
    
    def _landmarks_to_xy(self, landmarks, img_shape: tuple) -> np.ndarray:
        """將 FaceMesh 的相對座標轉為像素座標"""
        h, w = img_shape[:2]
        pts = []
        for lm in landmarks:
            x = float(lm.x * w)
            y = float(lm.y * h)
            pts.append([x, y])
        return np.array(pts, dtype=np.float64)
    
    def _build_face_mask(self, img_shape: tuple, face_points_xy: np.ndarray) -> np.ndarray:
        """建立臉部遮罩"""
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        if face_points_xy.shape[0] == 0:
            return mask
        hull = cv2.convexHull(face_points_xy.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        return mask
    
    def _estimate_midline(
        self,
        face_points_xy: np.ndarray,
        midline_idx: tuple = (10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """估計臉部中線"""
        idx = np.array(midline_idx, dtype=int)
        idx = idx[(idx >= 0) & (idx < face_points_xy.shape[0])]
        
        if idx.size == 0:
            ml_pts = face_points_xy
        else:
            idx = np.unique(idx)
            ml_pts = face_points_xy[idx, :]
        
        p0 = ml_pts.mean(axis=0)
        X = ml_pts - p0
        
        if not np.isfinite(X).all() or np.allclose(X, 0):
            xs = face_points_xy[:, 0]
            mid_x = 0.5 * (xs.min() + xs.max())
            p0 = np.array([mid_x, face_points_xy[:, 1].mean()], dtype=np.float64)
            n = np.array([1.0, 0.0], dtype=np.float64)
            return p0, n
        
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        u = Vt[0]
        u = u / (np.linalg.norm(u) + 1e-12)
        n = np.array([-u[1], u[0]], dtype=np.float64)
        
        if n[0] < 0:
            n = -n
        
        return p0, n
    
    def _align_to_canvas_premul(
        self,
        img_bgr: np.ndarray,
        mask_u8: np.ndarray,
        p0: np.ndarray,
        n: np.ndarray,
        side: str,
        out_size: Tuple[int, int] = (512, 512),
        margin: float = 0.08
    ) -> np.ndarray:
        """對齊到畫布並使用預乘 alpha"""
        H, W = out_size
        h, w = img_bgr.shape[:2]
        
        # 計算有號距離
        X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        d = (X - p0[0]) * n[0] + (Y - p0[1]) * n[1]
        
        # 反射座標
        Xr = X - 2.0 * d * n[0]
        Yr = Y - 2.0 * d * n[1]
        
        # 建立半臉 alpha
        if side == 'left':
            region = (mask_u8 > 0) & (d < 0)
        else:
            region = (mask_u8 > 0) & (d > 0)
        
        alpha = np.zeros_like(mask_u8, dtype=np.uint8)
        alpha[region] = 255
        
        # 羽化邊緣
        if self.feather_px > 0:
            alpha = cv2.GaussianBlur(alpha, (self.feather_px*2+1, self.feather_px*2+1), 0)
        
        alpha_f = alpha.astype(np.float32) / 255.0
        
        # 反射另一半
        reflected = cv2.remap(img_bgr, Xr, Yr, cv2.INTER_LINEAR)
        reflected_alpha = cv2.remap(alpha_f, Xr, Yr, cv2.INTER_LINEAR)
        
        # 合成
        img_f = img_bgr.astype(np.float32) / 255.0
        result_f = img_f * alpha_f[..., None] + \
                  (reflected.astype(np.float32) / 255.0) * reflected_alpha[..., None]
        final_alpha = np.clip(alpha_f + reflected_alpha, 0, 1)
        
        # 除以 alpha 還原
        eps = 1e-6
        result_f = np.where(final_alpha[..., None] > eps, 
                           result_f / final_alpha[..., None], 0)
        result = np.clip(result_f * 255, 0, 255).astype(np.uint8)
        
        # 不做旋轉，直接使用合成結果
        alpha_mask = (final_alpha * 255).astype(np.uint8)
        
        # 找邊界並裁切
        ys, xs = np.where(alpha_mask > 0)
        
        if len(xs) == 0:
            return np.zeros((H, W, 3), dtype=np.uint8)
        
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        
        cropped = result[y0:y1+1, x0:x1+1]
        
        # 計算縮放比例
        face_w = x1 - x0 + 1
        face_h = y1 - y0 + 1
        
        available_w = W * (1 - 2 * margin)
        available_h = H * (1 - 2 * margin)
        
        scale = min(available_w / face_w, available_h / face_h, 1.0)
        
        # 縮放
        new_w = int(face_w * scale)
        new_h = int(face_h * scale)
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 置中到畫布
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        start_x = (W - new_w) // 2
        start_y = (H - new_h) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        
        return canvas