# src/preprocessor.py
"""影像前處理模組 - 整合步驟 0-3"""
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

# MediaPipe 臉部中線定義
FACEMESH_MID_LINE = frozenset([
    (10, 151), (151, 9), (9, 8), (8, 168),
    (168, 6), (6, 197), (197, 195), (195, 5), (5, 4),
    (4, 1), (1, 19), (19, 94), (94, 2)
])


class ImagePreprocessor:
    """影像前處理器：篩選、校正、鏡射、直方圖均衡"""
    
    def __init__(
        self,
        detection_confidence: float = 0.8,
        tracking_confidence: float = 0.8,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: int = 8,
        n_select: int = 10,
        feather_px: int = 2,
        mirror_size: Tuple[int, int] = (512, 512)
    ):
        """
        Args:
            detection_confidence: MediaPipe 偵測信心值
            tracking_confidence: MediaPipe 追蹤信心值
            clahe_clip_limit: CLAHE 限制參數
            clahe_tile_size: CLAHE 分塊大小
            n_select: 每個個案選取的照片數量
            feather_px: 鏡射時邊緣羽化像素
            mirror_size: 鏡射輸出影像大小
        """
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.n_select = n_select
        self.feather_px = feather_px
        self.mirror_size = mirror_size
        
        # 初始化 MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = None
        
    def __enter__(self):
        """Context manager 進入"""
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 離開"""
        if self.face_mesh:
            self.face_mesh.close()
    
    # ========== 主要處理流程 ==========
    def process_folder(
        self,
        input_dir: str,
        output_dir: str,
        steps: List[str] = None
    ) -> Dict[str, List[Path]]:
        """處理整個資料夾
        
        Args:
            input_dir: 輸入資料夾
            output_dir: 輸出資料夾  
            steps: 要執行的步驟 ['select', 'align', 'mirror', 'clahe']
                   如果為 None 則執行所有步驟
        
        Returns:
            處理結果的檔案路徑字典
        """
        if steps is None:
            steps = ['select', 'align', 'mirror', 'clahe']
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # 對每個子資料夾處理
        for subfolder in input_path.iterdir():
            if not subfolder.is_dir():
                continue
                
            logger.info(f"處理資料夾: {subfolder.name}")
            subfolder_results = []
            
            # Step 0: 篩選最佳照片
            if 'select' in steps:
                selected_images = self.select_best_images(subfolder)
                logger.info(f"  選擇了 {len(selected_images)} 張照片")
            else:
                selected_images = list(subfolder.glob("*.jpg")) + list(subfolder.glob("*.png"))
            
            # 為每張選中的照片執行處理
            for img_path in tqdm(selected_images, desc=f"處理 {subfolder.name}"):
                try:
                    result_path = self._process_single_image(
                        img_path,
                        output_path / subfolder.name,
                        steps
                    )
                    subfolder_results.extend(result_path)
                except Exception as e:
                    logger.warning(f"處理 {img_path.name} 失敗: {e}")
                    continue
            
            results[subfolder.name] = subfolder_results
            
        return results
    
    def _process_single_image(
        self,
        image_path: Path,
        output_dir: Path,
        steps: List[str]
    ) -> List[Path]:
        """處理單張影像"""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = []
        
        # 讀取影像
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"無法讀取影像: {image_path}")
        
        # Step 1: 角度校正
        if 'align' in steps:
            image = self.correct_face_angle(image)
        
        # Step 2: 建立鏡射
        if 'mirror' in steps:
            left_mirror, right_mirror = self.create_mirror_images(image)
            
            # Step 3: CLAHE 處理
            if 'clahe' in steps:
                left_mirror = self.apply_clahe(left_mirror)
                right_mirror = self.apply_clahe(right_mirror)
            
            # 儲存結果
            stem = image_path.stem
            left_path = output_dir / f"{stem}_Lmirror_claheL.png"
            right_path = output_dir / f"{stem}_Rmirror_claheL.png"
            
            cv2.imwrite(str(left_path), left_mirror)
            cv2.imwrite(str(right_path), right_mirror)
            
            output_paths = [left_path, right_path]
        else:
            # 如果不做鏡射，只做 CLAHE
            if 'clahe' in steps:
                image = self.apply_clahe(image)
            
            output_path = output_dir / f"{image_path.stem}_processed.png"
            cv2.imwrite(str(output_path), image)
            output_paths = [output_path]
        
        return output_paths
    
    # ========== Step 0: 照片篩選 ==========
    def select_best_images(self, folder: Path) -> List[Path]:
        """根據臉部中軸角度選擇最佳照片"""
        if not self.face_mesh:
            raise RuntimeError("請使用 with 語句或先呼叫 __enter__")
        
        angles = {}
        image_paths = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        
        for img_path in image_paths:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
                
            height, width = image.shape[:2]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)
            
            if not results.multi_face_landmarks:
                continue
            
            # 計算中軸角度
            angle = self._calculate_midline_angle(results, height, width)
            angles[img_path] = angle
        
        # 選擇角度最小的 n 張
        sorted_images = sorted(angles.items(), key=lambda x: x[1])[:self.n_select]
        return [img for img, _ in sorted_images]
    
    def _calculate_midline_angle(self, results, height: int, width: int) -> float:
        """計算臉部中軸角度（使用4點法）"""
        landmarks = results.multi_face_landmarks[0].landmark
        
        # 使用關鍵點 10, 168, 4, 2
        points = [10, 168, 4, 2]
        dots = []
        for idx in points:
            point = landmarks[idx]
            dots.append(np.array([point.x * width, point.y * height, 0]))
        
        # 計算向量夾角
        total_angle = 0
        for i in range(len(dots) - 2):
            v1 = dots[i+1] - dots[i]
            v2 = dots[i+2] - dots[i+1]
            
            dot_product = np.dot(v1, v2)
            norm = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norm > 0:
                angle = np.arccos(np.clip(dot_product / norm, -1.0, 1.0))
                total_angle += np.degrees(angle)
        
        return total_angle
    
    # ========== Step 1: 角度校正 ==========
    def correct_face_angle(self, image: np.ndarray) -> np.ndarray:
        """校正臉部角度使其垂直"""
        if not self.face_mesh:
            raise RuntimeError("請使用 with 語句或先呼叫 __enter__")
        
        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return image
        
        # 計算中軸線角度
        angle = self._calculate_rotation_angle(results, h, w)
        
        # 旋轉影像
        M = cv2.getRotationMatrix2D((w//2, h//2), -angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        
        return rotated
    
    def _calculate_rotation_angle(self, results, height: int, width: int) -> float:
        """計算需要旋轉的角度"""
        angles = []
        
        for pair in FACEMESH_MID_LINE:
            point1 = results.multi_face_landmarks[0].landmark[pair[0]]
            point2 = results.multi_face_landmarks[0].landmark[pair[1]]
            
            dot1 = np.array([point1.x * width, point1.y * height])
            dot2 = np.array([point2.x * width, point2.y * height])
            
            vector = dot2 - dot1
            if vector[1] != 0:
                angle = np.degrees(np.arctan(vector[0] / vector[1]))
                angles.append(angle)
        
        return np.mean(angles) if angles else 0
    
    # ========== Step 2: 鏡射處理 ==========
    def create_mirror_images(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """建立左右臉鏡射影像"""
        if not self.face_mesh:
            raise RuntimeError("請使用 with 語句或先呼叫 __enter__")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            # 如果偵測不到臉，返回原圖
            return image.copy(), image.copy()
        
        # 取得臉部關鍵點
        face_landmarks = results.multi_face_landmarks[0].landmark
        pts_xy = self._landmarks_to_xy(face_landmarks, image.shape)
        
        # 建立臉部遮罩
        mask = self._build_face_mask(image.shape, pts_xy)
        
        # 估計中線
        p0, n = self._estimate_midline(pts_xy)
        
        # 計算有號距離
        h, w = image.shape[:2]
        X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        d = (X - p0[0]) * n[0] + (Y - p0[1]) * n[1]
        
        # 計算反射座標
        Xr = X - 2.0 * d * n[0]
        Yr = Y - 2.0 * d * n[1]
        
        # 建立左右半臉
        left_mirror = self._create_symmetric_face(image, mask, d, Xr, Yr, 'left')
        right_mirror = self._create_symmetric_face(image, mask, d, Xr, Yr, 'right')
        
        # 調整輸出大小
        left_mirror = cv2.resize(left_mirror, self.mirror_size)
        right_mirror = cv2.resize(right_mirror, self.mirror_size)
        
        return left_mirror, right_mirror
    
    def _landmarks_to_xy(self, landmarks, img_shape: tuple) -> np.ndarray:
        """將相對座標轉為像素座標"""
        h, w = img_shape[:2]
        pts = []
        for lm in landmarks:
            x = float(lm.x * w)
            y = float(lm.y * h)
            pts.append([x, y])
        return np.array(pts, dtype=np.float64)
    
    def _build_face_mask(self, img_shape: tuple, face_points: np.ndarray) -> np.ndarray:
        """建立臉部凸包遮罩"""
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        if face_points.shape[0] == 0:
            return mask
        hull = cv2.convexHull(face_points.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        return mask
    
    def _estimate_midline(self, face_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """估計臉部中線"""
        # 使用特定的中線點
        midline_idx = (10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2)
        idx = np.array(midline_idx, dtype=int)
        idx = idx[(idx >= 0) & (idx < face_points.shape[0])]
        
        if idx.size == 0:
            ml_pts = face_points
        else:
            ml_pts = face_points[idx, :]
        
        # PCA 找主軸
        p0 = ml_pts.mean(axis=0)
        X = ml_pts - p0
        
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        u = Vt[0]
        u = u / (np.linalg.norm(u) + 1e-12)
        
        # 法向量
        n = np.array([-u[1], u[0]], dtype=np.float64)
        if n[0] < 0:
            n = -n
            
        return p0, n
    
    def _create_symmetric_face(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        d: np.ndarray,
        Xr: np.ndarray,
        Yr: np.ndarray,
        side: str
    ) -> np.ndarray:
        """建立對稱臉"""
        if side == 'left':
            region = (mask > 0) & (d < 0)
        else:
            region = (mask > 0) & (d > 0)
        
        # 建立半臉 alpha
        alpha = np.zeros_like(mask, dtype=np.uint8)
        alpha[region] = 255
        
        # 羽化邊緣
        if self.feather_px > 0:
            alpha = cv2.GaussianBlur(alpha, (self.feather_px*2+1, self.feather_px*2+1), 0)
        
        alpha = alpha.astype(np.float32) / 255.0
        
        # 反射半臉
        reflected = cv2.remap(image, Xr, Yr, cv2.INTER_LINEAR)
        reflected_alpha = cv2.remap(alpha, Xr, Yr, cv2.INTER_LINEAR)
        
        # 合成
        result = image * alpha[:,:,None] + reflected * reflected_alpha[:,:,None]
        final_alpha = np.clip(alpha + reflected_alpha, 0, 1)
        
        # 避免除零
        result = np.where(final_alpha[:,:,None] > 0.01, result / final_alpha[:,:,None], 0)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    # ========== Step 3: 直方圖均衡 ==========
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """應用 CLAHE 直方圖均衡（Lab 色彩空間）"""
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


# ========== 輔助函數 ==========
def batch_preprocess(
    input_dir: str,
    output_dir: str,
    steps: List[str] = None,
    **kwargs
) -> Dict[str, List[Path]]:
    """批次前處理的便捷函數
    
    Args:
        input_dir: 輸入目錄
        output_dir: 輸出目錄
        steps: 要執行的步驟
        **kwargs: 傳給 ImagePreprocessor 的參數
        
    Returns:
        處理結果路徑
    """
    with ImagePreprocessor(**kwargs) as processor:
        results = processor.process_folder(input_dir, output_dir, steps)
    
    logger.info(f"處理完成，共處理 {len(results)} 個資料夾")
    return results