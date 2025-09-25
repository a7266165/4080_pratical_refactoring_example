"""
以 MediaPipe Face Mesh 將臉部切成左右兩半後鏡射，
為每張輸入照片各輸出：
  1) 左半臉鏡射合成（_Lmirror）
  2) 右半臉鏡射合成（_Rmirror）

本版：**輸入/輸出/參數直接寫在程式碼頂端**，並以「FaceMesh 中線關鍵點」估計任意方向的臉部中線，
透過幾何反射於該直線上完成鏡射（不是單純以垂直影像中線）。

必要套件：
    pip install mediapipe opencv-python numpy

注意：此腳本預設每張圖處理單一人臉；若偵測不到臉則略過。
"""

from __future__ import annotations
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# ========================
# 可在此修改你的設定
# ========================
INPUT_DIR = r"C:\Users\4080\Desktop\_temp_save\Alz\asymmetry\demo_pic_temp\2_rotated"  # 輸入資料夾（放 10 張人臉照片）
OUTPUT_DIR = r'C:\Users\4080\Desktop\_temp_save\Alz\asymmetry\demo_pic_temp\3_mirrored'   # 輸出資料夾
DET_CONF = 0.2                         # 偵測信心門檻（0~1）

# ---- 邊界處理參數 ----
FEATHER_PX = 2   # 邊緣羽化像素，降低接縫/鋸齒
ERODE_PX   = 0   # 可選：先收縮半臉 alpha，避免邊緣滲色

def read_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"無法讀取影像：{path}")
    return img

def landmarks_to_xy(landmarks, img_shape: tuple[int, int]) -> np.ndarray:
    """將 FaceMesh 的相對座標轉為像素座標（N,2）。"""
    h, w = img_shape[:2]
    pts = []
    for lm in landmarks:
        x = float(lm.x * w)
        y = float(lm.y * h)
        pts.append([x, y])
    return np.array(pts, dtype=np.float64)

def build_face_mask(img_shape: tuple[int, int], face_points_xy: np.ndarray) -> np.ndarray:
    """以臉部關鍵點的凸包建立二值遮罩（uint8, 0/255）。"""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    if face_points_xy.shape[0] == 0:
        return mask
    hull = cv2.convexHull(face_points_xy.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    return mask

def estimate_midline_from_landmarks(
    face_points_xy: np.ndarray,
    midline_idx: tuple[int, ...] = (10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2)
) -> tuple[np.ndarray, np.ndarray]:
    """
    以中線關鍵點做 PCA，回傳:
      p0: 直線上一點（均值）
      n : 單位法向量（固定 n.x >= 0，確保 d>0 為影像右側）
    """
    # 過濾非法/越界索引，並去重
    idx = np.array(midline_idx, dtype=int)
    idx = idx[(idx >= 0) & (idx < face_points_xy.shape[0])]
    if idx.size == 0:
        # fallback: 沒有效中線點，改用整臉點估計
        ml_pts = face_points_xy
    else:
        idx = np.unique(idx)
        ml_pts = face_points_xy[idx, :]

    # 均值與 PCA
    p0 = ml_pts.mean(axis=0)
    X = ml_pts - p0
    if not np.isfinite(X).all() or np.allclose(X, 0):
        # 極端退化情境：用垂直中線近似
        xs = face_points_xy[:, 0]
        mid_x = 0.5 * (xs.min() + xs.max())
        p0 = np.array([mid_x, face_points_xy[:, 1].mean()], dtype=np.float64)
        n = np.array([1.0, 0.0], dtype=np.float64)
        return p0, n

    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    u = Vt[0]; u = u / (np.linalg.norm(u) + 1e-12)
    n = np.array([-u[1], u[0]], dtype=np.float64)
    if n[0] < 0:  # 固定方向：右側為正
        n = -n
    return p0, n

def make_half_alpha(mask: np.ndarray, d: np.ndarray, side: str,
                    feather_px: int = 2, erode_px: int = 0) -> np.ndarray:
    """回傳 [0,1] 的半臉 alpha（只在臉部凸包內），並做可選羽化/收縮。"""
    if side == 'left':
        region = (mask > 0) & (d < 0)
    elif side == 'right':
        region = (mask > 0) & (d > 0)
    else:
        raise ValueError("side 需為 'left' 或 'right'")
    a = np.zeros_like(mask, dtype=np.uint8)
    a[region] = 255
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px*2+1, erode_px*2+1))
        a = cv2.erode(a, k)
    if feather_px > 0:
        a = cv2.GaussianBlur(a, (feather_px*2+1, feather_px*2+1), 0)
    return (a.astype(np.float32) / 255.0)

def remap_premultiplied(img_bgr: np.ndarray, alpha: np.ndarray,
                        Xr: np.ndarray, Yr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """對 (RGB×alpha) 與 alpha 同步 remap，回傳反射後的 RGB 與 alpha。"""
    img_f = img_bgr.astype(np.float32) / 255.0
    a = alpha.astype(np.float32)  # [0,1]
    premul = img_f * a[..., None]             # (H,W,3)
    premul_ref = cv2.remap(premul, Xr, Yr, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    a_ref = cv2.remap(a, Xr, Yr, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    eps = 1e-6
    rgb_ref = np.where(a_ref[..., None] > eps, premul_ref / a_ref[..., None], 0)
    rgb_ref_u8 = np.clip(rgb_ref * 255.0, 0, 255).astype(np.uint8)
    return rgb_ref_u8, a_ref

def align_to_canvas_premul(
    img_bgr: np.ndarray,
    mask_u8: np.ndarray,
    p0: np.ndarray,
    n: np.ndarray,
    out_size: tuple[int, int] = (512, 512),  # (H, W)
    margin: float = 0.08,                    # 兩側留白比例（相對於畫布）
    fit: str = "contain",                    # 'contain' 或 'cover'
) -> np.ndarray:
    """
    旋正(以中線豎直) + 等比縮放 + 置中，輸出固定尺寸 BGRA（mask 外透明）。
    用預乘 alpha 避免黑邊。
    """
    H, W = out_size
    # 1) 準備旋轉：讓中線方向 u 對齊 y 軸
    u = np.array([n[1], -n[0]], dtype=np.float64)
    u /= (np.linalg.norm(u) + 1e-12)
    if u[1] < 0:
        u = -u
    angle = np.arctan2(u[0], u[1])  # 旋到 y 軸
    cos, sin = np.cos(angle), np.sin(angle)

    # 2) 旋轉矩陣（繞 p0）
    R = np.array([[cos, -sin, (1 - cos) * p0[0] + sin * p0[1]],
                  [sin,  cos, (1 - cos) * p0[1] - sin * p0[0]]], dtype=np.float32)

    # 先把 mask 旋到原圖大小，量測旋後 bbox
    m_rot = cv2.warpAffine(mask_u8, R, (img_bgr.shape[1], img_bgr.shape[0]),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ys, xs = np.where(m_rot > 0)
    if xs.size == 0 or ys.size == 0:
        # 沒有臉就回傳全透明畫布
        return np.zeros((H, W, 4), dtype=np.uint8)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    bw, bh = (x1 - x0 + 1), (y1 - y0 + 1)

    # 3) 設定畫布內可用區域（預留 margin）
    Wfit = int(round(W * (1 - 2 * margin)))
    Hfit = int(round(H * (1 - 2 * margin)))
    Wfit = max(Wfit, 1); Hfit = max(Hfit, 1)

    # 4) 等比縮放比例
    if fit == "cover":
        s = max(Wfit / max(bw, 1), Hfit / max(bh, 1))
    else:  # 'contain'
        s = min(Wfit / max(bw, 1), Hfit / max(bh, 1))

    # 5) 組合最終 2x3 仿射：S * R + T，將 bbox 中心對到畫布中心
    # 先把 R 擴成 3x3，再左乘縮放，再加上平移
    R3 = np.vstack([R, [0, 0, 1]]).astype(np.float32)
    S3 = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float32)

    # 旋後 bbox 的中心（在原圖座標系）
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

    # 將 (cx, cy, 1) 先經 R3，再經 S3，得到旋縮後中心的位置
    center_vec = np.array([cx, cy, 1.0], dtype=np.float32)
    center_rot = (S3 @ (R3 @ center_vec))
    # 目標中心（畫布中心）
    target = np.array([W / 2.0, H / 2.0, 1.0], dtype=np.float32)

    # 平移矩陣，使中心對齊
    tx, ty = (target - center_rot)[:2]
    T3 = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)

    M3 = T3 @ S3 @ R3  # 最終 3x3
    M = M3[:2, :]      # 取 2x3 給 warpAffine

    # 6) 預乘 alpha → warp → 還原
    img_f = img_bgr.astype(np.float32) / 255.0
    a = (mask_u8.astype(np.float32) / 255.0)
    premul = img_f * a[..., None]

    premul_w = cv2.warpAffine(premul, M, (W, H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    a_w = cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    eps = 1e-6
    rgb_w = np.where(a_w[..., None] > eps, premul_w / a_w[..., None], 0)
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(rgb_w * 255.0, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.clip(a_w * 255.0, 0, 255).astype(np.uint8)
    return rgba

def process_image(path: str, out_dir: Path, face_mesh: object):
    """處理單張影像：
    新流程（依你需求）：
      1) 先取出左/右半臉區域（僅限於臉部凸包內）。
      2) 以該半臉作為*唯一來源*去鏡射到另一側，生成對稱臉。
      3) 最後再只保留標點內（凸包）的區域，並輸出 PNG（透明背景）。
    """
    try:
        img_bgr = read_image_bgr(path)
    except Exception as e:
        return False, f"讀取失敗：{path} | {e}"

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return False, f"未偵測到臉部：{path}"

    # 取第一張臉
    face_landmarks = results.multi_face_landmarks[0].landmark
    pts_xy = landmarks_to_xy(face_landmarks, img_bgr.shape)
    mask = build_face_mask(img_bgr.shape, pts_xy)

    # 估計臉部中線（p0, n），並建立 signed distance d
    p0, n = estimate_midline_from_landmarks(pts_xy)
    # 計算反射座標與有號距離 d
    h, w = img_bgr.shape[:2]
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    d = (X - p0[0]) * n[0] + (Y - p0[1]) * n[1]
    Xr = X - 2.0 * d * n[0]
    Yr = Y - 2.0 * d * n[1]

    # 半臉 alpha（[0,1]，含羽化）
    alpha_left  = make_half_alpha(mask, d, side='left',  feather_px=FEATHER_PX, erode_px=ERODE_PX)
    alpha_right = make_half_alpha(mask, d, side='right', feather_px=FEATHER_PX, erode_px=ERODE_PX)

    # 以「預乘 alpha」反射半臉來源
    left_reflect_rgb,  left_reflect_a  = remap_premultiplied(img_bgr, alpha_left,  Xr, Yr)
    right_reflect_rgb, right_reflect_a = remap_premultiplied(img_bgr, alpha_right, Xr, Yr)

    img_f = img_bgr.astype(np.float32) / 255.0
    eps = 1e-6

    # ---- Left composite：左半臉為唯一來源（右側用左的鏡射）----
    premul_L = img_f * alpha_left[..., None] + (left_reflect_rgb.astype(np.float32)/255.0) * left_reflect_a[..., None]
    alpha_L  = np.clip(alpha_left + left_reflect_a, 0.0, 1.0)
    left_sym = np.where(alpha_L[..., None] > eps, premul_L / alpha_L[..., None], 0)
    left_sym = (np.clip(left_sym, 0, 1) * 255.0).astype(np.uint8)

    # ---- Right composite：右半臉為唯一來源（左側用右的鏡射）----
    premul_R = img_f * alpha_right[..., None] + (right_reflect_rgb.astype(np.float32)/255.0) * right_reflect_a[..., None]
    alpha_R  = np.clip(alpha_right + right_reflect_a, 0.0, 1.0)
    right_sym = np.where(alpha_R[..., None] > eps, premul_R / alpha_R[..., None], 0)
    right_sym = (np.clip(right_sym, 0, 1) * 255.0).astype(np.uint8)

    # 只保留「實際覆蓋」區域作為輸出遮罩（避免黑邊）
    mask_L = np.clip(alpha_L * 255.0, 0, 255).astype(np.uint8)
    mask_R = np.clip(alpha_R * 255.0, 0, 255).astype(np.uint8)

    # 輸出：透明背景 PNG + 緊湊裁切
    left_fixed  = align_to_canvas_premul(left_sym,  mask_L, p0, n, out_size=(512, 512), margin=0.08, fit="contain")
    right_fixed = align_to_canvas_premul(right_sym, mask_R, p0, n, out_size=(512, 512), margin=0.08, fit="contain")


    # ===== 正式輸出 =====
    p = Path(path); stem = p.stem

    out_left = out_dir / f"{stem}_Lmirror.png"
    out_right = out_dir / f"{stem}_Rmirror.png"

    cv2.imwrite(str(out_left), left_fixed)
    cv2.imwrite(str(out_right), right_fixed)

    return True, f"完成：{p.name} -> {out_left.name}, {out_right.name}"

def main():
    in_root = Path(INPUT_DIR)   # 這裡把 INPUT_DIR 視為「輸入母資料夾」
    out_root = Path(OUTPUT_DIR) # 這裡把 OUTPUT_DIR 視為「輸出母資料夾」
    out_root.mkdir(parents=True, exist_ok=True)

    if not in_root.exists():
        raise FileNotFoundError(f"找不到輸入資料夾：{in_root}")

    allowed = ('.jpg', '.jpeg', '.png', '.bmp')

    # 先建立「子資料夾 -> 檔案清單」的索引，並計算總張數
    dirs = [in_root] + sorted([p for p in in_root.rglob('*') if p.is_dir()])
    groups: dict[Path, list[Path]] = {}
    for d in dirs:
        files = sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in allowed])
        if files:
            groups[d] = files

    total_files = sum(len(fs) for fs in groups.values())
    if total_files == 0:
        raise FileNotFoundError(f"{in_root} 及其所有子資料夾中沒有相符影像（允許：.jpg/.jpeg/.png/.bmp）")

    print(f"共 {total_files} 張影像將被處理。輸出至：{out_root}")

    # 建立 FaceMesh（static_image_mode=True 適合單張影像）
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=DET_CONF,
    ) as face_mesh:
        ok_total = 0
        for subdir, files in groups.items():
            # 對應的輸出子資料夾（保持與輸入相同的相對路徑）
            rel = subdir.relative_to(in_root)
            out_dir = out_root / rel
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n處理資料夾：{subdir} → {out_dir}（{len(files)} 張）")
            ok_count = 0
            for fp in files:
                try:
                    ok, msg = process_image(str(fp), out_dir, face_mesh)
                except Exception as e:
                    ok, msg = False, f"處理失敗：{fp.name} | {e}"
                print(msg)
                if ok:
                    ok_count += 1
            ok_total += ok_count
            print(f"資料夾完成：成功 {ok_count}/{len(files)}")

    print(f"\n全部完成：成功 {ok_total}/{total_files}")


if __name__ == '__main__':
    main()

