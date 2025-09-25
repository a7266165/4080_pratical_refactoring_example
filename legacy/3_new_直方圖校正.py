#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批次處理：遍歷輸入母資料夾的所有子資料夾
每個子資料夾內的影像以 Lab 的 L 通道做 CLAHE，並輸出到輸出母資料夾下的同名子資料夾。
依賴：pip install opencv-python numpy
"""

from pathlib import Path
import numpy as np
import cv2

# ========= 直接在這裡改你的設定 =========
INPUT_ROOT  = r'C:\Users\4080\Desktop\python\Alz\Face\_analysis\asymmetry\data\_pics\2_mirrored'   # 輸入母資料夾（底下有多個子資料夾）
OUTPUT_ROOT = r'C:\Users\4080\Desktop\python\Alz\Face\_analysis\asymmetry\data\_pics\3_histogram_matched'  # 輸出母資料夾（會建立與輸入子資料夾相同名稱）
CLIP_LIMIT  = 2.0                                # CLAHE clipLimit：1.5~3.0 常見
TILES       = 8                                  # tileGridSize 的邊長：8~12 常見
SUFFIX      = "_claheL"                          # 輸出檔名後綴
SUBDIR_RECURSIVE = True                          # 是否遞迴處理子資料夾內更深層結構
INCLUDE_ROOT_IMAGES = False                      # 是否也處理直接放在 INPUT_ROOT 下的影像檔
# =====================================

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def imread_unicode(path: Path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, flags)

def imwrite_unicode(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"Encode failed: {path}")
    buf.tofile(str(path))

def ensure_uint8(img):
    if img is None:
        return None
    if img.dtype == np.uint8:
        return img
    img_min, img_max = float(np.min(img)), float(np.max(img))
    if img_max <= img_min:
        return np.zeros_like(img, dtype=np.uint8)
    norm = (img - img_min) / (img_max - img_min)
    return (norm * 255.0).astype(np.uint8)

def clahe_lab_l(img_bgr, clip=2.0, tiles=8):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

def iter_images(root: Path, recursive: bool):
    if recursive:
        yield from (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)
    else:
        yield from (p for p in root.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)

def process_one_subdir(in_subdir: Path, out_parent: Path) -> tuple[int, int]:
    """
    處理單一輸入子資料夾，將結果輸出到 out_parent / in_subdir.name
    回傳：(成功數, 總檔數)
    """
    out_subdir_root = out_parent / in_subdir.name
    paths = list(iter_images(in_subdir, SUBDIR_RECURSIVE))
    total = len(paths)
    if total == 0:
        print(f"[INFO] 子資料夾無影像：{in_subdir}")
        return 0, 0

    ok = 0
    for p in paths:
        img = imread_unicode(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] 無法讀取：{p}")
            continue
        img = ensure_uint8(img)
        try:
            out = clahe_lab_l(img, clip=CLIP_LIMIT, tiles=TILES)
        except Exception as e:
            print(f"[ERROR] 處理失敗：{p} -> {e}")
            continue

        # 以目前子資料夾為相對根，保留其內部結構
        rel = p.relative_to(in_subdir)
        out_path = (out_subdir_root / rel).with_name(p.stem + SUFFIX + p.suffix)
        try:
            imwrite_unicode(out_path, out)
            ok += 1
        except Exception as e:
            print(f"[ERROR] 儲存失敗：{out_path} -> {e}")

    print(f"[DONE] 子資料夾完成：{in_subdir.name}  -> {ok}/{total}")
    return ok, total

def run():
    in_root, out_root = Path(INPUT_ROOT), Path(OUTPUT_ROOT)
    if not in_root.exists() or not in_root.is_dir():
        raise SystemExit(f"輸入母資料夾不存在：{in_root}")

    # 1) 先處理直接位於母資料夾下的影像（可選）
    total_ok = total_cnt = 0
    if INCLUDE_ROOT_IMAGES:
        root_imgs = list(iter_images(in_root, recursive=False))
        if root_imgs:
            print(f"[INFO] 處理母資料夾直屬影像 {len(root_imgs)} 張")
        for p in root_imgs:
            img = imread_unicode(p, cv2.IMREAD_COLOR)
            if img is None:
                print(f"[WARN] 無法讀取：{p}")
                continue
            img = ensure_uint8(img)
            try:
                out = clahe_lab_l(img, clip=CLIP_LIMIT, tiles=TILES)
            except Exception as e:
                print(f"[ERROR] 處理失敗：{p} -> {e}")
                continue

            # 直接輸出到 OUTPUT_ROOT，同層級，不再建立子資料夾
            out_path = (out_root / (p.stem + SUFFIX + p.suffix))
            try:
                imwrite_unicode(out_path, out)
                total_ok += 1
            except Exception as e:
                print(f"[ERROR] 儲存失敗：{out_path} -> {e}")
            total_cnt += 1

    # 2) 遍歷所有子資料夾（只看第一層名稱相同；其內部可再遞迴）
    subdirs = sorted([d for d in in_root.iterdir() if d.is_dir()])
    if not subdirs and not INCLUDE_ROOT_IMAGES:
        raise SystemExit("在輸入母資料夾下找不到任何子資料夾。")

    for d in subdirs:
        ok, cnt = process_one_subdir(d, out_root)
        total_ok += ok
        total_cnt += cnt

    print(f"\n總結：成功 {total_ok}/{total_cnt} 張。輸出根目錄：{out_root.resolve()}")

if __name__ == "__main__":
    run()
