import json
import os
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

def read_embedding_json(filepath: str) -> Dict:
    """讀取embedding JSON檔案"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# def calculate_embedding_difference(left_data: Dict, right_data: Dict) -> Dict:
#     """計算左右臉embedding的差值、平均值和相對差異"""
#     result = {
#         "source_images": {
#             "left": left_data.get("source_image", ""),
#             "right": right_data.get("source_image", "")
#         },
#         "image_names": {
#             "left": left_data.get("image_name", ""),
#             "right": right_data.get("image_name", "")
#         },
#         "embedding_differences": {},
#         "embedding_averages": {},  # 新增：平均向量
#         "relative_differences": {},  # 新增：相對差異
#         "embedding_dimensions": left_data.get("embedding_dimensions", {}),
#         "extraction_successful": {}
#     }
    
#     # 計算每個模型的向量差值、平均值和相對差異
#     for model_name in left_data.get("embeddings", {}).keys():
#         if model_name in right_data.get("embeddings", {}):
#             left_embedding = np.array(left_data["embeddings"][model_name])
#             right_embedding = np.array(right_data["embeddings"][model_name])
            
#             # 計算差值 (左 - 右)
#             diff = left_embedding - right_embedding
#             result["embedding_differences"][model_name] = diff.tolist()
            
#             # 計算平均值 (左 + 右) / 2
#             average = (left_embedding + right_embedding) / 2
#             result["embedding_averages"][model_name] = average.tolist()
            
#             # 計算相對差異 |左-右| / (|左| + |右|)
#             # 使用L2範數
#             left_norm = np.linalg.norm(left_embedding)
#             right_norm = np.linalg.norm(right_embedding)
#             diff_norm = np.linalg.norm(diff)
            
#             # 避免除以零
#             if (left_norm + right_norm) > 0:
#                 relative_diff = diff_norm / (left_norm + right_norm)
#             else:
#                 relative_diff = 0.0
            
#             result["relative_differences"][model_name] = float(relative_diff)
            
#             # 記錄兩邊是否都成功提取
#             left_success = left_data.get("extraction_successful", {}).get(model_name, False)
#             right_success = right_data.get("extraction_successful", {}).get(model_name, False)
#             result["extraction_successful"][model_name] = left_success and right_success
    
#     # 加入統計資訊
#     result["statistics"] = {}
#     for model_name, diff_values in result["embedding_differences"].items():
#         diff_array = np.array(diff_values)
#         avg_array = np.array(result["embedding_averages"][model_name])
        
#         result["statistics"][model_name] = {
#             "difference": {
#                 "mean": float(np.mean(diff_array)),
#                 "std": float(np.std(diff_array)),
#                 "min": float(np.min(diff_array)),
#                 "max": float(np.max(diff_array)),
#                 "l2_norm": float(np.linalg.norm(diff_array))
#             },
#             "average": {
#                 "mean": float(np.mean(avg_array)),
#                 "std": float(np.std(avg_array)),
#                 "min": float(np.min(avg_array)),
#                 "max": float(np.max(avg_array)),
#                 "l2_norm": float(np.linalg.norm(avg_array))
#             },
#             "relative_difference": result["relative_differences"][model_name],
#             "asymmetry_score": float(np.linalg.norm(diff_array) / np.linalg.norm(avg_array)) if np.linalg.norm(avg_array) > 0 else 0.0
#         }
    
#     # 新增：整體不對稱度評分（所有模型的平均相對差異）
#     if result["relative_differences"]:
#         result["overall_asymmetry"] = {
#             "mean_relative_difference": float(np.mean(list(result["relative_differences"].values()))),
#             "std_relative_difference": float(np.std(list(result["relative_differences"].values()))),
#             "max_relative_difference": float(np.max(list(result["relative_differences"].values()))),
#             "min_relative_difference": float(np.min(list(result["relative_differences"].values())))
#         }
    
#     return result

def calculate_embedding_difference(left_data: Dict, right_data: Dict) -> Dict:
    """計算左右臉embedding的差值、平均值和相對差異"""
    result = {
        "source_images": {
            "left": left_data.get("source_image", ""),
            "right": right_data.get("source_image", "")
        },
        "image_names": {
            "left": left_data.get("image_name", ""),
            "right": right_data.get("image_name", "")
        },
        "embedding_differences": {},
        "embedding_averages": {},  # 平均向量
        "relative_differences": {},  # 相對差異（逐元素）
        "embedding_dimensions": left_data.get("embedding_dimensions", {}),
        "extraction_successful": {}
    }
    
    # 計算每個模型的向量差值、平均值和相對差異
    for model_name in left_data.get("embeddings", {}).keys():
        if model_name in right_data.get("embeddings", {}):
            left_embedding = np.array(left_data["embeddings"][model_name])
            right_embedding = np.array(right_data["embeddings"][model_name])
            
            # 計算差值 (左 - 右)
            diff = left_embedding - right_embedding
            result["embedding_differences"][model_name] = diff.tolist()
            
            # 計算平均值 (左 + 右) / 2
            average = (left_embedding + right_embedding) / 2
            result["embedding_averages"][model_name] = average.tolist()
            
            # 計算相對差異（逐元素）：|左-右| / |左+右|
            # 每個元素單獨計算，結果仍為向量
            abs_diff = np.abs(left_embedding - right_embedding)
            abs_sum = np.abs(left_embedding + right_embedding)
            
            # 避免除以零，使用 np.where 處理
            # 當分母為0時，設定相對差異為0
            relative_diff = np.where(abs_sum > 0, abs_diff / abs_sum, 0.0)
            
            result["relative_differences"][model_name] = relative_diff.tolist()
            
            # 記錄兩邊是否都成功提取
            left_success = left_data.get("extraction_successful", {}).get(model_name, False)
            right_success = right_data.get("extraction_successful", {}).get(model_name, False)
            result["extraction_successful"][model_name] = left_success and right_success
    
    # 加入統計資訊
    result["statistics"] = {}
    for model_name, diff_values in result["embedding_differences"].items():
        diff_array = np.array(diff_values)
        avg_array = np.array(result["embedding_averages"][model_name])
        relative_diff_array = np.array(result["relative_differences"][model_name])
        
        result["statistics"][model_name] = {
            "difference": {
                "mean": float(np.mean(diff_array)),
                "std": float(np.std(diff_array)),
                "min": float(np.min(diff_array)),
                "max": float(np.max(diff_array)),
                "l2_norm": float(np.linalg.norm(diff_array))
            },
            "average": {
                "mean": float(np.mean(avg_array)),
                "std": float(np.std(avg_array)),
                "min": float(np.min(avg_array)),
                "max": float(np.max(avg_array)),
                "l2_norm": float(np.linalg.norm(avg_array))
            },
            "relative_difference": {
                "mean": float(np.mean(relative_diff_array)),
                "std": float(np.std(relative_diff_array)),
                "min": float(np.min(relative_diff_array)),
                "max": float(np.max(relative_diff_array)),
                "median": float(np.median(relative_diff_array))
            },
            "asymmetry_score": float(np.linalg.norm(diff_array) / np.linalg.norm(avg_array)) if np.linalg.norm(avg_array) > 0 else 0.0
        }
    
    # 整體不對稱度評分（基於所有模型的相對差異向量）
    all_relative_diffs = []
    for model_name in result["relative_differences"]:
        all_relative_diffs.extend(result["relative_differences"][model_name])
    
    if all_relative_diffs:
        all_relative_diffs = np.array(all_relative_diffs)
        result["overall_asymmetry"] = {
            "mean_relative_difference": float(np.mean(all_relative_diffs)),
            "std_relative_difference": float(np.std(all_relative_diffs)),
            "max_relative_difference": float(np.max(all_relative_diffs)),
            "min_relative_difference": float(np.min(all_relative_diffs)),
            "median_relative_difference": float(np.median(all_relative_diffs)),
            "percentile_75": float(np.percentile(all_relative_diffs, 75)),
            "percentile_90": float(np.percentile(all_relative_diffs, 90)),
            "percentile_95": float(np.percentile(all_relative_diffs, 95))
        }
    
    return result

def find_paired_files(folder_path: Path) -> List[Tuple[Path, Path]]:
    """找出資料夾中配對的左右臉檔案"""
    pairs = []
    
    # 取得所有JSON檔案
    json_files = list(folder_path.glob("*_embeddings.json"))
    
    # 建立檔名映射
    left_files = {}
    right_files = {}
    
    for file_path in json_files:
        filename = file_path.name
        if "_Lmirror_" in filename:
            # 提取基礎檔名（去除_Lmirror_claheL_embeddings.json）
            base_name = filename.replace("_Lmirror_claheL_embeddings.json", "")
            left_files[base_name] = file_path
        elif "_Rmirror_" in filename:
            # 提取基礎檔名（去除_Rmirror_claheL_embeddings.json）
            base_name = filename.replace("_Rmirror_claheL_embeddings.json", "")
            right_files[base_name] = file_path
    
    # 配對檔案
    for base_name in left_files:
        if base_name in right_files:
            pairs.append((left_files[base_name], right_files[base_name]))
    
    return pairs

def generate_summary_report(output_root: str):
    """生成總結報告，分析所有處理過的檔案"""
    output_path = Path(output_root)
    all_results = []
    
    print("\n生成總結報告...")
    
    # 收集所有差異檔案
    for subfolder in output_path.iterdir():
        if subfolder.is_dir():
            for json_file in subfolder.glob("*_LR_difference.json"):
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data["file_path"] = str(json_file)
                    data["folder"] = subfolder.name
                    all_results.append(data)
    
    if not all_results:
        print("沒有找到任何處理過的檔案")
        return
    
    # 統計分析
    summary = {
        "total_files": len(all_results),
        "by_model": {},
        "high_asymmetry_cases": [],
        "processing_time": None
    }
    
    # 按模型統計
    model_names = set()
    for result in all_results:
        if "relative_differences" in result:
            model_names.update(result["relative_differences"].keys())
    
    for model_name in model_names:
        relative_diffs = []
        for result in all_results:
            if model_name in result.get("relative_differences", {}):
                relative_diffs.append(result["relative_differences"][model_name])
        
        if relative_diffs:
            summary["by_model"][model_name] = {
                "mean_asymmetry": float(np.mean(relative_diffs)),
                "std_asymmetry": float(np.std(relative_diffs)),
                "max_asymmetry": float(np.max(relative_diffs)),
                "min_asymmetry": float(np.min(relative_diffs)),
                "median_asymmetry": float(np.median(relative_diffs))
            }
    
    # 找出高度不對稱的案例（相對差異 > 0.1）
    threshold = 0.1
    for result in all_results:
        if "overall_asymmetry" in result:
            if result["overall_asymmetry"]["mean_relative_difference"] > threshold:
                summary["high_asymmetry_cases"].append({
                    "file": result.get("image_names", {}).get("left", "").replace("_Lmirror_claheL", ""),
                    "folder": result["folder"],
                    "asymmetry_score": result["overall_asymmetry"]["mean_relative_difference"]
                })
    
    # 排序高度不對稱案例
    summary["high_asymmetry_cases"].sort(key=lambda x: x["asymmetry_score"], reverse=True)
    
    # 儲存總結報告
    summary_file = output_path / "asymmetry_summary_report.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"總結報告已儲存至: {summary_file}")
    print(f"\n分析了 {summary['total_files']} 個檔案")
    print(f"發現 {len(summary['high_asymmetry_cases'])} 個高度不對稱案例（相對差異 > {threshold}）")
    
    if summary["high_asymmetry_cases"][:5]:  # 顯示前5個最不對稱的案例
        print("\n前5個最不對稱的案例:")
        for i, case in enumerate(summary["high_asymmetry_cases"][:5], 1):
            print(f"  {i}. {case['folder']}/{case['file']}: {case['asymmetry_score']:.4f}")

def process_folder_structure(input_root: str, output_root: str):
    """處理整個資料夾結構"""
    input_path = Path(input_root)
    output_path = Path(output_root)
    
    # 統計資訊
    total_processed = 0
    errors = []
    
    print(f"開始處理資料夾: {input_root}")
    print(f"輸出至: {output_root}")
    print("-" * 50)
    
    # 處理每個子資料夾
    for subfolder in sorted(input_path.iterdir()):
        if subfolder.is_dir():
            print(f"\n處理子資料夾: {subfolder.name}")
            
            # 建立對應的輸出資料夾
            output_subfolder = output_path / subfolder.name
            output_subfolder.mkdir(parents=True, exist_ok=True)
            
            # 找出配對的檔案
            pairs = find_paired_files(subfolder)
            print(f"  找到 {len(pairs)} 對檔案")
            
            # 處理每對檔案
            for left_file, right_file in pairs:
                try:
                    # 讀取左右臉資料
                    left_data = read_embedding_json(left_file)
                    right_data = read_embedding_json(right_file)
                    
                    # 計算差值、平均值和相對差異
                    diff_result = calculate_embedding_difference(left_data, right_data)
                    
                    # 生成輸出檔名
                    base_name = left_file.name.replace("_Lmirror_claheL_embeddings.json", "")
                    output_filename = f"{base_name}_LR_difference.json"
                    output_filepath = output_subfolder / output_filename
                    
                    # 儲存結果
                    with open(output_filepath, 'w', encoding='utf-8') as f:
                        json.dump(diff_result, f, indent=2, ensure_ascii=False)
                    
                    # 顯示相對差異
                    if "overall_asymmetry" in diff_result:
                        asymmetry_score = diff_result["overall_asymmetry"]["mean_relative_difference"]
                        print(f"  ✓ 處理完成: {base_name} (不對稱度: {asymmetry_score:.4f})")
                    else:
                        print(f"  ✓ 處理完成: {base_name}")
                    
                    total_processed += 1
                    
                except Exception as e:
                    error_msg = f"處理 {left_file.name} 和 {right_file.name} 時發生錯誤: {str(e)}"
                    errors.append(error_msg)
                    print(f"  ✗ {error_msg}")
    
    # 輸出統計資訊
    print("\n" + "=" * 50)
    print(f"處理完成！")
    print(f"成功處理: {total_processed} 對檔案")
    if errors:
        print(f"錯誤數量: {len(errors)}")
        print("\n錯誤詳情:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("沒有錯誤發生")
    
    return total_processed, errors

def main():
    """主程式 - 使用基礎路徑方式"""
    # 基礎路徑
    base_input = r"C:\Users\4080\Desktop\python\Alz\Face\_analysis\asymmetry\data\features\datung\DeepLearning\4_pics_to_vector\datung"
    base_output = r"C:\Users\4080\Desktop\python\Alz\Face\_analysis\asymmetry\data\features\datung\DeepLearning\5_vector_to_feature_V2\datung"
    
    # 子資料夾列表
    subfolders = [
        "health/ACS",
        "health/NAD", 
        "patient"
    ]
    
    # 統計總體資訊
    overall_stats = {
        "total_processed": 0,
        "total_errors": 0,
        "processed_by_group": {}
    }
    
    # 處理每個子資料夾
    for i, subfolder in enumerate(subfolders, 1):
        print(f"\n{'='*60}")
        print(f"處理第 {i}/{len(subfolders)} 組資料: {subfolder}")
        print(f"{'='*60}")
        
        input_root = os.path.join(base_input, subfolder.replace("/", os.sep))
        output_root = os.path.join(base_output, subfolder.replace("/", os.sep))
        
        # 確認輸入資料夾存在
        if not os.path.exists(input_root):
            print(f"⚠️ 警告: 找不到輸入資料夾 '{input_root}'")
            print("  跳過此資料夾...")
            continue
        
        # 處理資料夾結構
        total_processed, errors = process_folder_structure(input_root, output_root)
        
        # 更新統計
        overall_stats["total_processed"] += total_processed
        overall_stats["total_errors"] += len(errors)
        overall_stats["processed_by_group"][subfolder] = {
            "processed": total_processed,
            "errors": len(errors)
        }
        
        if total_processed > 0:
            generate_summary_report(output_root)
            print(f"✅ {subfolder} 處理完成！")
    
    # 顯示總體統計
    print(f"\n📊 總共處理 {overall_stats['total_processed']} 對檔案")

if __name__ == "__main__":
    main()