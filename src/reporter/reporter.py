# src/reporter/reporter.py
"""結果報告生成器"""
import pandas as pd
from typing import Dict
from pathlib import Path
from datetime import datetime
import logging
from src.utils.utils import save_json

logger = logging.getLogger(__name__)


class Reporter:
    """報告生成器"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(self, results: Dict):
        """生成完整報告
        
        Args:
            results: 訓練結果字典，格式為:
                {
                    "config_key": {
                        "model": <model_object>,
                        "test_metrics": {...},
                        "train_metrics": {...},
                        "cv_mean": float,
                        "cv_std": float,
                        ...
                    },
                    ...
                }
        """
        logger.info("生成結果報告")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 儲存原始結果（JSON格式）
        self._save_json(results, timestamp)
        
        # 2. 生成摘要表格
        summary_df = self._create_summary_table(results)
        if not summary_df.empty:
            self._save_csv(summary_df, timestamp)
            
            # 3. 印出最佳結果
            self._print_best_results(summary_df)
            
            # 4. 生成統計報告
            self._print_statistics(summary_df)
        else:
            logger.warning("無有效結果可生成報告")
        
        logger.info(f"\n報告已儲存至: {self.output_dir}")
    
    def _save_json(self, results: Dict, timestamp: str):
        """儲存JSON格式結果"""
        filepath = self.output_dir / f"results_{timestamp}.json"
        
        # 移除無法序列化的 model 物件
        results_for_json = {}
        for key, value in results.items():
            if isinstance(value, dict):
                # 複製字典但排除 'model' 鍵
                results_for_json[key] = {k: v for k, v in value.items() if k != 'model'}
            else:
                results_for_json[key] = value
        
        # 使用 utils 的 save_json（已處理 numpy 類型）
        save_json(results_for_json, filepath)
        logger.info(f"  JSON結果: {filepath}")
    
    def _save_csv(self, df: pd.DataFrame, timestamp: str):
        """儲存CSV表格"""
        filepath = self.output_dir / f"summary_{timestamp}.csv"
        df.to_csv(filepath, index=False)
        logger.info(f"  CSV表格: {filepath}")
    
    def _create_summary_table(self, results: Dict) -> pd.DataFrame:
        """建立摘要表格"""
        rows = []
        
        for config_key, result in results.items():
            # 跳過無效的結果
            if not isinstance(result, dict) or 'test_metrics' not in result:
                continue
            
            # 解析配置名稱 (例如: "vggface_difference_cdr_0.5_random_forest")
            parts = config_key.split('_')
            
            # 基本解析
            embedding = parts[0] if len(parts) > 0 else ""
            feature = parts[1] if len(parts) > 1 else ""
            
            # 找出模型類型（通常在最後）
            model_type = result.get('model_type', '')
            if not model_type and len(parts) > 2:
                # 嘗試從 config_key 推測
                possible_models = ['random_forest', 'xgboost', 'svm', 'logistic']
                for pm in possible_models:
                    if pm in config_key:
                        model_type = pm
                        break
            
            # 建立資料列
            rows.append({
                'Config': config_key,
                'Embedding': embedding,
                'Feature': feature,
                'Model': model_type,
                'Accuracy': result['test_metrics'].get('accuracy', 0),
                'MCC': result['test_metrics'].get('mcc', 0),
                'F1': result['test_metrics'].get('f1', 0),
                'Sensitivity': result['test_metrics'].get('sensitivity', 0),
                'Specificity': result['test_metrics'].get('specificity', 0),
                'CV_Mean': result.get('cv_mean', 0),
                'CV_Std': result.get('cv_std', 0)
            })
        
        return pd.DataFrame(rows)
    
    def _print_best_results(self, df: pd.DataFrame):
        """印出最佳結果"""
        if df.empty:
            return
        
        print("\n" + "="*60)
        print("最佳結果")
        print("="*60)
        
        # 最佳準確率
        best_acc = None
        if 'Accuracy' in df.columns and not df['Accuracy'].isna().all():
            best_acc = df.loc[df['Accuracy'].idxmax()]
            print(f"\n最佳準確率:")
            print(f"  配置: {best_acc['Config']}")
            print(f"  Accuracy = {best_acc['Accuracy']:.3f}")
            print(f"  MCC = {best_acc['MCC']:.3f}")
            print(f"  F1 = {best_acc['F1']:.3f}")
            print(f"  Sensitivity = {best_acc['Sensitivity']:.3f}")
            print(f"  Specificity = {best_acc['Specificity']:.3f}")
        
        # 最佳MCC
        if 'MCC' in df.columns and not df['MCC'].isna().all():
            best_mcc = df.loc[df['MCC'].idxmax()]
            # 只在不同配置時顯示
            if best_acc is None or best_mcc['Config'] != best_acc['Config']:
                print(f"\n最佳MCC:")
                print(f"  配置: {best_mcc['Config']}")
                print(f"  MCC = {best_mcc['MCC']:.3f}")
                print(f"  Accuracy = {best_mcc['Accuracy']:.3f}")
                print(f"  F1 = {best_mcc['F1']:.3f}")
                print(f"  Sensitivity = {best_mcc['Sensitivity']:.3f}")
                print(f"  Specificity = {best_mcc['Specificity']:.3f}")
    
    def _print_statistics(self, df: pd.DataFrame):
        """印出統計資訊"""
        if df.empty:
            return
        
        print("\n" + "="*60)
        print("統計摘要")
        print("="*60)
        
        # 各嵌入模型表現
        if 'Embedding' in df.columns:
            print("\n各嵌入模型平均表現:")
            embedding_stats = df.groupby('Embedding')[['Accuracy', 'MCC', 'F1']].mean()
            if not embedding_stats.empty:
                print(embedding_stats.round(3))
        
        # 各特徵類型表現
        if 'Feature' in df.columns:
            print("\n各特徵類型平均表現:")
            feature_stats = df.groupby('Feature')[['Accuracy', 'MCC', 'F1']].mean()
            if not feature_stats.empty:
                print(feature_stats.round(3))
        
        # 各模型類型表現
        if 'Model' in df.columns:
            print("\n各模型類型平均表現:")
            model_stats = df.groupby('Model')[['Accuracy', 'MCC', 'F1']].mean()
            if not model_stats.empty:
                print(model_stats.round(3))
        
        # 整體統計
        print("\n整體統計:")
        print(f"  總配置數: {len(df)}")
        print(f"  平均準確率: {df['Accuracy'].mean():.3f} (±{df['Accuracy'].std():.3f})")
        print(f"  平均MCC: {df['MCC'].mean():.3f} (±{df['MCC'].std():.3f})")
        print(f"  平均F1: {df['F1'].mean():.3f} (±{df['F1'].std():.3f})")
        
        # 找出表現最好的前5個配置
        print("\n前5個最佳配置（按MCC排序）:")
        top5 = df.nlargest(5, 'MCC')[['Config', 'MCC', 'Accuracy', 'F1']]
        for idx, row in top5.iterrows():
            print(f"  {row['Config'][:50]}")
            print(f"    MCC={row['MCC']:.3f}, Acc={row['Accuracy']:.3f}, F1={row['F1']:.3f}")