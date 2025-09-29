# src/reporter/reporter.py (補充完整版)
"""結果報告生成器"""
import json
import pandas as pd
from typing import Dict
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class Reporter:
    """報告生成器"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate(self, results: Dict):
        """生成完整報告"""
        logger.info("生成結果報告")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 儲存原始結果
        self._save_json(results, timestamp)
        
        # 2. 生成摘要表格
        summary_df = self._create_summary_table(results)
        self._save_csv(summary_df, timestamp)
        
        # 3. 印出最佳結果
        self._print_best_results(summary_df)
        
        # 4. 生成統計報告
        self._print_statistics(summary_df)
        
        logger.info(f"\n報告已儲存至: {self.output_dir}")
        
    def _save_json(self, results: Dict, timestamp: str):
        """儲存JSON格式結果"""
        filepath = self.output_dir / f"results_{timestamp}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"  JSON結果: {filepath}")
        
    def _save_csv(self, df: pd.DataFrame, timestamp: str):
        """儲存CSV表格"""
        filepath = self.output_dir / f"summary_{timestamp}.csv"
        df.to_csv(filepath, index=False)
        logger.info(f"  CSV表格: {filepath}")
        
    def _create_summary_table(self, results: Dict) -> pd.DataFrame:
        """建立摘要表格"""
        rows = []
        
        for dataset_name, dataset_results in results.items():
            # 解析資料集名稱
            parts = dataset_name.split('_')
            embedding = parts[0] if len(parts) > 0 else ""
            feature = parts[1] if len(parts) > 1 else ""
            
            for cv_method, cv_results in dataset_results.items():
                for clf_name, metrics in cv_results.items():
                    if 'error' not in metrics:
                        rows.append({
                            'Dataset': dataset_name,
                            'Embedding': embedding,
                            'Feature': feature,
                            'CV_Method': cv_method,
                            'Classifier': clf_name,
                            'Accuracy': metrics.get('accuracy', 0),
                            'MCC': metrics.get('mcc', 0),
                            'Sensitivity': metrics.get('sensitivity', 0),
                            'Specificity': metrics.get('specificity', 0)
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
        if 'Accuracy' in df.columns and not df['Accuracy'].isna().all():
            best_acc = df.loc[df['Accuracy'].idxmax()]
            print(f"\n最佳準確率:")
            print(f"  {best_acc['Dataset']} / {best_acc['CV_Method']} / {best_acc['Classifier']}")
            print(f"  Accuracy = {best_acc['Accuracy']:.3f}")
            print(f"  MCC = {best_acc['MCC']:.3f}")
        
        # 最佳MCC
        if 'MCC' in df.columns and not df['MCC'].isna().all():
            best_mcc = df.loc[df['MCC'].idxmax()]
            print(f"\n最佳MCC:")
            print(f"  {best_mcc['Dataset']} / {best_mcc['CV_Method']} / {best_mcc['Classifier']}")
            print(f"  Accuracy = {best_mcc['Accuracy']:.3f}")
            print(f"  MCC = {best_mcc['MCC']:.3f}")
        
    def _print_statistics(self, df: pd.DataFrame):
        """印出統計資訊（補充缺少的方法）"""
        if df.empty:
            return
            
        print("\n" + "="*60)
        print("統計摘要")
        print("="*60)
        
        # 各嵌入模型表現
        if 'Embedding' in df.columns:
            print("\n各嵌入模型平均表現:")
            embedding_stats = df.groupby('Embedding')[['Accuracy', 'MCC']].mean()
            if not embedding_stats.empty:
                print(embedding_stats.round(3))
        
        # 各特徵類型表現
        if 'Feature' in df.columns:
            print("\n各特徵類型平均表現:")
            feature_stats = df.groupby('Feature')[['Accuracy', 'MCC']].mean()
            if not feature_stats.empty:
                print(feature_stats.round(3))
        
        # 各分類器表現
        if 'Classifier' in df.columns:
            print("\n各分類器平均表現:")
            clf_stats = df.groupby('Classifier')[['Accuracy', 'MCC']].mean()
            if not clf_stats.empty:
                print(clf_stats.round(3))
        
        # 各CV方法表現
        if 'CV_Method' in df.columns:
            print("\n各CV方法平均表現:")
            cv_stats = df.groupby('CV_Method')[['Accuracy', 'MCC']].mean()
            if not cv_stats.empty:
                print(cv_stats.round(3))