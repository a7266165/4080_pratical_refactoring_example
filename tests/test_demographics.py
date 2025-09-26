# tests/test_demographics.py
"""測試人口學資料處理"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from src.data.demographics import DemographicsProcessor
from src.data.balancing import DataBalancer, BalancingConfig

def test_demographics_loading():
    """測試人口學資料載入"""
    print("\n" + "="*60)
    print("測試人口學資料載入")
    print("="*60)
    
    # 建立測試資料
    test_data = {
        "P": pd.DataFrame({
            "ID": ["P1-1", "P2-1", "P2-2", "P3-1"],
            "Age": [65, 70, 71, 68],
            "Sex": ["M", "F", "F", "M"],
            "Global_CDR": [0.5, 1.0, 1.5, 0.5]
        }),
        "ACS": pd.DataFrame({
            "ID": ["ACS1-1", "ACS2-1", "ACS3-1"],
            "Age": [66, 69, 72],
            "Sex": ["M", "F", "M"]
        }),
        "NAD": pd.DataFrame({
            "ID": ["NAD1-1", "NAD2-1"],
            "Age": [64, 70],
            "Sex": ["F", "M"]
        })
    }
    
    processor = DemographicsProcessor()
    
    # 模擬載入資料
    processor.tables = test_data
    
    # 測試最新訪視選擇
    print("\n測試多次訪視處理:")
    filtered_p = processor._select_latest_visits(test_data["P"])
    print(f"  原始P組: {len(test_data['P'])} 筆")
    print(f"  處理後P組: {len(filtered_p)} 筆")
    assert "P2-2" in filtered_p["ID"].values, "應保留P2的第二次訪視"
    assert "P2-1" not in filtered_p["ID"].values, "應移除P2的第一次訪視"
    print("  ✓ 多次訪視處理正確")
    
    # 測試CDR篩選
    print("\n測試CDR篩選:")
    filtered_cdr = processor.filter_by_cdr(0.75, "P")
    print(f"  CDR > 0.75: {len(filtered_cdr)} 筆")
    assert len(filtered_cdr) == 2, "應該有2筆CDR > 0.75"
    print("  ✓ CDR篩選正確")
    
    # 測試查詢表建立
    print("\n測試查詢表建立:")
    lookup = processor.build_lookup_table()
    print(f"  查詢表大小: {len(lookup)} 筆")
    assert "P1-1" in lookup or "P1" in lookup, "查詢表應包含P1"
    print("  ✓ 查詢表建立成功")
    
    print("\n✓ 人口學資料處理測試完成")

def test_data_balancing():
    """測試資料平衡"""
    print("\n" + "="*60)
    print("測試資料平衡策略")
    print("="*60)
    
    # 建立測試資料（更多樣本以測試平衡）
    np.random.seed(42)
    n_p = 30
    n_acs = 40
    n_nad = 35
    
    test_data = {
        "P": pd.DataFrame({
            "ID": [f"P{i}-1" for i in range(n_p)],
            "Age": np.random.normal(70, 10, n_p),
            "Global_CDR": np.random.uniform(0, 3, n_p)
        }),
        "ACS": pd.DataFrame({
            "ID": [f"ACS{i}-1" for i in range(n_acs)],
            "Age": np.random.normal(68, 10, n_acs)
        }),
        "NAD": pd.DataFrame({
            "ID": [f"NAD{i}-1" for i in range(n_nad)],
            "Age": np.random.normal(69, 10, n_nad)
        })
    }
    
    processor = DemographicsProcessor()
    processor.tables = test_data
    
    # 測試年齡配對
    print("\n測試年齡配對:")
    config = BalancingConfig(
        enable_age_matching=True,
        enable_cdr_filter=False,
        n_bins=5
    )
    balancer = DataBalancer(processor, config)
    allowed_ids, summary = balancer.balance_groups()
    
    print(f"  配對前: P={n_p}, ACS={n_acs}, NAD={n_nad}")
    print(f"  配對後: P={len(allowed_ids['P'])}, "
          f"ACS={len(allowed_ids['ACS'])}, NAD={len(allowed_ids['NAD'])}")
    
    # 檢查年齡分布
    if not summary.empty:
        health_age = summary[summary["group"] == "Health"]["age_mean"].values[0]
        p_age = summary[summary["group"] == "P"]["age_mean"].values[0]
        age_diff = abs(health_age - p_age)
        print(f"  年齡差異: {age_diff:.1f}歲")
        assert age_diff < 3, "配對後年齡差異應該很小"
    
    print("  ✓ 年齡配對成功")
    
    # 測試CDR篩選+年齡配對
    print("\n測試CDR篩選+年齡配對:")
    config = BalancingConfig(
        enable_age_matching=True,
        enable_cdr_filter=True,
        cdr_threshold=1.0
    )
    balancer = DataBalancer(processor, config)
    allowed_ids, summary = balancer.balance_groups()
    
    print(f"  CDR>1.0篩選後配對結果:")
    print(f"    P={len(allowed_ids['P'])}")
    print("  ✓ CDR篩選+配對成功")
    
    print("\n✓ 資料平衡測試完成")

def main():
    """執行所有測試"""
    print("\n" + "#"*60)
    print("# 人口學資料與平衡策略測試")
    print("#"*60)
    
    test_demographics_loading()
    test_data_balancing()
    
    print("\n" + "#"*60)
    print("# 所有測試完成！")
    print("#"*60)

if __name__ == '__main__':
    main()