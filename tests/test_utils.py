# tests/test_utils.py
"""測試工具函數"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.id_parser import parse_subject_id

def test_id_parsing():
    """測試ID解析"""
    assert parse_subject_id("P15-2") == ("P15", 2)
    assert parse_subject_id("ACS1") == ("ACS1", 1)
    assert parse_subject_id("NAD12-3") == ("NAD12", 3)
    print("✓ ID解析測試通過")

if __name__ == '__main__':
    test_id_parsing()
    print("✓ 工具函數測試完成")