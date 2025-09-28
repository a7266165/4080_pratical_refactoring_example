# src/utils/id_parser.py
"""ID解析工具"""
import re
from typing import Tuple

def parse_subject_id(id_string: str) -> Tuple[str, int]:
    """解析受試者ID
    
    Args:
        id_string: ID字串，如 "P1-2", "ACS12-3"
        
    Returns:
        (base_id, visit_number) 如 ("P1", 2), ("ACS12", 3)
    """
    # 提取所有數字
    nums = re.findall(r"\d+", id_string)
    
    # 提取基礎ID（字母+第一個數字）
    match = re.match(r"^([A-Z]+\d+)", id_string.upper())
    if match:
        base_id = match.group(1)
    else:
        # 如果沒有匹配，使用'-'前的部分
        base_id = id_string.split('-')[0] if '-' in id_string else id_string
    
    # 提取訪視次數（最後一個數字，預設為1）
    visit = int(nums[-1]) if len(nums) >= 2 else 1
    
    return base_id, visit