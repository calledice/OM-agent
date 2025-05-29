from datetime import datetime
import json
import re
from typing import Optional 

def save_to_json(data, filename="calc_result.json"): 
    """将结果保存到JSON文件（嵌套在OM_result键下）"""
    # 将原始数据嵌套在OM_result下 
    wrapped_data = {
            "result": data["binary_result"]
        }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(wrapped_data,  f, ensure_ascii=False, indent=4)


def process_diagnosis(diagnosis_text):
    """ 
    处理诊断文本并生成二进制结果 
    
    参数:
        diagnosis_text (str): 诊断文本内容 
        
    返回:
        dict: 包含原始结果和二进制结果 
    """

    
    result_str = diagnosis_text
    
    # 分割状态码和措施码 
    # status_part, action_part = result_str.split("|")
    
    # 合并所有数字 
    numbers = re.findall(r'\d',  result_str)
    all_digits = list(map(int, numbers))
    
    # 转换为2位二进制 
    binary_parts = [f"{digit:02b}" for digit in all_digits]
    binary_str = ''.join(binary_parts)
    decimal_str = ''.join(numbers)
    
    return {
        "decimal_result": decimal_str,
        "binary_parts": binary_parts,
        "binary_result": binary_str 
    }   

def extract_date_from_query(query: str) -> Optional[str]:
    """
    从用户查询中提取日期并格式化为YYYYMMDD 
    
    参数:
        query: 用户输入的查询文本 
        
    返回:
        格式化后的日期字符串或None(如果未找到日期)
    """
    # 支持的日期格式 
    date_patterns = [
        (r'(\d{4})年(\d{1,2})月(\d{1,2})日', "%Y年%m月%d日"),  # 2025年5月20日 
        (r'(\d{4})-(\d{1,2})-(\d{1,2})', "%Y-%m-%d"),       # 2025-05-20 
        (r'(\d{4})/(\d{1,2})/(\d{1,2})', "%Y/%m/%d")        # 2025/05/20 
    ]
    
    for pattern, fmt in date_patterns:
        match = re.search(pattern,  query)
        if match:
            try:
                date_str = match.group()  
                dt = datetime.strptime(date_str,  fmt)
                return dt.strftime("%Y%m%d")    # 统一转为YYYYMMDD格式 
            except ValueError:
                continue 
    return None