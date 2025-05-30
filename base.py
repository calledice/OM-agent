from datetime import datetime
import json
import re
from typing import Optional 
from typing import Dict

def save_to_json(data,xxx="00", filename="calc_result.json"): 
    """将结果保存到JSON文件（嵌套在OM_result键下）"""
    aa = ("10"+xxx+"000000000000"+data["binary_result"])
    print("返回二进制：" + aa)
    bb = int(aa, 2)
    print("返回十进制：" , bb)
    # 将原始数据嵌套在OM_result下 
    wrapped_data = {
            "result": bb
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


class SystemRules:
    """系统诊断规则定义"""
    
    @staticmethod 
    def get_status_rules():
        """状态码生成规则"""
        return {
            "CPU": [
                {"max": 1, "code": 0, "desc": "≤1 (1分钟负载/4核心)"},
                {"max": 2, "code": 1, "desc": "≤2"},
                {"max": 3.5, "code": 2, "desc": "≤3.5"},
                {"code": 3, "desc": ">3.5"}
            ],
            "Memory": [
                {"max": 70, "code": 0, "desc": "≤70%"},
                {"max": 90, "code": 1, "desc": "70-90%"},
                {"max": 99, "code": 2, "desc": "90-99%"},
                {"code": 3, "desc": "OOM/≥99%"}
            ],
            "Swap": [
                {"max": 70, "code": 0, "desc": "≤70%"},
                {"max": 90, "code": 1, "desc": "70-90%"},
                {"code": 2, "desc": "≥90%"}
            ],
            "RootDisk": [
                {"max": 70, "code": 0, "desc": "≤70%"},
                {"max": 90, "code": 1, "desc": "70-90%"},
                {"code": 2, "desc": "≥90%"}
                
            ],
            "DataDisk": [
                {"max": 70, "code": 0, "desc": "≤70%"},
                {"max": 90, "code": 1, "desc": "70-90%"},
                {"max": 99, "code": 2, "desc": "90-99%"},
                {"code": 3, "desc": "≥99%"}
            ]
        }
    
    @staticmethod 
    def get_action_rules():
        """措施码生成规则"""
        return {
        "CPU": [
            {"conditions": [("CPU", "==", 0)], "code": 0, "desc": "无操作"},
            {"conditions": [("CPU", "==", 1)], "code": 1, "desc": "限制非核心进程"},
            {"conditions": [("CPU", "==", 2)], "code": 2, "desc": "强制降频"},
            {"conditions": [("CPU", "==", 3)], "code": 3, "desc": "重启模块"}  # 移除service_available条件 
        ],
        "Memory": [
            {"conditions": [("Memory", "==", 0)], "code": 0, "desc": "无操作"},
            {"conditions": [("Memory", "==", 1)], "code": 1, "desc": "释放缓存"},
            {"conditions": [("Memory", "==", 2)], "code": 2, "desc": "终止TOP3进程"},
            {"conditions": [("Memory", "==", 3)], "code": 3, "desc": "紧急内存压缩并告警"}
        ],
        "Storage": [
            {"conditions": [("RootDisk", "==", 0), ("DataDisk", "==", 0)], "code": 0, "desc": "无操作"},
            {"conditions": [("RootDisk", "==", 1)], "code": 1, "desc": "系统日志清理"},  # 单独处理 
            {"conditions": [("DataDisk", "==", 1)], "code": 1, "desc": "业务日志清理"},  # 单独处理 
            {"conditions": [("RootDisk", ">=", 2)], "code": 2, "desc": "停止非核心服务"},
            {"conditions": [("DataDisk", "==", 2)], "code": 2, "desc": "停止采集"},
            {"conditions": [("DataDisk", "==", 3)], "code": 3, "desc": "只读模式并告警"}
        ]
    }
class RuleEngine:
    """规则自动匹配引擎"""
    
    def __init__(self):
        self.status_rules  = SystemRules.get_status_rules() 
        self.action_rules  = SystemRules.get_action_rules() 
    
    def generate_status_code(self, metrics: dict) -> str:
        """生成状态码"""
        codes = []
        
        # CPU状态码 
        cpu_load = metrics.get("cpu_1min",  0)
        codes.append(self._match_rule("CPU",  cpu_load))
        
        # 内存状态码 
        mem_usage = metrics.get("memory_usage",  0)
        codes.append(self._match_rule("Memory",  mem_usage))
        
        # Swap状态码 
        swap_usage = metrics.get("swap_usage",  0)
        codes.append(self._match_rule("Swap",  swap_usage))
        
        # 根分区状态码 
        root_disk = metrics.get("root_disk_usage",  0)
        codes.append(self._match_rule("RootDisk",  root_disk))
        
        # 数据分区状态码 
        data_disk = metrics.get("data_disk_usage",  0)
        codes.append(self._match_rule("DataDisk",  data_disk))
        
        return "-".join(map(str, codes))
    
    def generate_action_code(self, metrics: dict, status_code: str) -> str:
        """生成措施码"""
        # 解析状态码 
        status_parts = list(map(int, status_code.split("-"))) 
        status_map = {
            "CPU": status_parts[0],
            "Memory": status_parts[1],
            "Swap": status_parts[2],
            "RootDisk": status_parts[3],
            "DataDisk": status_parts[4]
        }
        
        # 添加额外指标 
        context = {**status_map, **metrics}
        
        # 生成各部分的措施码 
        cpu_action = self._match_action("CPU", context)
        mem_action = self._match_action("Memory", context)
        storage_action = self._match_action("Storage", context)
        
        return f"{cpu_action}-{mem_action}-{storage_action}"
    
    def _match_rule(self, category: str, value: float) -> int:
        """匹配单个状态码规则"""
        for rule in self.status_rules[category]: 
            if "max" in rule and value > rule["max"]:
                continue 
            return rule["code"]
        return 0  # 默认安全值 
    
    def _match_action(self, category: str, context: dict) -> int:
        """匹配单个措施码规则"""
        for rule in self.action_rules[category]: 
            if all(self._eval_condition(cond, context) for cond in rule["conditions"]):
                return rule["code"]
        return 0  # 默认安全值 
    
    def _eval_condition(self, condition: tuple, context: dict) -> bool:
        """评估单个条件"""
        key, op, target = condition 
        value = context.get(key,  None)
        
        if value is None:
            return False 
            
        if op == ">":
            return value > target 
        elif op == ">=":
            return value >= target 
        elif op == "<":
            return value < target 
        elif op == "<=":
            return value <= target 
        elif op == "==":
            return value == target 
        elif op == "!=":
            return value != target 
        
        return False 
    

def build_system_prompt(metrics: Dict[str, float]) -> str:
    return f"""你是一个系统诊断助手，请严格按规则生成状态码和措施码：
 
        【当前指标】
        - CPU: {metrics['cpu']:.1f}
        - Mem: {metrics['mem']:.1f}%
        - Swap: {metrics['swap']:.1f}%
        - Root: {metrics['root_disk']:.1f}%
        - Data: {metrics['data_disk']:.1f}%
        
        【状态码规则】
        A: 3(CPU>350), 2(CPU>300), 1(CPU>100), 0(CPU≤100) 
        B: 3(OOM/Mem≥99), 2(Mem≥90), 1(Mem≥70), 0(Mem≤70)
        C: 2(Swap≥90), 1(Swap≥70), 0(Swap<70)
        D: 2(Root≥90), 1(Root≥70), 0(Root<70)
        E: 3(Data≥99), 2(Data≥90), 1(Data≥70), 0(Data≤70)

        【措施码规则】
        X:0(A=0),1(A=1),2(A=2),3(A=3)  
        Y:0(B=0),1(B=1),2(B=2),3(B=3)  
        Z:0(D=0+E=0),1(D≥1|E≥1),2(D≥2|E=2),3(E=3)
        
        【输出要求】
        1. 仅输出一行，格式为：A-B-C-D-E | X-Y-Z
        2. 示例 1-2-0-3-1 | 1-2-2
        3. 禁止任何解释！"""