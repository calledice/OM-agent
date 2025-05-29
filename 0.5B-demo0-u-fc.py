import json 
import re
import os 
import time
from datetime import datetime 
from typing import Dict, Any, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from base import save_to_json, process_diagnosis, extract_date_from_query
import sys 

"""
如果有qwen_agent包，那么可以LLM直接调用封装好的function
但是由于没有这个包，目前只能LLM用作是否执行function
"""


class SystemRules:
    """系统诊断规则定义"""
    
    @staticmethod 
    def get_status_rules():
        """状态码生成规则"""
        return {
            "CPU": [
                {"max": 1, "code": 0, "desc": "≤1 (1分钟负载/12核心)"},
                {"max": 3, "code": 1, "desc": "≤3"},
                {"max": 8, "code": 2, "desc": "≤8"},
                {"code": 3, "desc": ">8"}
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

class SystemDiagnosticTool:
    """系统诊断工具类，负责日志文件的读取和处理"""
    
    @staticmethod 
    def read_log_file(date_str: str) -> Dict[str, Any]:
        """
        读取指定日期的系统日志文件 
        
        参数:
            date_str: 日期字符串，格式为YYYYMMDD 
            
        返回:
            包含日志数据和状态的字典 
        """
        log_path = f"D:\python_code\OM-agent\log_file/{date_str}.json"
        print("正在读取日志文件:", log_path)
        # 检查文件是否存在 
        if not os.path.exists(log_path):  
            return {
                "status": "error",
                "message": f"日志文件 {log_path} 不存在",
                "data": None 
            }
        
        # 读取并解析JSON文件 
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                log_data = json.load(f)  
            return {
                "status": "success",
                "message": "日志读取成功",
                "data": log_data 
            }
        except json.JSONDecodeError:
            return {
                "status": "error",
                "message": f"日志文件 {log_path} 不是有效的JSON格式",
                "data": None 
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"读取日志时发生错误: {str(e)}",
                "data": None
                }

def diagnose_system_transformers(model, tokenizer, user_query: str, log_data: Dict[str, Any]) -> str:
    """
    使用Transformers模型分析日志数据（基于规则引擎）
    流程：
    1. 使用LLM提取关键指标 
    2. 使用规则引擎生成状态码和措施码 
    """
    # 1. 使用LLM提取关键指标
    extraction_prompt = f"""
    请从以下JSON数据中提取系统监控指标：
    {json.dumps(log_data,  indent=2)}
    
    需要提取的指标：
    - cpu_1min: CPU 1分钟负载
    - memory_usage: 内存使用率
    - swap_usage: Swap使用率
    - root_disk_usage: 根分区使用率
    - data_disk_usage: 数据分区使用率
    - buffers: 缓冲区占比
    
    请以严格的JSON格式返回提取结果，只包含数据，不要有任何解释或额外文本。
    """

    metrics = {"cpu_1min": float(log_data["system_log"]["system"]["uptime"]["load_average"]), 
            "memory_usage": int(log_data["system_log"]["hardware"]["memory"]["usage"].strip('%')), 
            "swap_usage":  int(log_data["system_log"]["hardware"]["memory"]["swap"]["usage"].strip('%')),
            "root_disk_usage": int(log_data["system_log"]["hardware"]["storage"][0]["usage"].strip('%')), 
            "data_disk_usage": int(log_data["system_log"]["hardware"]["storage"][1]["usage"].strip('%'))}
    print(f"提取的指标: {metrics}")
    # 3. 使用规则引擎生成状态码和措施码 
    engine = RuleEngine()
    try:
        status_code = engine.generate_status_code(metrics) 
        action_code = engine.generate_action_code(metrics,  status_code)
        return f"{status_code} | {action_code}"
    except Exception as e:
        print(f"规则引擎执行失败: {str(e)}")
        return "0-0-0-0-0 | 0-0-0"  # 默认安全值 

        
def main(user_input):
    # 加载 Qwen 模型 
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name) 
    model = AutoModelForCausalLM.from_pretrained(model_name,  device_map="auto")
    
    # 定义诊断函数 
    def diagnostics(user_query: str) -> str:
        """检查系统在指定日期的状况"""
        print(f"正在检查系统状况...")
        # 1. 提取日期 
        date_str = extract_date_from_query(user_query)
        if not date_str:
            print("错误: 无法从查询中提取有效日期")
            return "000000"
        # 2. 读取日志文件 
        log_result = SystemDiagnosticTool.read_log_file(date_str)  
        if log_result["status"] != "success":
            print(f"错误: {log_result['message']}")
            return "000000"
        # 3. 使用LLM进行诊断 
        print("\n正在分析系统日志...")
        diagnosis = diagnose_system_transformers(model, tokenizer, user_query, log_result["data"])
        # 4. 输出结果 
        print("\n系统诊断报告:")
        print(diagnosis)

        temp = process_diagnosis(diagnosis)
        save_to_json(temp) 
 
        return temp["decimal_result"]

    
    # 构造提示词 
    prompt = f"""
    你是一个智能助手，可以检查系统状况。以下是可用的功能：
    - diagnostics: 检查系统在指定日期的状况 
    
    用户输入：{user_input}
    请判断是否需要检查系统状况，如果需要，只回答需要检查或需要用到diagnostics来检查系统。
    """
    
    # 生成回复 
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device) 
    outputs = model.generate(**inputs,  max_new_tokens=200)
    response = tokenizer.decode(outputs[0],  skip_special_tokens=True)
    
    # 检查是否需要调用诊断功能 
    if "diagnostics" in response.lower()  or "检查" in response.lower(): 
        # 直接调用诊断函数 
        func_response = diagnostics(user_input)
        print(f"系统状态代码: {func_response}")
    else:
        print(response)
 
if __name__ == "__main__": 
    start_time = time.time()  
 
    # # 检查命令行参数 
    # if len(sys.argv)  > 1: 
    #     try: 
    #         xxx = int(sys.argv[1])  
    #         if xxx == 0: 
    #             user_input = "帮我看看2025年5月21日系统的状况。" 
    #         elif xxx == 1: 
    #             user_input = "帮我看看2025年5月20日系统的状况。" 
    #         else: 
    #             user_input = "你好" 
    #     except ValueError: 
    #         print("输入的参数不是有效的整数，请输入一个整数作为参数。") 
    #         sys.exit(1)  
    # else: 
    #     print("请在运行脚本时提供一个整数参数。") 
    #     sys.exit(1)  
    user_input = "帮我看看2025年5月20日系统的状况。" 
    main(user_input) 
 
    elapsed_time = (time.time()  - start_time) / 60 
    print(f"代码运行时间: {elapsed_time:.2f}min") 