import json 
import re
import os 
import time
from datetime import datetime 
from typing import Dict, Any, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import sys 
from llama_cpp_inference import llama_cpp_inference

"""
如果有qwen_agent包，那么可以LLM直接调用封装好的function
但是由于没有这个包，目前只能LLM用作是否执行function
"""
from datetime import datetime
import json
import re
from typing import Optional 
from typing import Dict

import psutil 
def get_system_log():
    # 1. 获取系统启动时间（计算 uptime）
    boot_time = datetime.fromtimestamp(psutil.boot_time()) 
    uptime = datetime.now()  - boot_time 
    uptime_days = uptime.days  
    uptime_hours, remainder = divmod(uptime.seconds,  3600)
    uptime_minutes, _ = divmod(remainder, 60)
 
    # 2. 获取内存和交换分区信息 
    mem = psutil.virtual_memory() 
    swap = psutil.swap_memory() 
 
    # 3. 获取磁盘分区信息 
    disks = []
    for partition in psutil.disk_partitions(): 
        usage = psutil.disk_usage(partition.mountpoint) 
        disks.append({ 
            "device": partition.device, 
            "mount": partition.mountpoint, 
            "size": f"{usage.total  / (1024**3):.1f} GiB",
            "usage": f"{usage.percent}%" 
        })
 
    # 4. 构建最终的 JSON 结构 
    log_data = {
        "system_log": {
            "timestamp": datetime.now().astimezone().replace(microsecond=0).isoformat(), 
            "system": {
                "uptime": {
                    "days": uptime_days,
                    "hours": uptime_hours,
                    "minutes": uptime_minutes,
                    "load_average": [x / psutil.cpu_count()  for x in psutil.getloadavg()]   # 标准化负载 
                }
            },
            "hardware": {
                "memory": {
                    "total": f"{mem.total  / (1024**3):.1f} GiB",
                    "usage": f"{mem.percent}%", 
                    "buffers": f"{mem.buffers  / (1024**3):.1f} GiB",
                    "swap": {
                        "total": f"{swap.total  / (1024**3):.1f} GiB",
                        "usage": f"{swap.percent}%" 
                    }
                },
                "storage": disks 
            }
        }
    }
    return log_data 
def save_to_json(data,xxx="00", filename="calc_result.json"): 
    """将结果保存到JSON文件（嵌套在OM_result键下）"""
    aa = ("10"+xxx+"000000000000"+data["binary_result"])
    bb = int(aa, 2)
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
        log_path = f"./log_file/{date_str}.json"
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

def diagnose_system_transformers(user_query: str, log_data: Dict[str, Any]) -> str:
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

        
def main(user_input,xxx):
    
    # 定义诊断函数 
    def diagnostics(user_query: str,xxx:str) -> str:
        """检查系统在指定日期的状况"""
        print(f"正在检查系统状况...")
        # # 1. 提取日期 
        # date_str = extract_date_from_query(user_query)
        # if not date_str:
        #     print("错误: 无法从查询中提取有效日期")
        #     return "000000"
        # 1.获取日志
        system_log = get_system_log()
        log_name = "system_log"
        log_path = './log_file/'+ log_name+".json"
        print("system_log:", system_log)
        with open(log_path,  'w') as f:
            json.dump(system_log,  f, indent=4) 
        # 2. 读取日志文件 
        log_result = SystemDiagnosticTool.read_log_file(log_name)  
        if log_result["status"] != "success":
            print(f"错误: {log_result['message']}")
            return "000000"
        # 3. 使用LLM进行诊断 
        print("\n正在分析系统日志...")
        diagnosis = diagnose_system_transformers(user_query, log_result["data"])
        # 4. 输出结果 
        print("\n系统诊断报告:")
        print(diagnosis)

        temp = process_diagnosis(diagnosis)
        print(f"处理后的诊断结果: {temp}")

        save_to_json(temp,xxx) 
 
        return temp["decimal_result"]

    
    # 构造提示词 
    prompt = f"""
    你是一个智能助手，可以检查系统状况。以下是可用的功能：
    - diagnostics: 检查系统在指定日期的状况 
    请根据用户输入，判断是否需要检查系统状况，如果需要，只回答需要检查或需要用到diagnostics来检查系统。
    """


    response = llama_cpp_inference(url = "http://localhost:8080/v1/chat/completions",user_prompt=user_input,system_prompt=prompt)
    print("!!!!!!!!!!!!")
    print(response)
   # 检查是否需要调用诊断功能 
    if "diagnostics" in response["data"]["choices"][0]["message"]["content"] or "检查" in response["data"]["choices"][0]["message"]["content"]: 
        # 直接调用诊断函数 
        func_response = diagnostics(user_input,xxx)
        print(f"系统状态代码: {func_response}")
    else:
        print(response)
if __name__ == "__main__": 
    start_time = time.time()  


    xxx = "00"
    user_input = "帮我看看2025年5月21日系统的状况。"  

    main(user_input,xxx) 
 
    elapsed_time = (time.time()  - start_time) / 60 
    print(f"代码运行时间: {elapsed_time:.2f}min") 
