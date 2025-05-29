import json 
import re 
from datetime import datetime 
from typing import Dict, Any, Union, List, Tuple 
import os 
import time 
from typing import Optional 
from base import save_to_json, process_diagnosis, extract_date_from_query
"""基于正则化表达式和规则引擎的系统诊断工具
该工具使用llama.cpp或transformers模型提取系统日志数据，更具正则化生成状态码和措施码。
0.5B模型对日志提取能力有限，因此提取替换为json数据的直接解析。
"""


 
try:
    from llama_cpp import Llama 
    LLAMA_CPP_AVAILABLE = True 
except ImportError:
    LLAMA_CPP_AVAILABLE = False 
 
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM 
    import torch 
    TRANSFORMERS_AVAILABLE = True 
except ImportError:
    TRANSFORMERS_AVAILABLE = False 

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
def diagnose_system_llamacpp(llm: 'Llama', user_query: str, log_data: Dict[str, Any]) -> str:
    """
    使用llama.cpp 模型分析日志数据（基于规则引擎）
    """
    # 1. 使用LLM提取关键指标 
    extraction_prompt = f"""
    请从以下JSON数据中提取系统监控指标：
    {json.dumps(log_data,  indent=2)}
    
    需要提取的指标：
    - cpu_1min: CPU 1分钟负载（12核心）
    - memory_usage: 内存使用率（百分比）
    - swap_usage: Swap使用率（百分比）
    - root_disk_usage: 根分区使用率（百分比）
    - data_disk_usage: 数据分区使用率（百分比）
    - buffers: 缓冲区占比（百分比，可选）
    
    请以JSON格式返回提取结果，只返回JSON数据，不要包含任何解释。
    """
    
    extracted = llm.create_chat_completion( 
        messages=[{"role": "user", "content": extraction_prompt}],
        response_format={"type": "json_object"}
    )
    
    try:
        metrics = json.loads(extracted["choices"][0]["message"]["content"]) 
    except:
        metrics = {}
    
    # 2. 使用规则引擎生成状态码和措施码 
    engine = RuleEngine()
    status_code = engine.generate_status_code(metrics) 
    action_code = engine.generate_action_code(metrics,  status_code)
    
    return f"{status_code} | {action_code}"

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
 
    # # 构建消息历史 
    # messages = [
    #     {"role": "system", "content": "你是一个严格的系统监控数据分析助手，只返回JSON格式的结果"},
    #     {"role": "user", "content": extraction_prompt}
    # ]
    
    # # 应用聊天模板
    # prompt = tokenizer.apply_chat_template( 
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True
    # )
    
    # # 生成回复（强制JSON格式）
    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device) 
    # outputs = model.generate( 
    #     **inputs,
    #     max_new_tokens=500,  # 设置为足够生成完整JSON的大小 
    #     min_new_tokens=30,   # 确保至少生成基本结构 
    #     pad_token_id=tokenizer.eos_token_id, 
    #     # do_sample=False  # 禁用随机性，提高稳定性
    # )
    
    # # 提取并清理响应
    # response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):],  skip_special_tokens=True)
    # # 2. 解析指标数据 
    # try:
    #     # 尝试提取JSON部分（处理模型可能添加的额外文本）
    #     json_start = response.find('{') 
    #     json_end = response.rfind('}')  + 1
    #     metrics = json.loads(response[json_start:json_end]) 
    # except (json.JSONDecodeError, ValueError) as e:
    #     print(f"指标提取失败: {str(e)}")
    #     metrics = {
    #         "cpu_1min": 0,
    #         "memory_usage": 0,
    #         "swap_usage": 0,
    #         "root_disk_usage": 0,
    #         "data_disk_usage": 0
    #     }

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


def process_user_request(model: Union['Llama', 'AutoModelForCausalLM'], 
                       tokenizer: Optional['AutoTokenizer'] = None, 
                       user_query: str = ""):
    """处理用户请求的完整流程"""
    print(f"\n用户请求: {user_query}")
    
    # 1. 提取日期 
    date_str = extract_date_from_query(user_query)
    if not date_str:
        print("错误: 无法从查询中提取有效日期")
        return 
    
    # 2. 读取日志文件 
    log_result = SystemDiagnosticTool.read_log_file(date_str) 
    if log_result["status"] != "success":
        print(f"错误: {log_result['message']}")
        return 
    
    # 3. 使用LLM进行诊断 
    print("\n正在分析系统日志...")
    if isinstance(model, Llama) and LLAMA_CPP_AVAILABLE:
        diagnosis = diagnose_system_llamacpp(model, user_query, log_result["data"])
    elif TRANSFORMERS_AVAILABLE and tokenizer is not None:
        diagnosis = diagnose_system_transformers(model, tokenizer, user_query, log_result["data"])
    else:
        raise RuntimeError("没有可用的模型后端")
    
    # 4. 输出结果 
    print("\n系统诊断报告:")
    print(diagnosis)
    # 5.保存
    # process_diagnosis(diagnosis)
    save_to_json(process_diagnosis(diagnosis), f"OM_result.json")


def load_llamacpp_model():
    """加载llama.cpp 模型"""
    if not LLAMA_CPP_AVAILABLE:
        raise ImportError("llama_cpp包未安装")
    
    print("正在初始化llama.cpp  Qwen模型...")
    return Llama.from_pretrained( 
        repo_id="Qwen/Qwen2-7B-Instruct-GGUF",
        filename="qwen2-7b-instruct-q4_k_m.gguf", 
        n_gpu_layers=10,
        n_ctx=2048,
        verbose=True,
        timeout=120,
    )
 
def load_transformers_model():
    """加载transformers模型"""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers包未安装")
    
    print("正在加载Qwen2 transformers模型...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct") 
    model = AutoModelForCausalLM.from_pretrained( 
        "Qwen/Qwen2-0.5B-Instruct",
        device_map="auto",
        # torch_dtype=torch.bfloat16  
    )
    return model, tokenizer 
 

if __name__ == "__main__":
    start_time = time.time()  
    
    # 选择模型后端 
    MODEL_BACKEND = "transformers"  # 可改为 "llamacpp" 或 "transformers"
    
    try:
        if MODEL_BACKEND == "llamacpp":
            qwen_model = load_llamacpp_model()
            tokenizer = None 
        elif MODEL_BACKEND == "transformers":
            qwen_model, tokenizer = load_transformers_model()
        else:
            raise ValueError(f"未知的模型后端: {MODEL_BACKEND}")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        exit(1)
    
    # 示例查询 
    user_input = "帮我看看2025年5月21日系统的状况。"
    
    # 处理请求 
    process_user_request(qwen_model, tokenizer, user_input)
 
    end_time = time.time()  
    elapsed_time = (end_time - start_time) / 60 
    print(f"代码运行时间: {elapsed_time:.2f}min")