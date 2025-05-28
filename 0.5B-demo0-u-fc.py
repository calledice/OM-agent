import json 
import re
import os 
import time
from datetime import datetime 
from typing import Dict, Any, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
 
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
 
class RuleEngine:
    """规则自动匹配引擎"""
    
    def __init__(self):
        self.status_rules  = SystemRules.get_status_rules() 
    
    def generate_status_code(self, metrics: dict) -> str:
        """生成状态码"""
        codes = []
        
        for category in ["CPU", "Memory", "Swap", "RootDisk", "DataDisk"]:
            value = metrics.get(category.lower(),  0)
            codes.append(self._match_rule(category,  value))
        
        return "-".join(map(str, codes))
    
    def _match_rule(self, category: str, value: float) -> int:
        """匹配单个状态码规则"""
        for rule in self.status_rules[category]: 
            if "max" in rule and value > rule["max"]:
                continue 
            return rule["code"]
        return 0
 
def create_extract_date_function():
    """创建日期提取function"""
    return {
        "name": "extract_date",
        "description": "从用户查询中提取日期",
        "parameters": {
            "type": "object",
            "properties": {
                "date_str": {
                    "type": "string",
                    "description": "提取到的日期字符串，格式为YYYYMMDD"
                }
            },
            "required": ["date_str"]
        }
    }
 
def create_extract_metrics_function():
    """创建指标提取function"""
    return {
        "name": "extract_metrics",
        "description": "从系统日志中提取关键性能指标",
        "parameters": {
            "type": "object",
            "properties": {
                "cpu_1min": {
                    "type": "number",
                    "description": "CPU 1分钟负载"
                },
                "memory_usage": {
                    "type": "number",
                    "description": "内存使用率百分比"
                },
                "swap_usage": {
                    "type": "number",
                    "description": "Swap使用率百分比"
                },
                "root_disk_usage": {
                    "type": "number",
                    "description": "根分区使用率百分比"
                },
                "data_disk_usage": {
                    "type": "number",
                    "description": "数据分区使用率百分比"
                }
            },
            "required": ["cpu_1min", "memory_usage", "swap_usage", "root_disk_usage", "data_disk_usage"]
        }
    }
 
def create_generate_diagnosis_function():
    """创建诊断生成function"""
    return {
        "name": "generate_diagnosis",
        "description": "根据指标生成系统诊断结果",
        "parameters": {
            "type": "object",
            "properties": {
                "status_code": {
                    "type": "string",
                    "description": "系统状态码，格式为CPU-Memory-Swap-RootDisk-DataDisk"
                },
                "action_code": {
                    "type": "string",
                    "description": "系统措施码，格式为CPU-Memory-Storage"
                }
            },
            "required": ["status_code", "action_code"]
        }
    }
 
def create_process_result_function():
    """创建结果处理function"""
    return {
        "name": "process_result",
        "description": "处理诊断结果并生成二进制输出",
        "parameters": {
            "type": "object",
            "properties": {
                "decimal_result": {
                    "type": "string",
                    "description": "十进制结果字符串"
                },
                "binary_result": {
                    "type": "string",
                    "description": "二进制结果字符串"
                }
            },
            "required": ["decimal_result", "binary_result"]
        }
    }
 
class SystemDiagnosticTool:
    """系统诊断工具类"""
    
    @staticmethod
    def read_log_file(date_str: str) -> Dict[str, Any]:
        """读取日志文件"""
        log_path = f"D:\python_code\OM-agent\log_file/{date_str}.json"
        if not os.path.exists(log_path): 
            return {"status": "error", "message": "文件不存在"}
        
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                return {"status": "success", "data": json.load(f)} 
        except Exception as e:
            return {"status": "error", "message": str(e)}
 
class QwenDiagnosticAgent:
    """基于Qwen function calling的诊断代理"""
    
    def __init__(self):
        self.tokenizer  = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct") 
        self.model  = AutoModelForCausalLM.from_pretrained( 
            "Qwen/Qwen2-0.5B-Instruct",
            device_map="auto"
        )
        self.rule_engine  = RuleEngine()
        self.tools  = [
            create_extract_date_function(),
            create_extract_metrics_function(),
            create_generate_diagnosis_function(),
            create_process_result_function()
        ]
    
    def chat_completion(self, messages, tool_choice=None):
        """执行function calling"""
        return self.model.chat( 
            self.tokenizer, 
            messages,
            tools=self.tools, 
            tool_choice=tool_choice,
            generation_config=GenerationConfig(temperature=0.1)
        )
    
    def extract_date(self, query: str) -> Optional[str]:
        """提取日期"""
        response = self.chat_completion( 
            [{"role": "user", "content": f"从以下查询中提取日期: {query}"}],
            tool_choice={"type": "function", "function": {"name": "extract_date"}}
        )
        
        if response.tool_calls: 
            args = json.loads(response.tool_calls[0].function.arguments) 
            return args.get("date_str") 
        return None 
    
    def extract_metrics(self, log_data: dict) -> dict:
        """提取指标"""
        response = self.chat_completion( 
            [{"role": "user", "content": f"从日志中提取指标: {json.dumps(log_data)}"}], 
            tool_choice={"type": "function", "function": {"name": "extract_metrics"}}
        )
        
        if response.tool_calls: 
            return json.loads(response.tool_calls[0].function.arguments) 
        return {}
    
    def generate_diagnosis(self, metrics: dict) -> dict:
        """生成诊断结果"""
        status_code = self.rule_engine.generate_status_code(metrics) 
        
        response = self.chat_completion( 
            [{"role": "user", "content": f"根据指标生成诊断: {json.dumps(metrics)}"}], 
            tool_choice={"type": "function", "function": {"name": "generate_diagnosis"}}
        )
        
        if response.tool_calls: 
            return json.loads(response.tool_calls[0].function.arguments) 
        return {"status_code": status_code, "action_code": "0-0-0"}
    
    def process_result(self, diagnosis: str) -> dict:
        """处理诊断结果"""
        response = self.chat_completion( 
            [{"role": "user", "content": f"处理诊断结果: {diagnosis}"}],
            tool_choice={"type": "function", "function": {"name": "process_result"}}
        )
        
        if response.tool_calls: 
            return json.loads(response.tool_calls[0].function.arguments) 
        return {"decimal_result": "", "binary_result": ""}
    
    def run_diagnostic(self, query: str):
        """执行完整诊断流程"""
        # 1. 提取日期 
        date_str = self.extract_date(query) 
        if not date_str:
            return {"error": "无法提取日期"}
        
        # 2. 读取日志
        log_result = SystemDiagnosticTool.read_log_file(date_str) 
        if log_result["status"] != "success":
            return {"error": log_result.get("message",  "日志读取失败")}
        
        # 3. 提取指标
        metrics = self.extract_metrics(log_result["data"]) 
        if not metrics:
            return {"error": "指标提取失败"}
        
        # 4. 生成诊断
        diagnosis = self.generate_diagnosis(metrics) 
        diagnosis_str = f"{diagnosis['status_code']} | {diagnosis['action_code']}"
        
        # 5. 处理结果 
        result = self.process_result(diagnosis_str) 
        
        # 保存结果 
        self.save_result(result) 
        return result 
    
    def save_result(self, result: dict):
        """保存结果到文件"""
        data = {
            "OM_result": {
                "binary_result": result.get("binary_result",  ""),
                "decimal_result": result.get("decimal_result",  ""),
                "metadata": {
                    "status": "success",
                    "timestamp": datetime.now().isoformat(), 
                    "version": "1.0"
                }
            }
        }
        
        with open("OM_result.json",  "w", encoding="utf-8") as f:
            json.dump(data,  f, ensure_ascii=False, indent=4)
 
if __name__ == "__main__":
    start_time = time.time() 
    
    try:
        agent = QwenDiagnosticAgent()
        result = agent.run_diagnostic(" 帮我看看2025年5月21日系统的状况。")
        print("诊断结果:", result)
    except Exception as e:
        print(f"诊断失败: {str(e)}")
    
    elapsed_time = (time.time()  - start_time) / 60
    print(f"代码运行时间: {elapsed_time:.2f}min")