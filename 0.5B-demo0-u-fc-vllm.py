import json 
import re
import os 
import time
from datetime import datetime 
from typing import Dict, Any, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from base import save_to_json, process_diagnosis, extract_date_from_query, RuleEngine
import sys 
from vllm_inference_sample import run_llm_inference

"""
如果有qwen_agent包，那么可以LLM直接调用封装好的function
但是由于没有这个包，目前只能LLM用作是否执行function
"""


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

        
def main(user_input,xxx):
    
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
        diagnosis = diagnose_system_transformers(user_query, log_result["data"])
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
    
    response, num_tokens, qps = run_llm_inference(prompt)
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
    xxx = "00"
    main(user_input,xxx) 
 
    elapsed_time = (time.time()  - start_time) / 60 
    print(f"代码运行时间: {elapsed_time:.2f}min") 