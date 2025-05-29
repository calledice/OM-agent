import json 
import re 
from datetime import datetime 
from typing import Dict, Any, Union 
import os 
import time 
from typing import Optional 
from datetime import datetime
from base import save_to_json, process_diagnosis, extract_date_from_query,build_system_prompt
 
"""基于LLM理解的系统诊断工具，LLM读入日志文件中的关键数据，基于prompt对比分析系统状态并生成诊断结果
对于7B-instruct模型效果不好，因为数学能力太弱了
对于7B-math模型效果都很好（math模型的指令遵循不大好，所以需要正则化从回答中提取编码）
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
 
 
def setup_tools():
    """设置function calling的工具描述"""
    return [
        {
            "type": "function",
            "function": {
                "name": "retrieve_system_logs",
                "description": "根据日期检索系统日志数据",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "日志日期，格式为YYYYMMDD",
                            "format": "date"
                        }
                    },
                    "required": ["date"]
                }
            }
        }
    ]

def preprocess_log_data(log_data: Dict[str, Any]) -> Dict[str, float]:
    """从日志中提取并格式化 CPU、内存、磁盘等指标"""
    # 1. CPU 使用率：用 load_average[0] 近似代替（假设逻辑核心数为 8）
    cpu_cores = 12  # 需根据实际硬件调整 
    cpu_usage = log_data["system_log"]["system"]["uptime"]["load_average"]
    
    # 2. 内存和 Swap 
    mem_usage = float(log_data["system_log"]["hardware"]["memory"]["usage"].strip("%"))
    swap_usage = float(log_data["system_log"]["hardware"]["memory"]["swap"]["usage"].strip("%"))
    
    # 3. 磁盘（Root 和 Data）
    root_disk = next(
        d for d in log_data["system_log"]["hardware"]["storage"] if d["mount"] == "/"
    )
    data_disk = next(
        d for d in log_data["system_log"]["hardware"]["storage"] if d["mount"] == "/data"
    )
    
    return {
        "cpu": round(cpu_usage*100), #避免小数比较
        "mem": round(mem_usage),
        "swap": round(swap_usage),
        "root_disk": round(float(root_disk["usage"].strip("%"))),
        "data_disk": round(float(data_disk["usage"].strip("%"))),
    }

def diagnose_system_llamacpp(llm: 'Llama', user_query: str, log_data: Dict[str, Any]) -> str:
    metrics = preprocess_log_data(log_data)
    print(metrics)
    prompt = build_system_prompt(metrics)
    
    response = llm.create_chat_completion( 
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "生成状态码和措施码"},
        ],
    # temperature=0.0,
    # max_tokens=20,
    # stop=["\n"] # 阻止多行输出 
)
    return response["choices"][0]["message"]["content"]
 
def diagnose_system_transformers(model, tokenizer, user_query: str, log_data: Dict[str, Any]) -> str:
    """
    使用transformers模型分析日志数据 
    
    参数:
        model: transformers模型实例 
        tokenizer: 对应的tokenizer 
        user_query: 原始用户查询 
        log_data: 日志数据 
        
    返回:
        [状态码] | [措施码]
    """
    metrics = preprocess_log_data(log_data)
    prompt = build_system_prompt(metrics)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_query},
        # {"role": "assistant", "content": f"已获取系统日志数据: {json.dumps(log_data,  ensure_ascii=False)}"}
    ]
    
    # 生成回复 
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device) 
    outputs = model.generate( 
        **inputs,
        max_new_tokens=100,
        pad_token_id=tokenizer.eos_token_id  
    )
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):],  skip_special_tokens=True)
    
    # 清理输出（确保格式严格）
    response = response.strip() 
    if "|" not in response:
        response = "0-0-0-0-0 | 0-0-0"  # 默认安全值 
    return response 
 
def process_user_request(model: Union['Llama', 'AutoModelForCausalLM'], 
                         tokenizer: Optional['AutoTokenizer'] = None, 
                         user_query: str = ""):
    """
    处理用户请求的完整流程 
    
    参数:
        model: 模型实例 (llama_cpp.Llama 或 transformers.AutoModelForCausalLM)
        tokenizer: transformers的tokenizer (仅transformers模型需要)
        user_query: 用户输入的查询 
    """
    print(f"\n用户请求: {user_query}")
    
    # 1. 提取日期 
    date_str = extract_date_from_query(user_query)
    if not date_str:
        print("错误: 无法从查询中提取有效日期")
        return 
    
    # 2. 读取日志文件 
    log_result = SystemDiagnosticTool.read_log_file(date_str)  
    print(log_result["data"])  # 输出日志读取结果
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
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct") 
    model = AutoModelForCausalLM.from_pretrained( 
        "Qwen/Qwen2-7B-Instruct",
        device_map="auto",
        torch_dtype=torch.bfloat16  
    )
    return model, tokenizer 
 
 
if __name__ == "__main__":
    start_time = time.time() 
    
    # 选择模型后端 
    MODEL_BACKEND = "llamacpp"  # 可改为 "llamacpp" 或 "transformers"
    
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
    user_input = "帮我看看2025年5月20日系统的状况。"
    
    # 处理请求 
    process_user_request(qwen_model, tokenizer, user_input)
 
    end_time = time.time() 
    elapsed_time = (end_time - start_time) / 60 
    print(f"代码运行时间: {elapsed_time:.2f}min")