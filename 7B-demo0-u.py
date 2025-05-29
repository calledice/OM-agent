import json 
import re 
from datetime import datetime 
from typing import Dict, Any, Union, List, Tuple 
import os 
import time 
from typing import Optional 
from base import save_to_json, process_diagnosis, extract_date_from_query, RuleEngine
"""基于正则化表达式和规则引擎的系统诊断工具
该工具使用llama.cpp或transformers模型提取系统日志数据，更具正则化生成状态码和措施码。
对于7B-instruct模型和7B-math模型效果都很好
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
    - trend: CPU负载趋势（rising/stable/falling，可选）
    - service_available: 服务是否可用（true/false，可选）
    
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
    使用transformers模型分析日志数据 
    
    参数:
        model: transformers模型实例 
        tokenizer: 对应的tokenizer 
        user_query: 原始用户查询 
        log_data: 日志数据 
        
    返回:
        [状态码] | [措施码]
    """
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": f"已获取系统日志数据: {json.dumps(log_data,  ensure_ascii=False)}"}
    ]
    
    # 应用聊天模板 
    prompt = tokenizer.apply_chat_template( 
        messages,
        tokenize=False,
        add_generation_prompt=True 
    )
    
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
    user_input = "帮我看看2025年5月21日系统的状况。"
    
    # 处理请求 
    process_user_request(qwen_model, tokenizer, user_input)
 
    end_time = time.time()  
    elapsed_time = (end_time - start_time) / 60 
    print(f"代码运行时间: {elapsed_time:.2f}min")