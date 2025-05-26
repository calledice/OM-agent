from llama_cpp import Llama 
import json
import re 
from datetime import datetime
from typing import Dict, Any
import os 
 
"""
仅支持运维相关的问答，能读取日志并做分析
输入格式：帮我看看2025年5月20日系统的状况。
2025年5月20日必须显示的给出
输出格式：[状态码] | [措施码]
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
        log_path = f"/data/cong/log_file/{date_str}.json"
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
 
def initialize_llm():
    """初始化Qwen模型"""
    return Llama.from_pretrained( 
        repo_id="Qwen/Qwen2-7B-Instruct-GGUF",
        filename="qwen2-7b-instruct-q4_k_m.gguf", 
        n_gpu_layers=10,
        n_ctx=2048,
        verbose=True
    )
 
def extract_date_from_query(query: str) -> str:
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
                return dt.strftime("%Y%m%d")   # 统一转为YYYYMMDD格式 
            except ValueError:
                continue
    return None
 
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
 
def diagnose_system(llm: Llama, user_query: str, log_data: Dict[str, Any]) -> str:
    """
    使用LLM分析日志数据并生成[状态码] | [措施码] 
    
    参数:
        llm: 初始化的模型实例 
        user_query: 原始用户查询
        log_data: 日志数据
        
    返回:
        [状态码] | [措施码]
    """
    response = llm.create_chat_completion( 
        messages=[
            {
            "role": "system",
            "content":"""
            你只能回复系统状态码和措施码，格式严格要求如下：
            [状态码] | [措施码]
            （两个代码用竖线分隔，首尾无空格，无其他字符）
            【输出示例】
            0-2-1-0-1 | 1-0-2 
            3-1-0-2-2 | 2-3-1
            
            【指令约束】
            1. 必须严格按以下规则生成状态码和措施码 
            2. **禁止任何解释性文字、分析过程或额外格式**
            3. 仅输出最终代码，格式为：[状态码] | [措施码]
            
            卫星系统状态码和措施码生成规则
            
            【状态码规则】（A-B-C-D-E）
            A(CPU): 0:≤1  1:≤3  2:≤8  3:>8（1分钟负载/12核心）
            B(内存): 0:<70%  1:70-90%  2:>90%  3:OOM/剩余<5%
            C(Swap): 0:未用  1:>50%  2:>90%
            D(根分区): 0:<70%  1:70-90%  2:>90%  3:剩余<1GB/≥99%
            E(数据分区): 0:<80%  1:80-95%  2:>95%
            
            【措施码规则】（X-Y-Z）
            X(CPU): 0:无操作 1:限制非核心进程 2:强制降频 3:重启模块
            Y(内存): 0:无操作 1:释放缓存 2:终止TOP3进程 3:激活压缩算法 
            Z(存储): 0:无操作 1:删除旧日志 2:停止数据采集 3:切换只读模式
            
            【执行要求】
            根据输入数据，进行回复
            
            """} ,
            {
                "role": "user",
                "content": user_query
            },
            {
                "role": "assistant",
                "content": f"已获取系统日志数据: {json.dumps(log_data,  ensure_ascii=False)}"
            }
        ]
    )
    return response["choices"][0]["message"]["content"]
 
def process_user_request(llm: Llama, user_query: str):
    """
    处理用户请求的完整流程
    
    参数:
        llm: 初始化的模型实例 
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
    if log_result["status"] != "success":
        print(f"错误: {log_result['message']}")
        return 
    
    # 3. 使用LLM进行诊断 
    print("\n正在分析系统日志...")
    diagnosis = diagnose_system(llm, user_query, log_result["data"])
    
    # 4. 输出结果 
    print("\n系统诊断报告:")
    print(diagnosis)
 
if __name__ == "__main__":
    start_time = time.time()
    # 初始化模型 
    print("正在初始化Qwen模型...")
    qwen_llm = initialize_llm()
    
    # 示例查询 
    user_input = "帮我看看2025年5月20日系统的状况。"
    
    # 处理请求
    process_user_request(qwen_llm, user_input)

    end_time = time.time()
    elapsed_time = (end_time - start_time)/60
    print(f"代码运行时间: {elapsed_time:.2f}min")