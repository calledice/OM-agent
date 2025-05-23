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
输出格式：系统诊断报告
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
        repo_id="Qwen/Qwen2-1.5B-Instruct-GGUF",
        filename="qwen2-1_5b-instruct-q4_k_m.gguf", 
        n_gpu_layers=40,
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
    使用LLM分析日志数据并生成诊断报告 
    
    参数:
        llm: 初始化的模型实例 
        user_query: 原始用户查询
        log_data: 日志数据
        
    返回:
        模型生成的诊断报告
    """
    response = llm.create_chat_completion( 
        messages=[
            { 
    "role": "system",
    "content": ("你是一个资深的系统运维专家，请严格按照工业标准分析系统日志：\n\n" 
 
                "# 分析要求\n"
                "1. 健康度评估标准（必须严格应用）：\n"
                "   - CPU负载：1分钟负载>8 → [严重问题]\n"
                "   - 内存使用率：(使用内存/总内存)>90% → [严重问题]\n" 
                "   - 存储使用率：(使用容量/总容量)>90% → [严重问题]\n" 
                "   - 网络错误：接收/发送错误>0 → [严重问题]\n"
                "   - 失败登录：累计次数>50 → [安全问题]\n\n" 
 
                "# 输出规范\n"
                "1. 变量替换规则：\n" 
                "   - 所有{}占位符必须用原始日志数据填充\n" 
                "   - 使用率计算保留1位小数（示例：{内存使用率:.1f}%）\n" 
                "   - 时间戳必须完整保留时区信息（如+08:00）\n\n" 
 
                "# 标准诊断报告模板\n"
                "---\n" 
                "【系统诊断报告】{当前日期}\n\n"
 
                "1. 硬件资源：\n"
                " - CPU：{核心数}核/负载{1分钟负载} [{CPU评估}] ← 负载趋势：{1分钟负载}/{5分钟负载}/{15分钟负载}\n"
                " - 内存：{使用内存}GiB/{总内存}GiB ({内存使用率:.1f}%) [{内存评估}]\n"
                " - 存储：\n" 
                "   • 根分区：{根使用}GiB/{根总容量}GiB ({根使用率:.1f}%) [{根评估}]\n" 
                "   • 数据分区：{数据使用}GiB/{数据总容量}TiB ({数据使用率:.1f}%) [{数据评估}]\n\n"
 
                "2. 网络接口（统计所有接口总和）：\n"
                " - 接口状态：{接口列表}\n" 
                " - 错误统计：接收错误{总rx_errors}次/发送错误{总tx_errors}次 [{网络评估}]\n\n" 
 
                "3. 进程监控：\n" 
                " - 运行状态：{进程总数}个进程（运行态{运行态}个）\n"
                " - 高负载进程：\n"
                "   • PID {pid1}：{命令1}（CPU使用率{cpu1}% | 内存使用率{mem1}%）\n"
                "   • PID {pid2}：{命令2}（CPU使用率{cpu2}% | 内存使用率{mem2}%）\n\n"
 
                "4. Docker服务：\n" 
                " - 网络模式：桥接模式IP {docker_ip} [{docker评估}]\n\n"
 
                "5. 安全审计：\n"
                " - 最后登录：{最后登录用户}@{登录IP}（{登录时间}）\n"
                " - 登录统计：累计失败{失败次数}次 [{安全评估}]\n\n" 
 
                "6. 健康评分体系：\n"
                " - 基础分：100分\n"
                " - 扣分项：\n" 
                "   • 每个[严重问题]扣5分\n"
                "   • 每个[安全问题]扣3分\n"
                " - 最终得分：{最终评分}/100 [{健康状态}]\n\n" 
 
                "7. 处理建议：\n" 
                " [紧急] 需立即处理：\n"
                "   {紧急事项}\n" 
                " [优化] 建议改进项：\n"
                "   • 使用htop实时监控\n"
                "   • 使用iostat -xmt 1分析存储IO\n" 
                "   • 使用tcpdump排查网络异常\n" 
                "   • 使用docker logs {容器ID}检查容器日志\n" 
                "---")
} ,
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
    # 初始化模型 
    print("正在初始化Qwen模型...")
    qwen_llm = initialize_llm()
    
    # 示例查询 
    user_input = "帮我看看2025年5月20日系统的状况。"
    
    # 处理请求
    process_user_request(qwen_llm, user_input)