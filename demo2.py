from langgraph.graph  import Graph 
from llama_cpp import Llama 
import json 
import os 
from typing import Dict, TypedDict, Optional 
import re 
from datetime import datetime 
 
# 初始化LLM 
llm = Llama.from_pretrained( 
    repo_id="Qwen/Qwen2-1.5B-Instruct-GGUF",
    filename="qwen2-1_5b-instruct-q4_k_m.gguf", 
    n_gpu_layers=40,
    verbose=True 
)

# 定义状态类型 
class AgentState(TypedDict):
    user_input: str 
    extracted_date: Optional[str]  # 新增字段，存储提取的日期 
    log_content: Optional[Dict]
    analysis_result: Optional[str]
    response: Optional[str]
 
# 工具函数：使用LLM提取日期信息 
def extract_date_with_llm(user_input: str) -> str:
    # 构建提示词让LLM提取日期 
    prompt = f"""
    请从以下用户指令中提取日期信息，并以YYYYMMDD格式返回。
    如果无法提取日期，请返回"无法识别日期"。
    
    用户指令: {user_input}
    
    只返回日期，不要包含其他任何文字。
    示例:
    输入: "查看2023年10月15日的系统日志"
    输出: 20231015 
    
    输入: "帮我看看上周五的系统状态"
    输出: 无法识别日期 
    """
    
    response = llm.create_chat_completion( 
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,  # 降低随机性 
        max_tokens=20 
    )
    
    date_str = response['choices'][0]['message']['content'].strip()
    
    # 验证提取的日期格式 
    if re.fullmatch(r"\d{8}",  date_str):
        return date_str 
    else:
        raise ValueError("无法从输入中提取有效日期")
 
# 工具节点：提取日期信息 
def extract_date_tool(state: AgentState) -> Dict:
    print("正在提取日期信息...")
    try:
        date_str = extract_date_with_llm(state["user_input"])
        return {"extracted_date": date_str}
    except Exception as e:
        return {"error": str(e)}
 
# 工具函数：构建日志文件路径 
def get_log_path(date_str: str) -> str:
    log_path = f"/data/log_file/{date_str}.json"
    return log_path 
 
# 工具函数：读取日志文件 
def read_log_file(log_path: str) -> Dict:
    if not os.path.exists(log_path): 
        raise FileNotFoundError(f"日志文件 {log_path} 不存在")
    
    with open(log_path, 'r', encoding='utf-8') as f:
        return json.load(f) 
 
# 工具节点：读取日志文件 
def read_log_tool(state: AgentState) -> Dict:
    print("正在读取日志文件...")
    if "extracted_date" not in state or not state["extracted_date"]:
        return {"error": "没有可用的日期信息"}
    
    try:
        log_path = get_log_path(state["extracted_date"])
        log_content = read_log_file(log_path)
        return {"log_content": log_content}
    except Exception as e:
        return {"error": str(e)}
 
# 工具节点：分析日志内容 
def analyze_log_tool(state: AgentState) -> Dict:
    print("正在分析日志内容...")
    if "log_content" not in state or not state["log_content"]:
        return {"error": "没有可分析的日志内容"}
    
    # 构建分析提示 
    prompt = f"""
    你是一个系统运维专家，请分析以下系统日志数据，找出潜在问题并提供建议。
    日志日期: {state['extracted_date']}
    日志内容:
    {json.dumps(state['log_content'],  indent=2, ensure_ascii=False)}
    
    请按照以下要求回答:
    "# 分析要求\n"
    "必须对每个指标进行健康度评估，使用以下标准：\n"
    "   - 内存使用>90%总内存 = 严重问题\n"
    "   - 存储使用>90%总容量 = 严重问题\n"
    "   - 网络存在错误 = 严重问题\n"
    "   - 失败登录>50次 = 安全问题\n\n"
    
    "# 输出格式\n"
    "---\n"
    "【系统诊断报告】{日期}\n\n"
    "1. 硬件状况：\n"
    "   - CPU：{核心数}核/{使用率}% (状态)\n"
    "   - 内存：{使用内存}GiB /{使用率}% [{评估}] ← 大于90%必须标注'严重不足'\n"
    "   - 存储：{使用容量}GiB /{使用率}% [{评估}] ← 大于90%必须标注'空间告急'\n\n"
    
    "2. 网络状况：\n"
    "   - {接口}：{状态} (错误代码) [{评估}]\n\n"
    
    "3. 运行状态：\n"
    "   - 进程数：{总数} (高负载进程：{列表})\n\n"
    
    "4. Docker状况：\n"
    "   - 连接状态：{状态} [{评估}]\n\n"
    
    "5. 安全状况：\n"
    "   - 失败登录：{次数}次 [{评估}]\n\n"
    
    "6. 综合健康状态：\n"
    "   - 健康评分：{分数}/100 [{评估}] ← 根据问题严重性扣分,一个严重问题扣15分，一个安全问题扣10分\n"
    
    "7. 诊断建议：\n"
    "   [紧急] 需要立即处理的问题：\n"
    "   - {问题1} → {解决方案}\n"
    "   - {问题2} → {解决方案}\n\n"
    "   [优化] 建议改进项：\n"
    "   - {建议1}\n"
    "   - {建议2}\n"
    "---\n\n"
    
    "# 强制规则\n"
    "1. 发现任何指标达到'严重问题'标准时，综合状态必须标注为'严重告警'\n"
    "2. 诊断建议必须包含可操作的具体命令（如：`df -h`查看存储详情）\n"
    "3. 禁止出现结论与指标分析矛盾的情况\n"
    "4. 使用中文术语（如：GiB而不是GB）"
    """
    
    # 调用LLM进行分析 
    response = llm.create_chat_completion( 
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2000 
    )
    
    analysis_result = response['choices'][0]['message']['content']
    return {"analysis_result": analysis_result}
 
# 工具节点：生成最终响应 
def generate_response_tool(state: AgentState) -> Dict:
    print("正在生成响应...")
    if "analysis_result" not in state:
        error_msg = state.get("error",  "抱歉，无法完成系统诊断。")
        return {"response": error_msg}
    
    # 格式化最终响应 
    formatted_response = f"""
    ===== 系统诊断报告 ===== 
    诊断日期: {state['extracted_date']}
    
    {state['analysis_result']}
    
    ===== 报告结束 ===== 
    """
    return {"response": formatted_response.strip()} 
 
# 构建工作流图 
def create_workflow():
    workflow = Graph()
    
    # 添加节点 
    workflow.add_node("extract_date",  extract_date_tool)  # 新增日期提取节点 
    workflow.add_node("read_log",  read_log_tool)
    workflow.add_node("analyze_log",  analyze_log_tool)
    workflow.add_node("generate_response",  generate_response_tool)
    
    # 设置入口点 
    workflow.set_entry_point("extract_date") 
    
    # 添加条件边 
    def should_continue(state: AgentState) -> str:
        if "error" in state:
            return "generate_response"  # 直接跳到生成响应 
        elif "extracted_date" in state:
            return "read_log"
        else:
            return "generate_response"
    
    # 添加边 
    workflow.add_conditional_edges("extract_date",  should_continue)
    workflow.add_edge("read_log",  "analyze_log")
    workflow.add_edge("analyze_log",  "generate_response")
    
    # 设置结束点 
    workflow.set_finish_point("generate_response") 
    
    return workflow 
 
# 主函数 
def main():
    # 创建并编译工作流 
    workflow = create_workflow()
    app = workflow.compile() 
    
    # 测试用例 
    test_inputs = [
        "帮我看看2025年5月20日系统的状况。",
        "请检查2025年12月31日的系统日志",
        "查看明年1月1日的系统状态",  # LLM需要能处理相对日期 
        "系统在上周三有什么问题吗",  # 测试相对日期 
        "随便说点什么没有日期的话"  # 测试错误处理 
    ]
    
    for user_input in test_inputs:
        print(f"\n{'='*40}")
        print(f"处理用户输入: {user_input}")
        
        # 初始化状态 
        initial_state = AgentState(
            user_input=user_input,
            extracted_date=None,
            log_content=None,
            analysis_result=None,
            response=None 
        )
        
        # 执行工作流 
        result = app.invoke(initial_state) 
        
        # 输出结果 
        print("\n诊断结果:")
        print(result["response"])
 
if __name__ == "__main__":
    main()