from llama_cpp import Llama 
import json 
import re 
from datetime import datetime 
from typing import Dict, Any, Optional, List
import os 
from pathlib import Path
 
 
"""
支持运维相关的问答和日常对话，能读取日志并做分析
输入格式：帮我看看2025年5月21日系统的状况。
日期信息显示的给出，但不局限于2025年5月20日
输出格式：系统诊断报告
"""

class LogManager:
    """日志管理类，负责日志文件的读取和处理"""
    
    def __init__(self, log_dir: str = "/data/cong/log_file"):
        self.log_dir  = Path(log_dir)
        
    def get_log_path(self, date_str: str) -> Path:
        """获取指定日期的日志文件路径"""
        return self.log_dir  / f"{date_str}.json"
    
    def read_log(self, date_str: str) -> Dict[str, Any]:
        """
        读取指定日期的日志文件 
        返回包含状态和数据的字典
        """
        log_path = self.get_log_path(date_str) 
        
        if not log_path.exists(): 
            return {
                "status": "error",
                "message": f"日志文件 {log_path} 不存在",
                "data": None
            }
        
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                return {
                    "status": "success",
                    "message": "日志读取成功",
                    "data": json.load(f) 
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"读取日志时发生错误: {str(e)}",
                "data": None
            }
 
class AIDialogSystem:
    """AI对话系统核心类"""
    
    def __init__(self):
        self.llm  = self.initialize_llm() 
        self.log_manager  = LogManager()
        
    @staticmethod
    def initialize_llm() -> Llama:
        """初始化LLM模型"""
        return Llama.from_pretrained( 
            repo_id="Qwen/Qwen2-1.5B-Instruct-GGUF",
            filename="qwen2-1_5b-instruct-q4_k_m.gguf", 
            n_gpu_layers=40,
            n_ctx=2048,
            verbose=False 
        )
    
    def extract_date(self, query: str) -> Optional[str]:
        """使用LLM结合规则从查询中提取日期"""
        # 1. 先尝试使用正则表达式快速提取（高效处理标准格式）
        date_patterns = {
            r'(\d{4})年(\d{1,2})月(\d{1,2})日': "%Y年%m月%d日",
            r'(\d{4})-(\d{1,2})-(\d{1,2})': "%Y-%m-%d",
            r'(\d{4})/(\d{1,2})/(\d{1,2})': "%Y/%m/%d",
            r'(\d{4})(\d{2})(\d{2})': "%Y%m%d"  # 添加纯数字格式 
        }
        
        for pattern, fmt in date_patterns.items():  
            if match := re.search(pattern,  query):
                try:
                    dt = datetime.strptime(match.group(),  fmt)
                    return dt.strftime("%Y%m%d")  
                except ValueError:
                    continue 
        
        # 2. 如果没有通过正则匹配到，使用LLM提取（处理复杂情况）
        prompt = f"""
        请从以下用户查询中精确提取日期信息，并严格按照YYYYMMDD格式返回:
        {query}
        
        要求:
        1. 支持绝对日期(如"2025年5月21日"→20250521)
        2. 支持相对日期(如"三天前"需转换为具体日期)
        3. 支持模糊日期(如"五一假期"→20250501)
        4. 支持节日日期(如"国庆节"→20241001)
        5. 若无明确日期或无法确定，返回"无"
        
        示例:
        输入: "查看2023年双十一的系统状态"
        输出: 20231111 
        
        输入: "上周五服务器怎么了"
        输出: 20231020 (假设今天是2023年10月25日)
        
        输入: "系统使用说明"
        输出: 无 
        
        现在请处理这个查询，只返回8位数字日期或"无":
        """
        
        try:
            response = self.llm.create_chat_completion( 
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # 完全确定性输出 
                max_tokens=10,
                stop=["\n", "。"]  # 防止多行输出 
            )
            
            result = response["choices"][0]["message"]["content"].strip()
            
            # 3. 结果验证和处理 
            if result == "无":
                return None 
                
            # 检查是否为8位数字日期 
            if re.fullmatch(r"\d{8}",  result):
                try:
                    # 验证是否是有效日期 
                    datetime.strptime(result,  "%Y%m%d")
                    return result 
                except ValueError:
                    pass 
                    
            return None 
            
        except Exception as e:
            print(f"日期提取异常: {str(e)}")
            return None 
    
    def is_log_query(self, query: str) -> bool:
        """使用LLM判断是否是日志查询"""
        # 使用更精确的LLM判断 
        response = self.llm.create_chat_completion( 
            messages=[
                {"role": "system", "content": (
                    "你是一个系统日志查询判断助手。请严格根据以下规则判断用户是否在请求查询系统日志或诊断系统状况:\n"
                    "1. 如果用户明确请求查看日志、系统状态或诊断问题，回答'是'\n"
                    "2. 如果用户询问系统运行状况、错误信息或性能问题，回答'是'\n"
                    "3. 如果用户只是询问一般系统信息或与日志无关的问题，回答'否'\n"
                    "4. 如果请求中包含日期时间信息且与系统监控相关，回答'是'\n"
                    "\n"
                    "请只回答单个词语'是'或'否'，不要解释。"
                )},
                {"role": "user", "content": query}
            ],
            max_tokens=2,
            temperature=0.0,
            top_p=0.1  # 限制采样范围提高确定性 
        )
        
        answer = response["choices"][0]["message"]["content"].strip().lower()
        
        # 双重验证机制 
        if answer == "是":
            # 对肯定回答进行二次验证 
            date_exists = bool(self.extract_date(query)) 
            if date_exists:
                return True 
            
            # 检查是否包含系统相关关键词 
            system_keywords = ["系统", "server", "服务", "状态", "application", "状况"]
            if any(keyword in query for keyword in system_keywords):
                return True 
            
            # 如果只有LLM判断为是但无其他佐证，进行二次确认 
            confirm_prompt = (
                f"用户询问: '{query}'\n"
                "这确实是一个请求查看系统日志或诊断系统状况的查询吗？"
                "请只回答'是'或'否'"
            )
            confirm_response = self.llm.create_chat_completion( 
                messages=[{"role": "user", "content": confirm_prompt}],
                max_tokens=2,
                temperature=0.0 
            )
            confirm_answer = confirm_response["choices"][0]["message"]["content"].strip().lower()
            return confirm_answer == "是"
        
        return False 
    
    def process_log_request(self, query: str) -> str:
        """处理日志查询请求"""
        date_str = self.extract_date(query) 
        if not date_str:
            return "请提供有效的日期信息，例如：2025年5月20日"
        
        log_result = self.log_manager.read_log(date_str) 
        if log_result["status"] != "success":
            return f"无法获取日志数据: {log_result['message']}"
        
        # 直接使用原始日志数据，不进行摘要处理 
        response = self.llm.create_chat_completion( 
            messages=[
                {"role": "system", "content": (
                    "你是一个资深的系统运维专家，请按照严格的工业标准分析系统日志：\n\n"
                    
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
                )},
                {"role": "user", "content": query},
                {"role": "assistant", "content": json.dumps( 
                    log_result["data"], 
                    ensure_ascii=False
                )}
            ],
            max_tokens=512 
        )
        return response["choices"][0]["message"]["content"]
    
    def process_general_query(self, query: str) -> str:
        """处理一般查询"""
        response = self.llm.create_chat_completion( 
            messages=[
                {"role": "system", "content": "你是一个乐于助人的AI助手"},
                {"role": "user", "content": query}
            ],
            max_tokens=512 
        )
        return response["choices"][0]["message"]["content"]
    
    def chat_loop(self):
        """主对话循环"""
        print("系统已就绪，输入'退出'或'quit'结束对话")
        
        while True:
            try:
                user_input = input("\n用户: ").strip()
                if user_input.lower()  in ["退出", "quit", "exit"]:
                    break 
                if not user_input:
                    continue
                
                if self.is_log_query(user_input): 
                    print("\n[系统] 正在查询日志...")
                    response = self.process_log_request(user_input) 
                else:
                    response = self.process_general_query(user_input) 
                
                print(f"\n助手: {response}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n发生错误: {str(e)}")
 
if __name__ == "__main__":
    print("正在初始化系统...")
    dialog_system = AIDialogSystem()
    dialog_system.chat_loop() 
    #帮我看看2025年5月21日系统的状况