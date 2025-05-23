import subprocess 
import re 
from typing import Tuple 
from pathlib import Path 
from llama_cpp import Llama 
 
class LocalShellGPT:
    def __init__(self):
        # 初始化本地模型 
        self.llm  = Llama(
            model_path="./models/qwen2-1_5b-instruct-q4_k_m.gguf", 
            n_ctx=2048,
            n_gpu_layers=40 
        )
        
        # 命令白名单 (可按需扩展)
        self.safe_commands  = {
            'list_files': {'pattern': r'^ls(?: -[lha]+)?(?: /[-\w./]*)?$', 'desc': '列出目录内容'},
            'disk_usage': {'pattern': r'^df -h$', 'desc': '查看磁盘使用'},
            'system_info': {'pattern': r'^(uname -a|hostnamectl)$', 'desc': '查看系统信息'}
        }
        
        # 危险命令黑名单 
        self.dangerous_patterns  = [
            r'rm\s+', r'^dd\s+', r'mv\s+.*\s+/', 
            r'^sudo\s+', r'chmod\s+[0-7]{3}\s+', 
            r'>', r'\|.*\b(sh|bash|zsh)\b'
        ]
 
    def is_command_safe(self, command: str) -> bool:
        """检查命令安全性"""
        # 检查黑名单 
        if any(re.search(p,  command) for p in self.dangerous_patterns): 
            return False 
            
        # 检查是否在白名单 
        return any(
            re.fullmatch(cmd['pattern'],  command.strip()) 
            for cmd in self.safe_commands.values() 
        )
 
    def execute_command(self, command: str) -> Tuple[str, bool]:
        """安全执行命令"""
        if not self.is_command_safe(command): 
            return f"⚠️ 拒绝执行危险命令: {command}", False 
            
        try:
            result = subprocess.run( 
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=15,
                executable="/bin/bash"
            )
            output = result.stdout  or result.stderr  
            return output, result.returncode  == 0 
        except Exception as e:
            return f"命令执行失败: {str(e)}", False 
 
    def generate_response(self, prompt: str) -> str:
        """生成LLM回复"""
        response = self.llm.create_chat_completion( 
            messages=[{
                "role": "system",
                "content": """你是一个安全的终端助手，可以：
                1. 解释Linux命令 
                2. 执行简单的安全命令（仅限白名单）
                3. 拒绝危险操作 
                回答要简明扼要"""
            }, {
                "role": "user",
                "content": prompt 
            }],
            max_tokens=512,
            temperature=0.2 
        )
        return response['choices'][0]['message']['content']
 
    def process_query(self, query: str) -> str:
        """处理用户查询"""
        # 第一步：判断是否需要执行命令 
        analysis = self.llm.create_chat_completion( 
            messages=[{
                "role": "system",
                "content": "判断用户是否需要执行实际命令（回答只要'是'或'否'）"
            }, {
                "role": "user",
                "content": query 
            }],
            max_tokens=2 
        )
        
        needs_execution = '是' in analysis['choices'][0]['message']['content']
        
        if not needs_execution:
            return self.generate_response(query) 
            
        # 第二步：生成要执行的命令 
        command_prompt = f"""用户请求: {query}
        
        请生成一个安全的Linux命令来满足这个请求。
        要求:
        1. 必须是简单的单条命令 
        2. 只能包含: {[k for k in self.safe_commands.keys()]} 
        3. 不要包含解释 
        
        只需返回命令本身，不要有其他内容。"""
        
        command = self.llm.create_chat_completion( 
            messages=[{"role": "user", "content": command_prompt}],
            max_tokens=20 
        )['choices'][0]['message']['content'].strip()
        
        # 第三步：执行并返回结果 
        output, success = self.execute_command(command) 
        response = f"$ {command}\n{output}"
        
        if not success:
            response += "\n\n⚠️ 命令执行出错，建议检查命令语法"
            
        return response 
 
# 使用示例 
if __name__ == "__main__":
    assistant = LocalShellGPT()
    
    # 测试不同场景 
    test_cases = [
        "查看当前目录内容",
        "显示磁盘使用情况",
        "删除所有文件",  # 应被拒绝 
        "如何解压tar文件?",  # 知识性问题 
        "列出/home下的文件"
    ]
    
    for query in test_cases:
        print(f"\n用户: {query}")
        print("助手:", assistant.process_query(query)) 