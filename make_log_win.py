import platform 
import os 
import socket  # 添加这行 
import psutil 
 
def get_linux_detailed_info():
    info = {
        "CPU Usage (%)": psutil.cpu_percent(interval=1), 
        "Memory Usage (%)": psutil.virtual_memory().percent, 
        "Disk Usage": {
            "Total (GB)": round(psutil.disk_usage("/").total  / (1024**3), 2),
            "Used (%)": round(psutil.disk_usage("/").used  / (1024**3)/(psutil.disk_usage("/").total  / (1024**3)), 2)*100,
            "Free (GB)": round(psutil.disk_usage("/").free  / (1024**3), 2),
        },
        "Network Interfaces": [
            {"Interface": name, "IP": addr.address} 
            for name, addrs in psutil.net_if_addrs().items() 
            for addr in addrs if addr.family  == socket.AF_INET  # 现在可以正常使用 socket.AF_INET 
        ],
        "Running Processes": len(list(psutil.process_iter())), 
    }
    return info 

# user_input = "请检查系统在2023年10月1日的状况。"
# prompt = f"""
# 你是一个智能助手，可以检查系统状况。以下是可用的功能：
# - diagnostics: 检查系统在指定日期的状况 

# 用户输入：{user_input}
# 请判断是否需要检查系统状况，如果需要，只回答需要检查或需要用到diagnostics来检查系统。
# """
# prompt = [prompt]

detailed_info = get_linux_detailed_info()
print(detailed_info)