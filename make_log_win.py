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
 
detailed_info = get_linux_detailed_info()
print(detailed_info)