import psutil 
import platform 
from datetime import datetime, timedelta 
import json 
 
def get_system_log():
    # 1. 获取系统启动时间（计算 uptime）
    boot_time = datetime.fromtimestamp(psutil.boot_time()) 
    uptime = datetime.now()  - boot_time 
    uptime_days = uptime.days  
    uptime_hours, remainder = divmod(uptime.seconds,  3600)
    uptime_minutes, _ = divmod(remainder, 60)
 
    # 2. 获取内存和交换分区信息 
    mem = psutil.virtual_memory() 
    swap = psutil.swap_memory() 
 
    # 3. 获取磁盘分区信息 
    disks = []
    for partition in psutil.disk_partitions(): 
        usage = psutil.disk_usage(partition.mountpoint) 
        disks.append({ 
            "device": partition.device, 
            "mount": partition.mountpoint, 
            "size": f"{usage.total  / (1024**3):.1f} GiB",
            "usage": f"{usage.percent}%" 
        })
 
    # 4. 构建最终的 JSON 结构 
    log_data = {
        "system_log": {
            "timestamp": datetime.now().astimezone().replace(microsecond=0).isoformat(), 
            "system": {
                "uptime": {
                    "days": uptime_days,
                    "hours": uptime_hours,
                    "minutes": uptime_minutes,
                    "load_average": [x / psutil.cpu_count()  for x in psutil.getloadavg()]   # 标准化负载 
                }
            },
            "hardware": {
                "memory": {
                    "total": f"{mem.total  / (1024**3):.1f} GiB",
                    "usage": f"{mem.percent}%", 
                    "buffers": f"{mem.buffers  / (1024**3):.1f} GiB",
                    "swap": {
                        "total": f"{swap.total  / (1024**3):.1f} GiB",
                        "usage": f"{swap.percent}%" 
                    }
                },
                "storage": disks 
            }
        }
    }
    return log_data 
if __name__ == "__main__":
    # 直接运行脚本时获取系统日志 
    system_log = get_system_log()
    print(json.dumps(system_log, indent=2))
