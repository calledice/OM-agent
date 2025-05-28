import json 
 
# 文件路径 
file_path = r"./OM_result.json" 
 
try:
    # 打开并读取 JSON 文件 
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file) 
    
    # 打印读取的数据（或进行其他处理）
    print("成功读取 JSON 文件:")
    print(data)

    print("二进制编码为：")
    print(data["OM_result"]["binary_result"])

    print("十进制编码为：")
    print(data["OM_result"]["decimal_result"])
 

except FileNotFoundError:
    print(f"错误: 文件 {file_path} 不存在")
except json.JSONDecodeError:
    print(f"错误: 文件 {file_path} 不是有效的 JSON 格式")
except Exception as e:
    print(f"读取文件时发生错误: {str(e)}")

