import requests
import logging
import json
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def llama_cpp_inference(url, user_prompt, model="default", system_prompt="You are a helpful assistant..", timeout=30):
    """
    发送大模型推理请求

    Args:
        url (str): 推理服务的 URL
        user_prompt (str): 用户的提示内容
        model (str, optional): 使用的模型名称，默认为 "default"
        system_prompt (str, optional): 系统的提示内容，默认为 "You are a helpful assistant.."
        timeout (int, optional): 请求超时时间（秒），默认为 30

    Returns:
        dict: 响应内容，包含推理结果或错误信息
    """
    try:
        # 构造消息列表
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # 设置请求头部和数据
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        data = {
            "messages": messages,
            "model": model
        }

        # 开始计时
        start_time = time.time()

        # 发送 POST 请求
        response = requests.post(url, headers=headers, json=data, timeout=timeout)

        # 计算耗时
        elapsed_time = time.time() - start_time

        # 记录请求相关信息
        logging.info(f"请求 URL: {url}")
        logging.info(f"请求数据: {json.dumps(data, ensure_ascii=False)}")
        logging.info(f"响应状态码: {response.status_code}")
        logging.info(f"响应内容: {response.text}")
        logging.info(f"请求耗时: {elapsed_time:.2f} 秒")

        # 检查响应状态码
        if response.status_code == 200:
            # 解析响应内容
            response_data = response.json()
            # 如果需要额外处理响应内容，可以在这里进行
            return {"success": True, "data": response_data, "elapsed_time": elapsed_time}
        else:
            return {
                "success": False,
                "error": f"请求失败，状态码：{response.status_code}",
                "response_text": response.text,
                "elapsed_time": elapsed_time
            }

    except Exception as e:
        # 记录异常信息
        elapsed_time = time.time() - start_time
        logging.error(f"请求发生异常: {str(e)}")
        return {
            "success": False,
            "error": f"请求发生异常: {str(e)}",
            "elapsed_time": elapsed_time
        }

# 示例调用
if __name__ == "__main__":
    url = "http://localhost:8080/v1/chat/completions"
    user_prompt = "你好"
    model = "default"

    result = llama_cpp_inference(url, user_prompt, model)

    if result["success"]:
        print("请求成功！")
        print(f"请求耗时: {result['elapsed_time']:.2f} 秒")
        print("响应数据:")
        print(json.dumps(result["data"], indent=2, ensure_ascii=False))
    else:
        print("请求失败！")
        print(f"请求耗时: {result['elapsed_time']:.2f} 秒")
        print(f"错误信息: {result['error']}")
