-e 版本 使用LLM提取关键指标而不是直接让LLM理解log做出回答
-m 版本 直接将读取到的json赋值为metric，不需要LLM做提取工作
decoder.py 用于二进制和十进制之间的解码和编码

demo0 还停留在读取已准备好的日志
demo1 实现直接读取系统信息并存为日志用于后面的任务