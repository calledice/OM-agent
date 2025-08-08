def decimal_to_binary(decimal_num):
    """
    将十进制整数转换为二进制字符串 
    :param decimal_num: 十进制整数 
    :return: 二进制字符串 
    """
    if decimal_num == 0:
        return "0"
    
    binary_str = ""
    is_negative = False 
    
    # 处理负数 
    if decimal_num < 0:
        is_negative = True 
        decimal_num = abs(decimal_num)
    
    # 转换过程 
    while decimal_num > 0:
        remainder = decimal_num % 2 
        binary_str = str(remainder) + binary_str 
        decimal_num = decimal_num // 2 
    
    # 添加负号 
    if is_negative:
        binary_str = "-" + binary_str 
    
    return binary_str 
 


def binary_to_decimal(binary_str):
    """
    将二进制字符串转换为十进制整数 
    :param binary_str: 二进制字符串 
    :return: 十进制整数 
    """
    binary_str = binary_str.strip()   # 去除前后空格 
    
    if not binary_str:
        return 0 
    
    is_negative = False 
    
    # 处理负号 
    if binary_str[0] == '-':
        is_negative = True 
        binary_str = binary_str[1:]
    
    decimal_num = 0 
    length = len(binary_str)
    
    # 转换过程 
    for i in range(length):
        digit = binary_str[i]
        if digit not in ('0', '1'):
            raise ValueError("无效的二进制字符串")
        power = length - 1 - i 
        decimal_num += int(digit) * (2 ** power)
    
    # 添加负号 
    if is_negative:
        decimal_num = -decimal_num 
    
    return decimal_num 
def decimal_to_hex(decimal_num):
    """
    将十进制整数转换为十六进制字符串
    参数:
        decimal_num: 十进制整数（支持正负数）
    返回:
        带"0x"前缀的十六进制字符串（字母大写）
    """
    # 处理0的特殊情况 
    if decimal_num == 0:
        return "0x0"

    # 处理负数：取绝对值后添加负号
    is_negative = decimal_num < 0 
    num = abs(decimal_num)

    hex_chars = []  # 存储十六进制字符 
    hex_map = "0123456789ABCDEF"  # 映射关系 

    # 核心转换算法：反复除以16取余数 
    while num > 0:
        remainder = num % 16  # 获取当前余数（0-15）
        hex_chars.append(hex_map[remainder])   # 转换为十六进制字符 
        num = num // 16  # 更新为商 

    # 组合最终结果
    hex_str = '0x' + ''.join(hex_chars[::-1])  # 逆序拼接字符 
    return '-' + hex_str if is_negative else hex_str 
# 示例用法 
# 输出: 10 
# aa = decimal_to_binary(626724)
# print(decimal_to_binary(626724))
print("运维")
print(decimal_to_binary(2147483648))
print(decimal_to_hex(2147483648))
print(decimal_to_binary(2415956004))
print(decimal_to_hex(2415956004))
print(decimal_to_binary(2684355074))
print(decimal_to_hex(2684355074))
# print(decimal_to_binary(2684354560))
# print(len("10000000000000000000000000000000"))
# print("调控")
# print(decimal_to_binary(699048087))
# print(decimal_to_binary(699048103))
# print(len("00101001101010101010000010010111"))
# print("健康")
# print(decimal_to_binary(1073741824))
# print(decimal_to_binary(1342373900))
# print(len("01000000000000000000000000000000"))



# print(decimal_to_binary(1879011292))
# print(len(decimal_to_binary(1879011292)))
# print(decimal_to_binary(1610612736))
# print(len(decimal_to_binary(1610612736)))
