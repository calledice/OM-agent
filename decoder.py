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
 
# 示例用法 
# 输出: 10 
# aa = decimal_to_binary(626724)
# print(decimal_to_binary(626724))
aa = binary_to_decimal("10011001000000100100"+"000000000000")
print(binary_to_decimal("10011001000000100100"+"000000000000"))  # 输出: 626724
print(decimal_to_binary(aa))


