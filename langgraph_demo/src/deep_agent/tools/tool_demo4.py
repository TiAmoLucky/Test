from typing import Annotated

from langchain_core.tools import tool, StructuredTool


def calculate4 (a:float,
                b:float,
                operation:str)->float:
    """工具函数：计算两个数字的运算结果

    Args:
        a: 第一个数字:
        b: 第二个数字:
        operation: 运算类型

    Returns:
        运算结果:
    """
    print(f"调用 calculate 工具，第一个数字：{a}，第二个数字：{b}，运算类型：{operation}")

    result = 0.0
    match operation:
        case "add":
            result = a + b
        case "subtract":
            result = a - b
        case "multiply":
            result = a * b
        case "divide":
            if b != 0:
                result = a / b
            else:
                raise ValueError("除数不能为零")

    return result


async def calculate5(a:float,
                b:float,
                operation:str)->float:
    """工具函数：计算两个数字的运算结果

    Args:
        a: 第一个数字:
        b: 第二个数字:
        operation: 运算类型

    Returns:
        运算结果:
    """
    print(f"调用 calculate 工具，第一个数字：{a}，第二个数字：{b}，运算类型：{operation}")

    result = 0.0
    match operation:
        case "add":
            result = a + b
        case "subtract":
            result = a - b
        case "multiply":
            result = a * b
        case "divide":
            if b != 0:
                result = a / b
            else:
                raise ValueError("除数不能为零")

    return result

# 创建了一个工具
calculator = StructuredTool.from_function(
    func=calculate4,
    name="calculator",
    description="工具函数：计算两个数字的运算结果",
    return_direct=False,
    coroutine=calculate5
)

print(calculator.name)
print(calculator.invoke({"a": 1, "b": 2, "operation": "add"}))





