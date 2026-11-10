from typing import Annotated

from langchain_core.tools import tool


@tool("calculate")
def calculate2 (a:Annotated[float, "第一个数字"],
                b:Annotated[float, "第二个数字"],
                operation:Annotated[str, "运算类型"])->float:
    """工具函数：计算两个数字的运算结果"""
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

#
# calculate2.invoke({"a": 1, "b": 2, "operation": "add"})





