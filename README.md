## 要求

使用 `run_qa.py` 中的 `evaluate` 函数，该函数输入为询问列表 `queries` ，该函数返回为对应的答案列表 。

## 实现方法

使用了openai提供的GPT接口来回答相关问题，避免使用国内的api来避开有关问题的屏蔽词

## 使用方法

命令行运行run_qa.py，需要手动选择需要测试的问题类型，目前可以分为简单问题（simple）与开放题（open）。后续可根据需要添加问题类型

```bash
python run_qa.py
```

输出为答案的列表
