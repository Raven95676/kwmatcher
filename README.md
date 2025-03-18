# kwmatcher

多模式字符串匹配工具，带有简单的逻辑匹配。

> [!important]
> 本项目**大小写敏感**，请确保关键词规则与待检测文本大小写一致或进行预处理。
>
> 本项目**不支持分词**，请确保关键词规则与待检测文本已经过适当的预处理。

## 安装

```sh
pip install kwmatcher
```

## 使用方法

默认情况下启用逻辑表达式解析：

```python
from kwmatcher import AhoMatcher

matcher = AhoMatcher()
```

如需禁用，请传入`False`:

```python
from kwmatcher import AhoMatcher

matcher = AhoMatcher(use_logic=False)
```

build方法需要传入一个包含关键词的集合，find方法需要传入一个字符串，输出为一个包含匹配到的关键词的集合。

```python
patterns = {"A&B~C&D&E", "X&Y&Z~M&N"}
matcher.build(patterns)
result = matcher.find("AB")
print(result)  # 输出：{"A&B~C&D&E"}
```

当启用逻辑表达式解析时，使用`&`表示要求多个关键词同时出现，使用`~`表示排除包含特定关键词。排除条件组内部可用`&`连接，要求组内的所有关键词必须同时存在。

假设有如下关键词规则：

```
A&B~C&D&E~F&G&H&I&J
```

将被解析为两个组：

包含组：
 - {"A","B"}

排除组：
 - {"C","D","E"}
 - {"F","G","H","I","J"}

如果文本缺少"A"或"B"中的任意一个，匹配失败。

如果文本同时包含"C"、"D"、"E"全部三个，匹配失败。

如果文本同时包含"F"、"G"、"H"、"I"、"J" 全部五个，匹配失败。

在包含组都出现的情况下，只要任一排除组全部出现就匹配失败。
