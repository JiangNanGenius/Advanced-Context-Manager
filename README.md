# Advanced Context Manager

## 简介
这个OpenWebUI插件通过递归总结技术智能管理长对话上下文，避免传统截断方法导致的信息丢失。当对话长度超出token限制时，插件会：
1. 自动识别需要保留的关键内容
2. 对大消息进行智能分割处理
3. 使用AI模型生成保留核心信息的摘要
4. 递归应用总结策略直到满足token限制

## Features
- **递归总结策略**：多层深度总结保证内容完整性
- **大消息智能处理**：自动分割超大消息片段
- **精确Token计算**：支持tiktoken精确计数或字符估算
- **进度实时反馈**：可视化摘要生成过程
- **并发处理优化**：并行处理多个消息片段
- **上下文保留比例**：自定义原文保留比例(默认70%)
- **调试模式**：多级调试信息输出

## 安装方法
```bash
# 确保已安装所需依赖
pip install openai tiktoken

# 在OpenWebUI中直接导入插件函数
# 将插件文件保存为 context_manager.py 并放入OpenWebUI插件目录
```

## 配置参数(Valves)
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_auto_summary` | True | 启用自动对话摘要 |
| `debug_level` | 1 | 调试级别(0-3) |
| `total_token_limit` | 40000 | 总token限制 |
| `context_preserve_ratio` | 0.7 | 原文保留比例 |
| `large_message_threshold` | 10000 | 大消息阈值 |
| `max_recursion_depth` | 3 | 最大递归深度 |
| `max_summary_length` | 3000 | 摘要最大长度 |
| `summarizer_base_url` | - | 摘要API地址 |
| `summarizer_api_key` | - | 摘要API密钥 |
| `summarizer_model` | doubao-1-5-thinking-pro-250415 | 摘要模型 |
| `max_chunk_tokens` | 8000 | 分片最大token数 |
| `max_concurrent_requests` | 3 | 最大并发请求数 |
| `max_summary_tokens_per_chunk` | 500 | 分片摘要最大token数 |

## 使用示例
```python
# 初始化插件
manager = Filter()

# 配置参数
manager.valves.summarizer_api_key = "your_api_key_here"
manager.valves.summarizer_base_url = "https://your.summary.api"

# 处理消息
processed_messages = await manager.inlet({
    "messages": long_conversation_history
})
```

## 工作原理
1. **Token计算**：精确计算对话token总量
2. **会话分割**：分离系统消息/历史消息/新会话
3. **智能保留**：按比例保留关键原文内容
4. **大消息处理**：分割超大消息片段
5. **并发摘要**：并行处理多个消息片段
6. **递归优化**：多层总结直到满足token限制
7. **结果整合**：将摘要以系统消息形式插入

## 注意事项
1. 必须安装`openai`和`tiktoken`包
2. 需要有效的摘要API密钥
3. 递归深度限制防止无限循环
4. 大消息处理保留最新内容(通常最重要)
5. 调试级别2+可查看详细token分布

---

# Advanced Context Manager - Recursive Summary Without Truncation

## Introduction
This OpenWebUI plugin intelligently manages long conversation contexts using recursive summarization techniques, avoiding information loss caused by traditional truncation methods. When conversations exceed token limits, the plugin:
1. Automatically identifies key content to preserve
2. Intelligently splits large messages
3. Generates summaries retaining core information using AI models
4. Recursively applies summarization until token limits are satisfied

## Features
- **Recursive Summarization**: Multi-layer summarization preserves content integrity
- **Large Message Handling**: Automatic segmentation of oversized messages
- **Precise Token Counting**: Supports tiktoken or character estimation
- **Real-time Progress**: Visual feedback during summary generation
- **Concurrent Processing**: Parallel handling of message segments
- **Context Preservation Ratio**: Customizable original content retention (default 70%)
- **Debug Mode**: Multi-level debug information output

## Installation
```bash
# Ensure required dependencies are installed
pip install openai tiktoken

# Directly import plugin functions in OpenWebUI
# Save plugin file as context_manager.py in OpenWebUI plugins directory
```

## Configuration Parameters (Valves)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_auto_summary` | True | Enable automatic conversation summarization |
| `debug_level` | 1 | Debug level (0-3) |
| `total_token_limit` | 40000 | Total token limit |
| `context_preserve_ratio` | 0.7 | Original content retention ratio |
| `large_message_threshold` | 10000 | Large message threshold |
| `max_recursion_depth` | 3 | Maximum recursion depth |
| `max_summary_length` | 3000 | Maximum summary length |
| `summarizer_base_url` | - | Summary API URL |
| `summarizer_api_key` | - | Summary API key |
| `summarizer_model` | doubao-1-5-thinking-pro-250415 | Summarization model |
| `max_chunk_tokens` | 8000 | Maximum tokens per chunk |
| `max_concurrent_requests` | 3 | Maximum concurrent requests |
| `max_summary_tokens_per_chunk` | 500 | Maximum tokens per chunk summary |

## Usage Example
```python
# Initialize plugin
manager = Filter()

# Configure parameters
manager.valves.summarizer_api_key = "your_api_key_here"
manager.valves.summarizer_base_url = "https://your.summary.api"

# Process messages
processed_messages = await manager.inlet({
    "messages": long_conversation_history
})
```

## How It Works
1. **Token Calculation**: Precisely calculates total conversation tokens
2. **Session Segmentation**: Separates system messages/history/new session
3. **Intelligent Preservation**: Retains key original content by ratio
4. **Large Message Handling**: Splits oversized message segments
5. **Concurrent Summarization**: Parallel processing of message segments
6. **Recursive Optimization**: Multi-layer summarization until token limits met
7. **Result Integration**: Inserts summaries as system messages

## Notes
1. Requires `openai` and `tiktoken` packages
2. Valid summary API key required
3. Recursion depth limit prevents infinite loops
4. Large message processing retains newest content (typically most important)
5. Debug level 2+ shows detailed token distribution
