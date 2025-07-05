"""
title: Advanced Context Manager - Recursive Summary Without Truncation
author: JiangNanGenius
Github: https://github.com/JiangNanGenius/Advanced-Context-Manager/
version: 7.0.0
license: MIT
required_open_webui_version: 0.4.0
"""

import json
import hashlib
import asyncio
from typing import Optional, List, Dict, Callable, Any, Tuple
from pydantic import BaseModel, Field

# 导入所需库
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None

try:
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None


class Filter:
    class Valves(BaseModel):
        enable_auto_summary: bool = Field(
            default=True, description="启用自动对话摘要功能"
        )

        # Debug设置
        debug_level: int = Field(
            default=1, description="调试级别：0=关闭，1=基础，2=详细，3=完整"
        )

        # Token管理
        total_token_limit: int = Field(
            default=40000, description="总token限制（建议设置为模型限制的60-70%）"
        )

        # 上下文保留策略 - 核心配置
        context_preserve_ratio: float = Field(
            default=0.7, description="上下文保留比例（0.7表示保留70%原文，30%总结）"
        )

        large_message_threshold: int = Field(
            default=10000,
            description="大消息阈值，超过此值的消息将应用智能内部保留策略",
        )

        max_recursion_depth: int = Field(
            default=3, description="最大递归总结深度，防止无限循环"
        )

        # 摘要配置
        max_summary_length: int = Field(default=3000, description="摘要的最大token长度")

        # 摘要模型配置
        summarizer_base_url: str = Field(
            default="https://ark.cn-beijing.volces.com/api/v3",
            description="摘要模型的API基础URL",
        )

        summarizer_api_key: str = Field(default="", description="摘要模型的API密钥")

        summarizer_model: str = Field(
            default="doubao-1-5-thinking-pro-250415", description="用于摘要的模型名称"
        )

        # 分片和并发配置
        max_chunk_tokens: int = Field(
            default=8000, description="每个分片的最大token数（摘要时）"
        )

        max_concurrent_requests: int = Field(default=3, description="最大并发请求数")

        max_summary_tokens_per_chunk: int = Field(
            default=500, description="每个分片摘要的最大token数"
        )

        summarizer_temperature: float = Field(
            default=0.2, description="摘要模型的温度参数"
        )

        request_timeout: int = Field(default=60, description="API请求超时时间（秒）")

    def __init__(self):
        self.valves = self.Valves()
        self.conversation_summaries = {}
        self._client = None
        self._encoding = None

    def debug_log(self, level: int, message: str):
        """分级debug日志"""
        if self.valves.debug_level >= level:
            prefix = ["", "[DEBUG]", "[DETAIL]", "[VERBOSE]"][min(level, 3)]
            print(f"{prefix} {message}")

    def get_encoding(self):
        """获取tiktoken编码器"""
        if not TIKTOKEN_AVAILABLE:
            return None
        if self._encoding is None:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        return self._encoding

    def count_tokens(self, text: str) -> int:
        """精确计算token数量"""
        if not text:
            return 0
        encoding = self.get_encoding()
        if encoding is None:
            return len(text) // 4
        try:
            return len(encoding.encode(text))
        except:
            return len(text) // 4

    def count_message_tokens(self, message: dict) -> int:
        """计算单个消息的token数"""
        content = message.get("content", "")
        role = message.get("role", "")
        total_tokens = 0

        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    text = item.get("text", "")
                    total_tokens += self.count_tokens(text)
        elif isinstance(content, str):
            total_tokens = self.count_tokens(content)

        total_tokens += self.count_tokens(role) + 4
        return total_tokens

    def count_messages_tokens(self, messages: List[dict]) -> int:
        """计算消息列表的总token数"""
        return sum(self.count_message_tokens(msg) for msg in messages)

    def analyze_messages(self, messages: List[dict], recursion_depth: int = 0) -> None:
        """详细分析消息token分布"""
        depth_prefix = "  " * recursion_depth
        self.debug_log(
            2, f"{depth_prefix}=== 消息分析 (递归深度: {recursion_depth}) ==="
        )
        total = 0
        user_total = 0
        assistant_total = 0
        system_total = 0

        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            tokens = self.count_message_tokens(msg)
            total += tokens

            if role == "user":
                user_total += tokens
            elif role == "assistant":
                assistant_total += tokens
            elif role == "system":
                system_total += tokens

            content_preview = content[:100] + "..." if len(content) > 100 else content
            self.debug_log(
                2,
                f"{depth_prefix}消息{i}: {role}, {tokens}tokens, 内容: {content_preview}",
            )

        self.debug_log(
            2,
            f"{depth_prefix}总计: {total}tokens (系统: {system_total}, 用户: {user_total}, 助手: {assistant_total})",
        )
        self.debug_log(2, f"{depth_prefix}=== 分析结束 ===")

    def smart_split_large_message(
        self, message: dict, preserve_tokens: int
    ) -> Tuple[dict, str]:
        """
        智能分割大消息：保留70%原文，30%作为摘要内容
        返回: (保留部分的消息, 需要摘要的内容)
        """
        content = message.get("content", "")
        if not isinstance(content, str):
            return message, ""

        msg_tokens = self.count_message_tokens(message)
        if msg_tokens <= preserve_tokens:
            return message, ""

        self.debug_log(
            1, f"智能分割大消息：{msg_tokens}tokens，保留{preserve_tokens}tokens原文"
        )

        # 计算要保留的内容长度
        encoding = self.get_encoding()

        if encoding is None:
            # 简单字符分割
            preserve_ratio = preserve_tokens / msg_tokens
            preserve_chars = int(len(content) * preserve_ratio)

            # 从后面保留（通常最新的内容更重要）
            if preserve_chars < len(content):
                preserved_content = content[-preserve_chars:]
                to_summarize_content = content[:-preserve_chars]
            else:
                preserved_content = content
                to_summarize_content = ""
        else:
            # 精确token分割
            tokens = encoding.encode(content)
            preserve_token_count = preserve_tokens - 10  # 预留空间给role等

            if preserve_token_count < len(tokens):
                # 从后面保留
                preserved_tokens = tokens[-preserve_token_count:]
                to_summarize_tokens = tokens[:-preserve_token_count]

                try:
                    preserved_content = encoding.decode(preserved_tokens)
                    to_summarize_content = encoding.decode(to_summarize_tokens)
                except:
                    # 如果解码失败，使用字符分割
                    preserve_ratio = preserve_tokens / msg_tokens
                    preserve_chars = int(len(content) * preserve_ratio)
                    preserved_content = content[-preserve_chars:]
                    to_summarize_content = content[:-preserve_chars]
            else:
                preserved_content = content
                to_summarize_content = ""

        # 创建保留的消息
        preserved_message = message.copy()
        preserved_message["content"] = preserved_content

        preserved_tokens = self.count_message_tokens(preserved_message)
        to_summarize_tokens = self.count_tokens(to_summarize_content)

        self.debug_log(
            1,
            f"大消息分割完成：保留{preserved_tokens}tokens，待摘要{to_summarize_tokens}tokens",
        )

        return preserved_message, to_summarize_content

    def get_openai_client(self):
        """获取OpenAI客户端"""
        if not OPENAI_AVAILABLE:
            raise Exception("OpenAI库未安装，请执行: pip install openai")
        if not self.valves.summarizer_api_key:
            raise Exception("未配置摘要API密钥")

        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.valves.summarizer_base_url,
                api_key=self.valves.summarizer_api_key,
                timeout=self.valves.request_timeout,
            )
        return self._client

    def extract_new_session(
        self, messages: List[dict]
    ) -> Tuple[List[dict], List[dict], List[dict]]:
        """
        提取新会话（最后一条用户消息及其后的所有消息）
        返回: (系统消息, 历史消息, 新会话消息)
        """
        if not messages:
            return [], [], []

        # 分离系统消息
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        conversation_messages = [msg for msg in messages if msg.get("role") != "system"]

        self.debug_log(
            2,
            f"系统消息: {len(system_messages)}条, 对话消息: {len(conversation_messages)}条",
        )

        if not conversation_messages:
            return system_messages, [], []

        # 找到最后一条用户消息的位置
        last_user_idx = -1
        for i in range(len(conversation_messages) - 1, -1, -1):
            if conversation_messages[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx == -1:
            # 没有用户消息，全部作为历史
            self.debug_log(2, "未找到用户消息，全部作为历史")
            return system_messages, conversation_messages, []

        # 分割历史和新会话
        history_messages = conversation_messages[:last_user_idx]
        new_session_messages = conversation_messages[last_user_idx:]

        self.debug_log(
            2,
            f"新会话提取：历史{len(history_messages)}条，新会话{len(new_session_messages)}条",
        )

        return system_messages, history_messages, new_session_messages

    def smart_preserve_context_with_large_message_handling(
        self, history_messages: List[dict], available_tokens: int
    ) -> Tuple[List[dict], List[dict], List[str]]:
        """
        智能保留上下文，包含大消息处理
        返回: (保留的原文消息, 需要摘要的完整消息, 需要摘要的大消息片段)
        """
        if not history_messages or available_tokens <= 0:
            return [], history_messages, []

        history_tokens = self.count_messages_tokens(history_messages)
        self.debug_log(
            1, f"历史内容: {history_tokens}tokens, 可用空间: {available_tokens}tokens"
        )

        if history_tokens <= available_tokens:
            # 历史内容完全放得下，无需总结
            self.debug_log(1, "历史内容完全适合，无需总结")
            return history_messages, [], []

        # 计算保留的token数
        preserve_tokens = int(available_tokens * self.valves.context_preserve_ratio)
        summary_budget = available_tokens - preserve_tokens

        self.debug_log(
            1,
            f"按{self.valves.context_preserve_ratio:.1%}比例保留：保留{preserve_tokens}tokens原文，{summary_budget}tokens用于总结",
        )

        # 检查是否有大消息需要特殊处理
        large_message_fragments = []
        preserved_messages = []
        to_summarize_messages = []
        current_preserve_tokens = 0

        # 从后往前处理消息
        for i in range(len(history_messages) - 1, -1, -1):
            msg = history_messages[i]
            msg_tokens = self.count_message_tokens(msg)

            # 检查是否是大消息
            if (
                msg_tokens > self.valves.large_message_threshold
                and current_preserve_tokens < preserve_tokens
            ):
                # 对大消息进行智能分割
                remaining_preserve_tokens = preserve_tokens - current_preserve_tokens
                if remaining_preserve_tokens > 0:
                    self.debug_log(
                        1,
                        f"处理大消息({msg_tokens}tokens)，剩余保留空间: {remaining_preserve_tokens}tokens",
                    )

                    preserved_part, to_summarize_part = self.smart_split_large_message(
                        msg, remaining_preserve_tokens
                    )

                    if preserved_part:
                        preserved_messages.insert(0, preserved_part)
                        current_preserve_tokens += self.count_message_tokens(
                            preserved_part
                        )

                    if to_summarize_part:
                        large_message_fragments.append(to_summarize_part)

                    # 将前面的消息加入摘要队列
                    to_summarize_messages = history_messages[:i] + to_summarize_messages
                    break
                else:
                    # 保留空间已满，全部摘要
                    to_summarize_messages = (
                        history_messages[: i + 1] + to_summarize_messages
                    )
                    break
            else:
                # 普通消息
                if current_preserve_tokens + msg_tokens <= preserve_tokens:
                    preserved_messages.insert(0, msg)
                    current_preserve_tokens += msg_tokens
                else:
                    # 超出保留空间，剩余的全部摘要
                    to_summarize_messages = history_messages[: i + 1]
                    break

        preserved_tokens = self.count_messages_tokens(preserved_messages)
        to_summarize_tokens = self.count_messages_tokens(to_summarize_messages)
        fragment_tokens = sum(
            self.count_tokens(frag) for frag in large_message_fragments
        )

        self.debug_log(
            1,
            f"智能保留结果：保留{len(preserved_messages)}条消息({preserved_tokens}tokens)，"
            + f"摘要{len(to_summarize_messages)}条消息({to_summarize_tokens}tokens)，"
            + f"大消息片段{len(large_message_fragments)}个({fragment_tokens}tokens)",
        )

        return preserved_messages, to_summarize_messages, large_message_fragments

    def split_large_message_for_summary(
        self, message: dict, max_tokens: int
    ) -> List[str]:
        """
        将过大的消息分割成多个片段用于摘要（保持完整内容）
        返回: 分割后的内容片段列表
        """
        content = message.get("content", "")
        if not isinstance(content, str):
            return [content]

        tokens = self.count_message_tokens(message)
        if tokens <= max_tokens:
            return [content]

        self.debug_log(
            2, f"分割大消息用于摘要：{tokens}tokens → 多个{max_tokens}tokens片段"
        )

        encoding = self.get_encoding()
        if encoding is None:
            # 简单字符分割
            chunk_size = max_tokens * 3  # 粗略估算
            chunks = []
            for i in range(0, len(content), chunk_size):
                chunks.append(content[i : i + chunk_size])
            return chunks
        else:
            # 精确token分割
            encoded_tokens = encoding.encode(content)
            chunks = []

            for i in range(0, len(encoded_tokens), max_tokens - 100):  # 预留空间
                chunk_tokens = encoded_tokens[i : i + max_tokens - 100]
                try:
                    chunk_text = encoding.decode(chunk_tokens)
                    chunks.append(chunk_text)
                except:
                    # 如果解码失败，使用字符分割
                    start_pos = i * 3
                    end_pos = min((i + max_tokens - 100) * 3, len(content))
                    chunks.append(content[start_pos:end_pos])

            return chunks

    def create_summary_chunks_with_fragments(
        self, messages: List[dict], fragments: List[str]
    ) -> List[dict]:
        """创建用于摘要的chunks，包含大消息片段"""
        chunks = []

        # 处理完整消息
        if messages:
            self.debug_log(2, f"处理{len(messages)}条完整消息用于摘要")
            current_chunk = []
            current_tokens = 0
            max_tokens = self.valves.max_chunk_tokens - 1000

            for i, msg in enumerate(messages):
                msg_tokens = self.count_message_tokens(msg)

                if msg_tokens > max_tokens:
                    # 大消息需要分割
                    if current_chunk:
                        chunks.append({"type": "messages", "content": current_chunk})
                        current_chunk = []
                        current_tokens = 0

                    # 分割大消息
                    content_chunks = self.split_large_message_for_summary(
                        msg, max_tokens
                    )
                    role = msg.get("role", "unknown")

                    for content_chunk in content_chunks:
                        chunks.append(
                            {
                                "type": "message_fragment",
                                "role": role,
                                "content": content_chunk,
                            }
                        )
                else:
                    if current_tokens + msg_tokens > max_tokens and current_chunk:
                        chunks.append({"type": "messages", "content": current_chunk})
                        current_chunk = [msg]
                        current_tokens = msg_tokens
                    else:
                        current_chunk.append(msg)
                        current_tokens += msg_tokens

            if current_chunk:
                chunks.append({"type": "messages", "content": current_chunk})

        # 处理大消息片段
        for fragment in fragments:
            chunks.append({"type": "large_fragment", "content": fragment})

        self.debug_log(1, f"创建摘要chunks完成，共{len(chunks)}个片段")
        return chunks

    async def update_progress(
        self,
        __event_emitter__,
        current: int,
        total: int,
        stage: str,
        recursion_depth: int = 0,
    ):
        """更新进度显示"""
        if __event_emitter__ and total > 0:
            percentage = int((current / total) * 100)
            depth_info = f" (递归: {recursion_depth})" if recursion_depth > 0 else ""
            await self.send_status(
                __event_emitter__,
                f"{stage}{depth_info} - 进度: {current}/{total} ({percentage}%)",
                False,
            )

    async def summarize_chunk_async_with_progress(
        self,
        chunk_data: dict,
        chunk_index: int,
        total_chunks: int,
        summary_token_budget: int,
        semaphore: asyncio.Semaphore,
        __event_emitter__,
        recursion_depth: int = 0,
    ) -> Tuple[int, str]:
        """异步摘要单个chunk，包含进度更新"""
        async with semaphore:
            try:
                # 更新进度
                await self.update_progress(
                    __event_emitter__,
                    chunk_index + 1,
                    total_chunks,
                    "正在摘要",
                    recursion_depth,
                )

                client = self.get_openai_client()

                # 根据chunk类型构建对话文本
                chunk_type = chunk_data.get("type", "messages")

                if chunk_type == "messages":
                    # 处理完整消息列表
                    conversation_text = ""
                    for msg in chunk_data["content"]:
                        role = msg.get("role", "unknown")
                        content = str(msg.get("content", ""))
                        conversation_text += f"{role}: {content}\n\n"
                    context_info = (
                        f"这是一段包含{len(chunk_data['content'])}条消息的对话"
                    )

                elif chunk_type == "message_fragment":
                    # 处理消息片段
                    role = chunk_data.get("role", "unknown")
                    content = chunk_data.get("content", "")
                    conversation_text = f"{role}: {content}\n\n"
                    context_info = f"这是一个{role}消息的片段"

                elif chunk_type == "large_fragment":
                    # 处理大消息片段
                    content = chunk_data.get("content", "")
                    conversation_text = f"content: {content}\n\n"
                    context_info = "这是一个大消息的片段"

                else:
                    conversation_text = str(chunk_data)
                    context_info = "未知类型的内容"

                # 计算这个chunk的摘要token限制
                max_tokens_per_chunk = min(
                    self.valves.max_summary_tokens_per_chunk,
                    (
                        max(300, summary_token_budget // total_chunks)
                        if total_chunks > 0
                        else 500
                    ),
                )

                recursion_info = (
                    f"（递归深度: {recursion_depth}）" if recursion_depth > 0 else ""
                )

                system_prompt = f"""你是专业的对话摘要专家。请为这段内容创建简洁但完整的摘要（第{chunk_index + 1}部分，共{total_chunks}部分）{recursion_info}。

内容说明：{context_info}

摘要要求：
1. 保留所有重要信息、关键决定和讨论要点
2. 如果涉及技术内容、数据或代码，务必保留核心信息
3. 保持内容的逻辑流程和因果关系
4. 如果内容很多，请分要点总结，确保不遗漏重要信息
5. 摘要长度控制在{max_tokens_per_chunk}字以内
6. 使用简洁准确的语言，但不能丢失重要细节

内容："""

                user_prompt = f"请根据要求摘要以下内容：\n\n{conversation_text}"

                response = await client.chat.completions.create(
                    model=self.valves.summarizer_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens_per_chunk,
                    temperature=self.valves.summarizer_temperature,
                    stream=False,
                )

                if response.choices and len(response.choices) > 0:
                    summary = response.choices[0].message.content.strip()
                    self.debug_log(
                        3, f"Chunk {chunk_index + 1}摘要完成: {len(summary)}字符"
                    )
                    return chunk_index, summary
                else:
                    return chunk_index, f"第{chunk_index + 1}部分摘要失败（无响应）"

            except Exception as e:
                self.debug_log(1, f"摘要chunk {chunk_index + 1}时出错: {str(e)}")
                return (
                    chunk_index,
                    f"第{chunk_index + 1}部分摘要失败: {str(e)[:100]}...",
                )
            finally:
                # 避免请求过于频繁
                await asyncio.sleep(0.3)

    async def create_comprehensive_summary_with_progress(
        self,
        messages_to_summarize: List[dict],
        large_fragments: List[str],
        summary_token_budget: int,
        __event_emitter__,
        recursion_depth: int = 0,
    ) -> str:
        """创建comprehensive摘要，包含进度显示"""
        if not messages_to_summarize and not large_fragments:
            return ""

        # 创建摘要chunks
        chunks = self.create_summary_chunks_with_fragments(
            messages_to_summarize, large_fragments
        )
        if not chunks:
            return ""

        original_tokens = self.count_messages_tokens(messages_to_summarize)
        fragment_tokens = sum(self.count_tokens(frag) for frag in large_fragments)
        total_original_tokens = original_tokens + fragment_tokens

        depth_info = f" (递归: {recursion_depth})" if recursion_depth > 0 else ""
        await self.send_status(
            __event_emitter__,
            f"开始摘要 {len(chunks)} 个片段（原始{total_original_tokens}tokens，保留{self.valves.context_preserve_ratio:.1%}原文）{depth_info}",
            False,
        )

        # 并发摘要处理
        semaphore = asyncio.Semaphore(self.valves.max_concurrent_requests)
        tasks = []

        for i, chunk in enumerate(chunks):
            task = self.summarize_chunk_async_with_progress(
                chunk,
                i,
                len(chunks),
                summary_token_budget,
                semaphore,
                __event_emitter__,
                recursion_depth,
            )
            tasks.append(task)

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            await self.send_status(__event_emitter__, f"摘要处理出错: {str(e)}")
            return "摘要处理失败"

        # 处理结果
        summaries = {}
        successful_count = 0

        for result in results:
            if isinstance(result, Exception):
                self.debug_log(1, f"摘要任务异常: {result}")
                continue

            chunk_index, summary = result
            summaries[chunk_index] = summary
            successful_count += 1

        await self.update_progress(
            __event_emitter__,
            successful_count,
            len(chunks),
            "摘要完成",
            recursion_depth,
        )
        self.debug_log(1, f"摘要完成：{successful_count}/{len(chunks)}个片段成功")

        # 按顺序组合摘要
        ordered_summaries = []
        for i in range(len(chunks)):
            if i in summaries:
                ordered_summaries.append(summaries[i])

        if not ordered_summaries:
            return "所有摘要任务都失败了"

        # 组合最终摘要
        if len(ordered_summaries) == 1:
            final_summary = ordered_summaries[0]
        else:
            final_summary = "\n\n".join(
                f"【第{i+1}部分摘要】\n{summary}"
                for i, summary in enumerate(ordered_summaries)
            )

        summary_tokens = self.count_tokens(final_summary)
        self.debug_log(
            1,
            f"最终摘要：{summary_tokens}tokens，原始内容：{total_original_tokens}tokens",
        )

        return final_summary

    async def recursive_context_processing(
        self, messages: List[dict], __event_emitter__, recursion_depth: int = 0
    ) -> List[dict]:
        """
        递归上下文处理：如果结果超限则再次执行总结流程
        返回: 最终处理后的消息列表
        """
        if recursion_depth >= self.valves.max_recursion_depth:
            self.debug_log(
                1, f"达到最大递归深度 {self.valves.max_recursion_depth}，停止处理"
            )
            await self.send_status(__event_emitter__, f"达到最大递归深度，停止处理")
            return messages

        if not messages:
            return messages

        total_tokens = self.count_messages_tokens(messages)
        self.debug_log(
            1,
            f"递归处理 (深度: {recursion_depth})：{total_tokens}tokens，限制: {self.valves.total_token_limit}",
        )

        # 详细分析消息
        if self.valves.debug_level >= 2:
            self.analyze_messages(messages, recursion_depth)

        # 检查是否需要处理
        if total_tokens <= self.valves.total_token_limit:
            self.debug_log(1, f"递归深度 {recursion_depth}：内容未超限，处理完成")
            if recursion_depth > 0:
                await self.send_status(
                    __event_emitter__, f"递归总结完成 (深度: {recursion_depth})"
                )
            return messages

        # 超限处理
        if recursion_depth > 0:
            await self.send_status(
                __event_emitter__,
                f"第{recursion_depth}次递归：内容仍超限({total_tokens}>{self.valves.total_token_limit})，继续总结...",
                False,
            )

        # 提取新会话
        system_messages, history_messages, new_session_messages = (
            self.extract_new_session(messages)
        )

        # 计算token分布
        system_tokens = self.count_messages_tokens(system_messages)
        history_tokens = self.count_messages_tokens(history_messages)
        new_session_tokens = self.count_messages_tokens(new_session_messages)

        self.debug_log(
            1,
            f"递归 {recursion_depth} Token分布 - 系统: {system_tokens}, 历史: {history_tokens}, 新会话: {new_session_tokens}",
        )

        # 计算历史内容可用空间
        available_for_history = (
            self.valves.total_token_limit - system_tokens - new_session_tokens
        )

        if available_for_history <= 0:
            self.debug_log(
                1, f"递归 {recursion_depth}：新会话已占满所有空间，历史内容将全部摘要"
            )
            summary_budget = min(
                self.valves.max_summary_length, self.valves.total_token_limit // 10
            )
            final_messages = system_messages + new_session_messages

            if history_messages:
                # 对历史内容进行摘要
                summary = await self.create_comprehensive_summary_with_progress(
                    history_messages,
                    [],
                    summary_budget,
                    __event_emitter__,
                    recursion_depth,
                )

                if summary and summary.strip():
                    summary_content = (
                        f"=== 历史对话摘要 (递归: {recursion_depth}) ===\n{summary}"
                    )

                    # 添加摘要到系统消息
                    system_msg_found = False
                    for msg in final_messages:
                        if msg.get("role") == "system":
                            msg["content"] = f"{msg['content']}\n\n{summary_content}"
                            system_msg_found = True
                            break

                    if not system_msg_found:
                        final_messages.insert(
                            0, {"role": "system", "content": summary_content}
                        )

            # 检查结果是否还超限
            result_tokens = self.count_messages_tokens(final_messages)
            if result_tokens > self.valves.total_token_limit:
                # 递归处理
                return await self.recursive_context_processing(
                    final_messages, __event_emitter__, recursion_depth + 1
                )
            else:
                return final_messages

        self.debug_log(
            1,
            f"递归 {recursion_depth}：历史内容可用空间: {available_for_history}tokens",
        )

        # 智能保留历史内容
        preserved_history, to_summarize, large_fragments = (
            self.smart_preserve_context_with_large_message_handling(
                history_messages, available_for_history
            )
        )

        # 组装结果
        final_messages = system_messages + preserved_history + new_session_messages

        if to_summarize or large_fragments:
            # 有内容需要摘要
            preserved_tokens = self.count_messages_tokens(preserved_history)
            summary_budget = available_for_history - preserved_tokens
            summary_budget = min(summary_budget, self.valves.max_summary_length)

            total_items = len(to_summarize) + len(large_fragments)
            await self.send_status(
                __event_emitter__,
                f"递归 {recursion_depth}：开始摘要 {total_items} 项内容，保留 {self.valves.context_preserve_ratio:.1%} 原文...",
                False,
            )

            summary = await self.create_comprehensive_summary_with_progress(
                to_summarize,
                large_fragments,
                summary_budget,
                __event_emitter__,
                recursion_depth,
            )

            if summary and summary.strip() and summary != "所有摘要任务都失败了":
                # 将摘要添加到系统消息
                to_summarize_tokens = self.count_messages_tokens(to_summarize)
                fragment_tokens = sum(
                    self.count_tokens(frag) for frag in large_fragments
                )
                total_summarized_tokens = to_summarize_tokens + fragment_tokens

                summary_content = f"=== 历史对话摘要 (递归: {recursion_depth}, {len(to_summarize)}条消息+{len(large_fragments)}个片段，{total_summarized_tokens}tokens原始内容，保留{self.valves.context_preserve_ratio:.1%}原文) ===\n{summary}"

                # 添加摘要到系统消息
                system_msg_found = False
                for msg in final_messages:
                    if msg.get("role") == "system":
                        if (
                            f"=== 历史对话摘要 (递归: {recursion_depth}"
                            not in msg["content"]
                        ):
                            msg["content"] = f"{msg['content']}\n\n{summary_content}"
                        system_msg_found = True
                        break

                if not system_msg_found:
                    final_messages.insert(
                        0, {"role": "system", "content": summary_content}
                    )

        # 检查最终结果是否还超限
        result_tokens = self.count_messages_tokens(final_messages)
        self.debug_log(
            1,
            f"递归 {recursion_depth} 处理后: {len(final_messages)} 条消息, {result_tokens} tokens",
        )

        if result_tokens > self.valves.total_token_limit:
            self.debug_log(1, f"递归 {recursion_depth} 结果仍超限，继续下一轮处理")
            # 递归处理
            return await self.recursive_context_processing(
                final_messages, __event_emitter__, recursion_depth + 1
            )
        else:
            self.debug_log(1, f"递归 {recursion_depth} 处理成功，结果在限制内")
            return final_messages

    def get_chat_id(self, __event_emitter__) -> Optional[str]:
        """提取聊天ID"""
        try:
            if (
                hasattr(__event_emitter__, "__closure__")
                and __event_emitter__.__closure__
            ):
                info = __event_emitter__.__closure__[0].cell_contents
                return info.get("chat_id")
        except:
            pass
        return None

    async def send_status(self, __event_emitter__, message: str, done: bool = True):
        """发送状态消息"""
        if __event_emitter__:
            try:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": message,
                            "done": done,
                        },
                    }
                )
            except:
                pass

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable] = None,
    ) -> dict:
        """主要处理逻辑"""
        if not self.valves.enable_auto_summary:
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        # 检查依赖
        if not OPENAI_AVAILABLE:
            await self.send_status(__event_emitter__, "错误：需要安装OpenAI库")
            return body

        if not self.valves.summarizer_api_key:
            await self.send_status(__event_emitter__, "错误：未配置摘要API密钥")
            return body

        # 计算原始总token数
        total_tokens = self.count_messages_tokens(messages)
        self.debug_log(
            1, f"当前总token数: {total_tokens}, 限制: {self.valves.total_token_limit}"
        )

        # 检查是否需要处理
        if total_tokens <= self.valves.total_token_limit:
            self.debug_log(1, "内容未超限，无需处理")
            return body

        # 超限处理 - 使用递归总结策略
        chat_id = (
            self.get_chat_id(__event_emitter__) or f"chat_{hash(str(messages[:2]))}"
        )

        await self.send_status(
            __event_emitter__,
            f"内容超限({total_tokens}>{self.valves.total_token_limit})，启动递归总结策略...",
            False,
        )

        try:
            # 递归上下文处理
            final_messages = await self.recursive_context_processing(
                messages, __event_emitter__
            )

            # 应用处理结果
            body["messages"] = final_messages
            final_tokens = self.count_messages_tokens(final_messages)

            await self.send_status(
                __event_emitter__,
                f"递归总结完成：{len(messages)}→{len(final_messages)}条消息，{total_tokens}→{final_tokens}tokens",
            )

            self.debug_log(
                1, f"最终结果: {len(final_messages)} 条消息, {final_tokens} tokens"
            )

            # 保存摘要记录
            if chat_id:
                summary_msgs = [
                    msg
                    for msg in final_messages
                    if msg.get("role") == "system"
                    and "=== 历史对话摘要" in msg.get("content", "")
                ]
                if summary_msgs:
                    self.conversation_summaries[chat_id] = summary_msgs[-1]["content"]

        except Exception as e:
            await self.send_status(__event_emitter__, f"递归总结失败: {str(e)}")
            self.debug_log(1, f"错误: {str(e)}")
            if self.valves.debug_level >= 2:
                import traceback

                traceback.print_exc()

        return body

    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable] = None,
    ) -> dict:
        """输出处理"""
        return body
