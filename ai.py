#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
智谱AI大模型LangChain封装实现
支持同步/流式调用、多轮对话和参数配置
"""

from typing import Any, Dict, Iterator, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    BaseMessage,
    SystemMessage,
    AIMessageChunk,
)
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic import Field, ConfigDict, SecretStr
from dotenv import load_dotenv
import os
import argparse
from zhipuai import ZhipuAI
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

class ZhipuModel(BaseChatModel):
    """智谱AI大模型LangChain封装
    
    特性：
    - 支持GLM-4/GLM-3等系列模型
    - 完整的消息类型转换
    - 同步/流式双模式调用
    - 完善的错误处理和日志记录
    - 安全的API密钥管理
    """
    
    model: str = Field(
        default="glm-4",
        description="模型名称，如glm-4, glm-3-turbo等"
    )
    temperature: float = Field(
        default=0.7,
        ge=0,
        le=1,
        description="生成温度，0-1之间"
    )
    api_key: Optional[SecretStr] = Field(
        default=None,
        exclude=True,
        description="API密钥，可从环境变量ZHIPU_API_KEY读取"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="最大输出token数"
    )
    top_p: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="核采样概率"
    )
    request_timeout: Optional[int] = Field(
        default=60,
        description="API请求超时时间(秒)"
    )
    
    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=()
    )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """同步生成方法"""
        try:
            formatted_messages = self._convert_messages(messages)
            client = self._get_client()
            
            logger.info(f"调用模型: {self.model}, 温度: {self.temperature}")
            
            response = client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=self.temperature,
                stop=stop,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                timeout=self.request_timeout,
                **kwargs
            )
            
            return self._create_chat_result(response)
            
        except Exception as e:
            logger.error(f"生成请求失败: {str(e)}", exc_info=True)
            raise

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,  
    ) -> Iterator[ChatGenerationChunk]:
        """流式生成方法"""
        try:
            formatted_messages = self._convert_messages(messages)
            client = self._get_client()
            
            logger.info(f"流式调用模型: {self.model}")
            
            stream = client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=self.temperature,
                stop=stop,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                timeout=self.request_timeout,
                stream=True,
                **kwargs
            )
            
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    chunk = ChatGenerationChunk(
                        message=AIMessageChunk(content=delta)
                    )
                    
                    if run_manager:
                        run_manager.on_llm_new_token(delta)
                    
                    yield chunk
                    
        except Exception as e:
            logger.error(f"流式请求失败: {str(e)}", exc_info=True)
            raise

    def _convert_messages(
        self, 
        messages: List[BaseMessage]
    ) -> List[Dict[str, str]]:
        """将LangChain消息转换为智谱API格式"""
        converted = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, SystemMessage):
                role = "system"
            else:
                logger.warning(f"忽略不支持的消息类型: {type(msg)}")
                continue
                
            converted.append({
                "role": role,
                "content": msg.content
            })
            
        logger.debug(f"转换后的消息: {converted}")
        return converted

    def _get_client(self) -> ZhipuAI:
        """获取API客户端实例"""
        return ZhipuAI(api_key=self._get_api_key())

    def _get_api_key(self) -> str:
        """安全获取API密钥"""
        if self.api_key:
            return self.api_key.get_secret_value()
            
        if key := os.getenv("ZHIPU_API_KEY"):
            return key
            
        raise ValueError(
            "未提供API密钥，请通过以下方式之一设置:\n"
            "1. 构造函数参数: ZhipuModel(api_key='your_key')\n"
            "2. 环境变量: export ZHIPU_API_KEY='your_key'"
        )

    def _create_chat_result(self, response) -> ChatResult:
        """构造LangChain返回结果"""
        if not response.choices:
            raise ValueError("API返回空响应")
            
        message = AIMessage(
            content=response.choices[0].message.content
        )
        return ChatResult(
            generations=[ChatGeneration(message=message)]
        )

    @property
    def _llm_type(self) -> str:
        """返回LLM类型标识"""
        return "zhipuai"

def main():
    """命令行交互入口"""
    parser = argparse.ArgumentParser(
        description="智谱AI大模型交互终端",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        default="glm-4",
        help="模型名称"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="生成温度(0-1)"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="使用流式输出模式"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="最大输出token数"
    )
    args = parser.parse_args()

    try:
        # 初始化模型
        model = ZhipuModel(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        # 初始化对话
        messages = [
            SystemMessage(content="你是一个有帮助的AI助手，回答要简明扼要"),
            HumanMessage(content="你好！请介绍一下你自己")
        ]
        
        # 首次响应
        print("\nAI助手已启动 (输入exit退出)")
        print(f"模型: {args.model}, 温度: {args.temperature}")
        
        if args.stream:
            print("[AI]: ", end="", flush=True)
            full_response = ""
            for chunk in model.stream(messages[1:]):  # 跳过系统消息
                print(chunk.content, end="", flush=True)
                full_response += chunk.content
            messages.append(AIMessage(content=full_response))
        else:
            response = model.invoke(messages[1:])
            print(f"[AI]: {response.content}")
            messages.append(AIMessage(content=response.content))
        
        # 交互循环
        while True:
            try:
                user_input = input("\n[user]: ").strip()
                if user_input.lower() in ("exit", "quit"):
                    break
                    
                messages.append(HumanMessage(content=user_input))
                
                print("[AI]: ", end="", flush=True)
                full_response = ""
                
                if args.stream:
                    for chunk in model.stream([messages[-1]]):  # 只发送最新消息
                        content = chunk.content
                        print(content, end="", flush=True)
                        full_response += content
                else:
                    response = model.invoke([messages[-1]])
                    print(response.content)
                    full_response = response.content
                
                messages.append(AIMessage(content=full_response))
                
            except KeyboardInterrupt:
                print("\n对话已终止")
                break
            except Exception as e:
                logger.error(f"对话出错: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"程序初始化失败: {str(e)}")
        return

if __name__ == "__main__":
    main()