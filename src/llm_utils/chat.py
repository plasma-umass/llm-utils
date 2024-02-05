#!/usr/bin/env python3
# from __future__ import typing
import abc
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic
from typing_extensions import Literal, Required, TypedDict

# from mypy_extensions import TypedDict
import openai
import logging
from time import sleep
import json
import botocore.session
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
import requests

T = TypeVar("T")
from llm_utils.utils import contains_valid_json, extract_code_blocks

log = logging.getLogger("rich")
logging.basicConfig(filename="llm_utils.log", encoding="utf-8", level=logging.DEBUG)


class ChatAPI(abc.ABC, Generic[T]):
    prompt_tokens: int
    completion_tokens: int

    @abc.abstractmethod
    def assistant_message(cls, msg: str) -> T:
        ...

    @abc.abstractmethod
    def user_message(cls, msg: str) -> T:
        ...

    @abc.abstractmethod
    async def send_message(cls, conversation: List[T], n: int) -> Optional[List[str]]:
        ...


class ChatGPT(ChatAPI[ChatCompletionMessageParam]):
    MODEL: str = "gpt-4-0314"
    prompt_tokens = 0
    completion_tokens = 0

    @classmethod
    def assistant_message(cls, msg: str) -> ChatCompletionSystemMessageParam:
        return {"role": "system", "content": msg}

    @classmethod
    def user_message(cls, msg: str) -> ChatCompletionUserMessageParam:
        return {"role": "user", "content": msg}

    @classmethod
    async def send_message(
        cls, conversation: List[ChatCompletionMessageParam], n: int
    ) -> Optional[List[str]]:
        while True:
            try:
                client = openai.AsyncOpenAI(timeout=30)
                response = await client.chat.completions.create(
                    model=cls.MODEL, messages=conversation, n=n
                )
                # response = await openai.ChatCompletion.acreate(
                #    model=cls.MODEL, messages=conversation, n=n
                # )
                break
            except openai.RateLimitError:
                log.warning("Rate limit exceeded. Retrying...")
                sleep(1)
            except openai.APITimeoutError:
                log.warning("API request timed out. Retrying...")
                sleep(1)
            except (
                openai.APIError,
                openai.BadRequestError,
                openai.APIConnectionError,
            ) as e:
                log.warning(f"API error encountered: {e}. Retrying...")
                sleep(5)
        cls.prompt_tokens += (
            response.usage.prompt_tokens if response.usage is not None else 0
        )
        cls.completion_tokens += (
            response.usage.completion_tokens if response.usage is not None else 0
        )
        return [
            resp.message.content
            for resp in response.choices
            if resp.message.content is not None
        ]


class ClaudeAssistantMessageParam(TypedDict):
    role: Required[Literal["Assistant"]]
    content: Required[str]


class ClaudeUserMessageParam(TypedDict):
    role: Required[Literal["Human"]]
    content: Required[str]


ClaudeMessageParam = Union[ClaudeAssistantMessageParam, ClaudeUserMessageParam]


class Claude(ChatAPI[ClaudeMessageParam]):
    ANTHROPIC_VERSION: str = "bedrock-2023-05-31"
    USER_PREFIX: Literal["Human"] = "Human"
    ASSISTANT_PREFIX: Literal["Assistant"] = "Assistant"
    REGION: str = "us-west-2"
    MODEL_ID: str = "anthropic.claude-v2"
    SERVICE_NAME: str = "bedrock"
    MAX_RETRY: int = 5
    prompt_tokens: int = 0  # FIXME not yet implemented
    completion_tokens: int = 0  # ibid

    @classmethod
    def generate_chatlog(cls, conversations: List[ClaudeMessageParam]) -> str:
        log = ""
        for conv in conversations:
            role = conv.get("role", "")
            content = conv.get("content", "")
            log += "{}: {}\n\n".format(role, content)
        return (
            log.rstrip() + f"\n\n{cls.ASSISTANT_PREFIX}: "
        )  # remove the last newline characters

    @classmethod
    def create_payload(cls, conversation: List[ClaudeMessageParam]) -> Dict[Any, Any]:
        blob = {
            "prompt": cls.generate_chatlog(conversation),
            "max_tokens_to_sample": 2048,
            "temperature": 0,
            "top_k": 250,
            "top_p": 1,
            "stop_sequences": [f"\n\n{cls.USER_PREFIX}:"],
            "anthropic_version": cls.ANTHROPIC_VERSION,
        }
        return blob

    @classmethod
    def assistant_message(cls, msg: str) -> ClaudeAssistantMessageParam:
        return {"role": cls.ASSISTANT_PREFIX, "content": msg}

    @classmethod
    def user_message(cls, msg: str) -> ClaudeUserMessageParam:
        return {"role": cls.USER_PREFIX, "content": msg}

    @classmethod
    async def send_message(
        cls, conversation: List[ClaudeMessageParam], n: int
    ) -> Optional[List[str]]:
        # conversation[0]['role'] = 'Human'
        first_msg = conversation.pop(0)[
            "content"
        ]  # A temporary hack to get around the fact that AWS wants the HUMAN message to be first.
        conversation[0]["content"] = (
            first_msg.rstrip() + "\n" + conversation[0]["content"]
        )
        payload = cls.create_payload(conversation)
        for _ in range(cls.MAX_RETRY):
            inference = cls.get_inference(payload)
            log.info(f'Result: {inference["completion"]}')

            jsonified_completion = contains_valid_json(inference["completion"])
            if jsonified_completion is not None:
                if "responses" in jsonified_completion:
                    return [json.dumps(s) for s in jsonified_completion["responses"]]
                else:
                    return [json.dumps(jsonified_completion)]
            else:
                code_blocks = extract_code_blocks(inference["completion"])
                if len(code_blocks) == 2:
                    return [
                        json.dumps(
                            {"snippet1": code_blocks[0], "snippet2": code_blocks[1]}
                        )
                    ]
                # raise RuntimeError()
        log.error("Could not get JSON after multiple attempts")
        return None
        # return [x for x in parse_chatlog(inference['completion'])[-1]['content']['responses']]

    @classmethod
    def get_inference(cls, payload: Dict[Any, Any]) -> Any:
        # print(f"making an inference request to {model_id}, payload={payload}")
        try:
            ## Initialize the runtime rest API to be called for the endpoint
            endpoint: str = f"https://{cls.SERVICE_NAME}-runtime.{cls.REGION}.amazonaws.com/model/{cls.MODEL_ID}/invoke"

            # Converting the payload dictionary into a JSON-formatted string to be sent in the HTTP request
            request_body = json.dumps(payload)

            # Creating an AWSRequest object for a POST request with the service specified endpoint, JSON request body, and HTTP headers
            request = AWSRequest(
                method="POST",
                url=endpoint,
                data=request_body,
                headers={"content-type": "application/json"},
            )

            # Initializing a botocore session
            session = botocore.session.Session()

            # Adding a SigV4 authentication information to the AWSRequest object, signing the request
            sigv4 = SigV4Auth(session.get_credentials(), cls.SERVICE_NAME, cls.REGION)
            sigv4.add_auth(request)

            # Prepare the request by formatting it correctly
            prepped = request.prepare()

            # Send the HTTP POST request to the prepared URL with the specified headers & JSON-formatted request body, storing the response
            response = requests.post(
                prepped.url, headers=prepped.headers, data=request_body  # type:ignore
            )

            if response.status_code == 200:
                return response.json()
            else:
                raise RuntimeError(
                    f"Error: Received status code {response.status_code}, Response: {response.text}"
                )

        except Exception as e:
            print(f"Exception occurred: {e}")
            raise


def get_model_from_str(
    model_name: str, llm_malformed_max_retry: int
) -> Optional[ChatGPT | Claude]:
    model_name = model_name.lower()
    match model_name:
        case "openai":
            return ChatGPT()
        case "claude":
            Claude.MAX_RETRY = llm_malformed_max_retry
            return Claude()

    return None
