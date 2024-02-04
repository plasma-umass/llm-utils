#!/usr/bin/env python3

from abc import ABC, abstractmethod, abstractclassmethod, abstractproperty
from typing import Dict, List, Optional, Tuple
import openai
import logging
from time import sleep
import json
import botocore.session
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
import requests as req
import re

from llm_utils.utils import contains_valid_json, extract_code_blocks

log = logging.getLogger("rich")

class ChatAPI(ABC):
    prompt_tokens: int
    completion_tokens: int
    @abstractclassmethod
    def assistant_message(cls, msg: str) -> Dict[str, str]:
        ...

    @abstractclassmethod
    def user_message(cls, msg: str) -> Dict[str, str]:
        ...

    @abstractclassmethod
    async def send_message(cls, conversation, n) -> List[str]:
        ...


class ChatGPT(ChatAPI):
    MODEL: str = "gpt-4-0314"
    prompt_tokens = 0
    completion_tokens = 0
    @classmethod
    def assistant_message(cls, msg: str) -> Dict[str, str]:
        return {"role": "system", "content": msg}

    @classmethod
    def user_message(cls, msg: str) -> Dict[str, str]:
        return {"role": "user", "content": msg}

    @classmethod
    async def send_message(cls, conversation: List[Dict[str, str]], n) -> List[str]:
        while True:
            try:
                response = await openai.ChatCompletion.acreate(
                    model=cls.MODEL, messages=conversation, n=n
                )
                break
            except openai.error.RateLimitError:  # type: ignore
                log.warn("rate limit!")
                sleep(1)
            except openai.error.Timeout:  # type: ignore
                log.warn("TIMEOUT")
                sleep(1)
            except (openai.error.APIError, openai.error.InvalidRequestError, openai.error.APIConnectionError) as e:
                log.warn(f"Other API error: {e}")
                sleep(5)
        cls.prompt_tokens += response["usage"]["prompt_tokens"]
        cls.completion_tokens += response['usage']['completion_tokens']
        return [resp['message']['content'] for resp in response['choices']]

class Claude(ChatAPI):
    ANTHROPIC_VERSION: str = "bedrock-2023-05-31"
    USER_PREFIX = "Human"
    ASSISTANT_PREFIX = "Assistant"
    REGION: str = 'us-west-2'
    MODEL_ID: str = 'anthropic.claude-v2'
    SERVICE_NAME: str = 'bedrock'
    MAX_RETRY: int = 5
    prompt_tokens: int = 0
    completion_tokens: int = 0
    @classmethod
    def generate_chatlog(cls, conversations):
        log = ""
        for conv in conversations:
            role = conv.get('role', "")
            content = conv.get('content', "")
            log += "{}: {}\n\n".format(role, content)
        return log.rstrip() + f"\n\n{cls.ASSISTANT_PREFIX}: " # remove the last newline characters

    @classmethod
    def create_payload(cls, conversation: List[Dict[str, str]]) -> Dict:
        blob = {
            "prompt" : cls.generate_chatlog(conversation),
            "max_tokens_to_sample" : 2048,
            "temperature": 0,
            "top_k": 250,
            "top_p": 1,
            "stop_sequences": [
                f"\n\n{cls.USER_PREFIX}:"
            ],
            "anthropic_version": cls.ANTHROPIC_VERSION
        }
        return blob
    @classmethod
    def assistant_message(cls, msg: str) -> Dict[str, str]:
        return {"role": cls.ASSISTANT_PREFIX, "content": msg}

    @classmethod
    def user_message(cls, msg: str) -> Dict[str, str]:
        return {"role": cls.USER_PREFIX, "content": msg}

    @classmethod
    async def send_message(cls, conversation: List[Dict[str, str]], n) -> Optional[List[str]]:
        # conversation[0]['role'] = 'Human'
        first_msg = conversation.pop(0)['content'] # A temporary hack to get around the fact that AWS wants the HUMAN message to be first.
        conversation[0]['content'] = first_msg.rstrip() + '\n' + conversation[0]['content']
        payload = cls.create_payload(conversation)
        for _ in range(5):
            inference = cls.get_inference(payload)
            print("INFERENCE", inference['completion'])

            jsonified_completion = contains_valid_json(inference['completion'])
            if jsonified_completion is not None:
                if 'responses' in jsonified_completion:
                    return [json.dumps(s) for s in jsonified_completion["responses"]]
                else:
                    return [json.dumps(jsonified_completion)]
            else:
                code_blocks = extract_code_blocks(inference['completion'])
                if len(code_blocks) == 2:
                    return [json.dumps({"snippet1": code_blocks[0], "snippet2": code_blocks[1]})]
                # raise RuntimeError()
        log.error("Could not get JSON after multiple attempts")
        return None
        # return [x for x in parse_chatlog(inference['completion'])[-1]['content']['responses']]
    @classmethod
    def get_inference(cls, payload: Dict) -> Dict:
        # print(f"making an inference request to {model_id}, payload={payload}")
        try:
            ## Initialize the runtime rest API to be called for the endpoint
            endpoint: str = f"https://{cls.SERVICE_NAME}-runtime.{cls.REGION}.amazonaws.com/model/{cls.MODEL_ID}/invoke"

            # Converting the payload dictionary into a JSON-formatted string to be sent in the HTTP request
            request_body = json.dumps(payload)

            # Creating an AWSRequest object for a POST request with the service specified endpoint, JSON request body, and HTTP headers
            request = AWSRequest(method='POST',
                                 url=endpoint,
                                 data=request_body,
                                 headers={'content-type': 'application/json'})

            # Initializing a botocore session
            session = botocore.session.Session()

            # Adding a SigV4 authentication information to the AWSRequest object, signing the request
            sigv4 = SigV4Auth(session.get_credentials(), cls.SERVICE_NAME, cls.REGION)
            sigv4.add_auth(request)

            # Prepare the request by formatting it correctly
            prepped = request.prepare()

            # Send the HTTP POST request to the prepared URL with the specified headers & JSON-formatted request body, storing the response
            response = req.post(prepped.url, headers=prepped.headers, data=request_body)

            if response.status_code == 200:
                return response.json()
            else:
                raise RuntimeError(f"Error: Received status code {response.status_code}, Response: {response.text}")


        except Exception as e:
            print(f"Exception occurred: {e}")
            raise

def get_model_from_str(model_name, llm_malformed_max_retry) -> Optional[ChatAPI]:
    model_name = model_name.lower()
    match model_name:
        case 'openai':
            return ChatGPT()
        case 'claude':
            Claude.MAX_RETRY = llm_malformed_max_retry
            return Claude()

    return None