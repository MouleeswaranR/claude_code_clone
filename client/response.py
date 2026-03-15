from __future__ import annotations #to use same class Objevct within the same class
from dataclasses import dataclass
from enum import Enum


@dataclass
class TextDelta:
    content:str

    def __str__(self):
        return self.content

@dataclass
class EventType(str,Enum):#mentioning types of output from LLM
    TEXT_DELTA="text_delta" #mentioning partial output streamed from LLM
    MESSAGE_COMPLETE="message_complete"#output complete message from LLM
    ERROR="error"

@dataclass
class TokenUsage:
    prompt_tokens:int=0
    completion_tokens:int=0
    total_tokens:int=0
    cached_tokens:int=0

    def __add__(self,other:TokenUsage):
        return TokenUsage(
            prompt_tokens=self.prompt_tokens+other.prompt_tokens,
            completion_tokens=self.completion_tokens+other.completion_tokens,
            total_tokens=self.total_tokens+other.total_tokens,
            cached_tokens=self.cached_tokens+other.cached_tokens
        )


@dataclass
class StreamEvent:
    type:EventType
    text_delta:TextDelta|None=None #can have none because output may be tool calling
    error: str|None=None
    finish_reason: str|None=None
    usage:TokenUsage|None=None