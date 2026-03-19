from openai import AsyncOpenAI,RateLimitError,APIConnectionError,APIError
from typing import Any,AsyncGenerator
from .response import TextDelta,TokenUsage,StreamEvent,StreamEventType
import asyncio


class LLMClient:

    def __init__(self)->None:
        #initially client is none
        self._client:AsyncOpenAI|None=None
        self._max_retries:int=3
    

    def get_client(self)->AsyncOpenAI:
        #if client is None object create new client
        if self._client is None:
            self._client=AsyncOpenAI(
                api_key='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',
                base_url="https://openrouter.ai/api/v1",
            ) 
        #else, return current client
        return self._client

    async def close(self)->None:
        #check the current client and close it
        if self._client:
            await self._client.close()
            self._client=None
    

    async def chat_completion(self,messages:list[dict[str,Any]],stream:bool=False)->AsyncGenerator[StreamEvent,None]:
        """
        messages: list of messages between user and llm ,dict is to mention whether it is user or llm
        (messages needs to be stored because llm'are stateless)

        stream:gives outputs chunk by chunk without waiting for whole response from the LLM
        """

        client=self.get_client()

        kwargs={
            "model":"meta-llama/llama-3.1-8b-instruct",
            "messages":messages,
            "stream":stream
        }
        
        for attempt in range(self._max_retries+1):
            try:
                
                if stream:
                    async for event in self._stream_response(client,kwargs):
                        yield event
                else:
                    event=await self._non_stream_response(client,kwargs)
                    yield event
                return
            except RateLimitError as e:
                if attempt<self._max_retries:
                    #attempt->failed -> wait for some time
                    wait_time=2**attempt
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        error=f'Rate limit exceeded : {e}'
                    )
                    return
            except APIConnectionError as e:
                if attempt<self._max_retries:
                    #attempt->failed -> wait for some time
                    wait_time=2**attempt
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        error=f'Connection error: {e}'
                    )
                    return
            except APIError as e:
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        error=f'API error: {e}'
                    )
                    return 
            except Exception as e:   # ← catch-all
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        error=f"Unexpected exception in chat_completion: {type(e).__name__}: {str(e)}"
                    )

    
    #method for stream response
    async def _stream_response(
            self,client:AsyncOpenAI,
            kwargs:dict[str,Any]
    )->AsyncGenerator[StreamEvent,None]:
        response=await client.chat.completions.create(**kwargs)

        # print("STREAM CALLED")

        usage:TokenUsage|None=None
        finish_reason:str|None=None
        async for chunk in response:
            #getting token usage if available in each streaming chunk
            if hasattr(chunk,"usage") and chunk.usage:
                # print("CHUNK:", chunk)
                usage = TokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                    cached_tokens=getattr(chunk.usage.prompt_tokens_details, "cached_tokens", 0)
                    if chunk.usage.prompt_tokens_details
                    else 0
                     )
            
            #continue if chunk has no content
            if not chunk.choices:
                continue
            
            #getting chunk output
            choice=chunk.choices[0]
            delta=choice.delta #delta-streamed partial generated output from LLM

            #getting finish reason
            if choice.finish_reason:
                finish_reason=choice.finish_reason
            
            if delta.content:
                #yield response for current chunk
                yield StreamEvent(
                    type=StreamEventType.TEXT_DELTA,
                    text_delta=TextDelta(delta.content)
                )
        
        #yield response after all the chunks processed with message complete type and finish reason
        yield StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            finish_reason=finish_reason,
            usage=usage
        )

    #method for non-stream output
    async def _non_stream_response(self,client:AsyncOpenAI,kwargs:dict[str,Any])->StreamEvent:
        response=await client.chat.completions.create(**kwargs)
        # print("NON STREAM CALLED")
        choice=response.choices[0]
        message=choice.message

        #extracting message content and tokens usage from response of LLM
        text_delta=None
        if message.content:
            text_delta = TextDelta(content=message.content)
        
        usage=None
        if response.usage:
            usage=TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                cached_tokens=response.usage.prompt_tokens_details.cached_tokens,
            )
        
        return StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            text_delta=text_delta,
            finish_reason=choice.finish_reason,
            usage=usage
        )