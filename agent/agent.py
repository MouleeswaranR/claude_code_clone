from __future__ import annotations

from typing import AsyncGenerator
from agent.events import AgentEvent,AgentEventType
from client.llm_client import LLMClient
from client.response import StreamEventType
from context.manager import ContextManager

 
class Agent:

    def __init__(self):
        self.client=LLMClient()
        self.context_manager=ContextManager()

    #function for accessing agentic loop(private method)
    async def run(self,message:str):#message-input from user
        #agent event started
        yield AgentEvent.agent_start(message=message)
        #adding user message to context manager for the agent to remember
        self.context_manager.add_user_message(message)

        final_response:str|None=None
        async for event in self._agentic_loop():
            yield event
            
            #getting final response content from text_complete event
            if event.type==AgentEventType.TEXT_COMPLETE:
                final_response=event.data.get("content")
        
        #agent event end after agentic loop
        yield AgentEvent.agent_end(final_response or "No response generated.")#agent end requires text complete contains final response

    #creating a agentic loop for agent
    async def _agentic_loop(self)->AsyncGenerator[AgentEvent,None]:
        messages=[{'role':'user','content':"Hey what is going on"}]
        #passing previous messages to LLM client in streaming event
        response_text=""#final response text from llm
        async for event in  self.client.chat_completion(self.context_manager.get_messages(),True):#accessing all messages from context manager
            # print(f"[CLIENT EVENT RECEIVED] type={event.type}")
            if event.type==StreamEventType.TEXT_DELTA:#if stream event is text delta(partially generated text)
                if event.text_delta:#check if event has text delta content
                    content=event.text_delta.content
                    response_text+=content
                    # print(f"[DEBUG delta] {content!r}")
                    yield AgentEvent.text_delta(content)#text delta event on Agent events

            # elif event.type == StreamEventType.MESSAGE_COMPLETE:
            #     # 🔥 IMPORTANT FIX
            #     yield AgentEvent.text_complete(response_text)

            elif event.type==StreamEventType.ERROR:#if streaming evebnt returned error
                error_msg = event.error or "Unknown streaming error (no message provided)"
                # print(f"[STREAMING ERROR FROM LLM CLIENT] {error_msg}") 
                # print(f"          Full event: {event}")
                yield AgentEvent.agent_error(
                    event.error 
                )
        
        #adding assistant message to the context manager
        self.context_manager.add_assistant_message(
            response_text or None,
        )

        #if response_text present means after agentic loop, it inidcates text_complete and then agent_end event type
        if response_text:
            yield AgentEvent.text_complete(response_text)

    async def __aenter__(self)->Agent:
        return self
    
    async def __aexit__(self,exc_type,exc_val,exc_tb)->None:
        #closing the instance of the agent by closing llm instance
        if self.client:
            await self.client.close()
            self.client=None