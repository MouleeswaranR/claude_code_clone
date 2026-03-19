from prompts.system import get_system_prompt
from dataclasses import dataclass
from utils.text import count_tokens
from typing import Any


@dataclass
class MessageItem:
    #what messages should hold for context
    role:str
    content:str
    token_count:int

    #function for converting each message item to dictionary
    def to_dict(self)->dict[str,Any]:
        result:dict[str,Any]={"role":self.role}

        #adding content to dictionary if present
        if self.content:
            result['content']=self.content
        
        return result



class ContextManager:

    def __init__(self)->None:
        #system prompt for the context to LLM
        self._system_prompt=get_system_prompt()
        self._messages:list[MessageItem]=[]
        self._model_name="meta-llama/llama-3.1-8b-instruct"
    
    #user message adding function
    def add_user_message(self,content:str)->None:
        #creating one message item
        item=MessageItem(
            role='user',
            content=content,
            token_count=count_tokens(
                content,
                self._model_name
            )
        )

        #appending it to the messages list which contains all the previous older messages
        self._messages.append(item)
    
    #assistant message adding function
    def add_assistant_message(self,content:str)->None:
        #creating one message item
        item=MessageItem(
            role='assistant',
            content=content or "",
            token_count=count_tokens(
                content,
                self._model_name
            )
        )

        #appending it to the messages list which contains all the previous older messages
        self._messages.append(item)
    
    #getting messages from messagelist
    def get_messages(self)->list[dict[str,Any]]:
        messages=[]

        #adding system prompt to the message list
        if self._system_prompt:
            messages.append({
                "role":"system",
                "content":self._system_prompt
            })
        
        #getting previous messages and adding it to message list
        for item in self._messages:
            messages.append(item.to_dict())#to_dict function will only return role and content not token_usage
        

        return messages