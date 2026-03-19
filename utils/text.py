#tiktoken for calculating tokens in each message
import tiktoken

#tokenizer for current given model
def get_tokenizer(model:str):
    try:
        #trying to get encoder for current model
        encoding=tiktoken.encoding_for_model(model)
        #returning encode function
        return encoding.encode
    except Exception as e:
        #using base tokenizer used by gpt-4 if given model tokenizer is not available
        encoding=tiktoken.get_encoding("cl100k_base")
        return encoding.encode

def count_tokens(text:str,model:str)->int:
    #getting tokenizer encode function from above model
    tokenizer=get_tokenizer(model)

    #if tokenizer available, counting tokens
    if tokenizer:
        return len(tokenizer(text))
    
    #a fallback function if tokenizer is not available
    return estimate_tokens(text)
    

def estimate_tokens(text:str)->int:
    return max(1,len(text)//4)

