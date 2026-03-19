import asyncio
import sys
import click
from agent.agent import Agent
from agent.events import AgentEventType
from ui.tui import TUI,get_console

#console for tui
console=get_console()

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class CLI:
    def __init__(self):
        self.agent:Agent|None=None
        self.tui=TUI(console)

    async def run_single(self,message:str)->str|None:
        #to use agent like this , added __aenter__,__aexit__ to agent
        async with Agent() as agent:
            self.agent=agent
            return await self._process_message(message)

    async def _process_message(self,message:str)->str|None:
        
        #check if there is agent instance
        if not self.agent:
            return None
        
        assistant_streaming=False
        final_response:str|None=None
        #run the agentic loop using agent instance
        async for event in self.agent.run(message):
            # print("EVENT:", event.type, event.data)
            #if event is TEXT_DELTA(partially generated text)
            if event.type==AgentEventType.TEXT_DELTA:
                content=event.data.get("content","")
                #stream messages to tui
                # print(f"[DEBUG TUI] {content!r}")
                if not assistant_streaming:
                    self.tui.begin_assistant()
                    assistant_streaming=True
                self.tui.stream_assistant_delta(content)
            elif event.type==AgentEventType.TEXT_COMPLETE:
                final_response=event.data.get("content")
                if assistant_streaming:
                    self.tui.end_assistant()
                    assistant_streaming=False
            elif event.type==AgentEventType.AGENT_ERROR:
                error=event.data.get("error","Unknown error")
                console.print(f"\n[error]Error:{error}[/error]")

        
        return final_response


@click.command()
@click.argument("prompt",required=False)
def main(
    prompt:str|None,

):
    cli=CLI()
    # messages=[{
    #     'role':'user',
    #     'content':prompt
    # }]
    if prompt:
       result= asyncio.run(cli.run_single(prompt))
       if result is None:
           sys.exit(1)
   

main()