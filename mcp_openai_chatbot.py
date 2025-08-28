from dotenv import load_dotenv
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import List, Dict, TypedDict
import asyncio
import nest_asyncio
import os
from contextlib import AsyncExitStack
import json
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

nest_asyncio.apply()
load_dotenv()

class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict

class MCP_ChatBot:
    def __init__(self):
        self.sessions: List[ClientSession] = []
        self.exit_stack = AsyncExitStack() 
        self.llm = ChatOpenAI(
            deployment_id=os.environ.get('model_id'),
            temperature=0
        )
        self.available_tools: List[ToolDefinition] = []
        self.tool_to_session: Dict[str, ClientSession] = {}

    async def process_query(self, query: str):
        messages = [HumanMessage(content=query)]

        while True:
            response: AIMessage = self.llm.invoke(messages, tools=self.available_tools)

            if response.tool_calls:
                tool_messages = []  

                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]

                    print(f" Calling MCP tool: {tool_name} with args {tool_args}")

                    try:
                        session = self.tool_to_session[tool_name]
                        result = await session.call_tool(tool_name, arguments=tool_args)

                        tool_result = (
                            result.content if hasattr(result, "content") else str(result)
                        )
                        print(f"Tool result: {tool_result}")

                        tool_messages.append(
                            ToolMessage(tool_call_id=tool_id, content=str(tool_result))
                        )

                    except Exception as e:
                        print(f" Error calling tool {tool_name}: {str(e)}")
                        return

                messages.append(response)
                messages.extend(tool_messages)

            else:
                print(f"\n Final Answer:\n{response.content}\n")
                break



    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
        
                if query.lower() == 'quit':
                    break
                    
                await self.process_query(query)
                print("\n")
                    
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        # server_params = StdioServerParameters(
        #     command="uv",
        #     args=["run", "research_server.py"],
        #     env=None,
        # )
        try: 
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            ) # new
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            ) 
            await session.initialize()
            self.sessions.append(session)

            response = await session.list_tools()
            tools = response.tools

            print(f"\n Connected to {server_name} with tools:", [tool.name for tool in tools])

            for tool in tools:
                self.available_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })
                # 🔑 Map each tool to the correct session
                self.tool_to_session[tool.name] = session

        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")

    async def connect_to_servers(self):
        """Connect to all configured MCP servers."""
        try:
            with open("server_config.json", "r") as file:
                data = json.load(file)
            
            servers = data.get("mcpServers", {})

            print("servers ", servers)
            
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            print(f"Error loading server configuration: {e}")
            raise
    async def cleanup(self): # new
        """Cleanly close all resources using AsyncExitStack."""
        await self.exit_stack.aclose()

async def main():
    chatbot = MCP_ChatBot()
    try:
        # the mcp clients and sessions are not initialized using "with"
        # like in the previous lesson
        # so the cleanup should be manually handled
        await chatbot.connect_to_servers() # new! 
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup() #new! 


if __name__ == "__main__":
    asyncio.run(main())
