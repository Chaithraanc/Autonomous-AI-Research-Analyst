import asyncio
import os
import json
from contextlib import asynccontextmanager
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from groq import Groq

# 1. Configuration Setup
GROQ_API_KEY = os.environ.get("MY_API_KEY_AI", "your_groq_api_key_here")
groq_client = Groq(api_key=GROQ_API_KEY)

# Define how to talk to your local server script safely on Windows
# .venv/Scripts/python is the explicit path to your local virtual environment execution environment
server_params = StdioServerParameters(
    command=os.path.join(".venv", "Scripts", "python.exe"),
    args=["server.py"]
)


@asynccontextmanager
async def mcp_agent_bridge():
    """Spawns your local server.py script as a background process over stdio."""
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize connection protocol
            await session.initialize()
            yield session

async def run_research_agent(user_prompt: str):
    print(f"🚀 Starting autonomous research agent...")
    print(f"👤 User Request: '{user_prompt}'\n" + "="*50)
    
    async with mcp_agent_bridge() as session:
        # 1. Discover tools dynamically from your local MCP server
        mcp_tools_response = await session.list_tools()
        
        # 2. Format the tools into schemas that the Groq LLM understands
        groq_tools = []
        tool_name_map = {}
        
        for tool in mcp_tools_response.tools:
            tool_name_map[tool.name] = tool
            groq_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema
                }
            })

            
     # Initialize conversation state loop with absolute structural enforcement
        messages = [
            {"role": "system", "content": (
                "You are an expert Research Analyst Agent equipped with specialized local database "
                "and paper scraping tools. Your workflow is: \n"
                "1. Search arXiv for papers related to the topic.\n"
                "2. Pick the best direct PDF URL and call ingest_and_vectorize to parse/embed it.\n"
                "3. Use query_vector_db with precise keywords to extract granular table facts/text.\n"
                "4. Compile a brilliant, data-backed summary report for the user citing sources.\n\n"
                "CRITICAL INSTRUCTION FOR TOOL CALLING:\n"
                "When you choose to call a tool, you MUST NOT output any conversational text, explanations, "
                "or thoughts before or alongside the tool call. Output ONLY the raw tool call payload. "
                "Do not think out loud or narrate your steps."
            )},
            {"role": "user", "content": user_prompt}
        ]
        
        # Max ReAct cycles to avoid infinite loops
        max_iterations = 8
        
        for iteration in range(max_iterations):
            print(f"\n🧠 [Thinking Cycle {iteration + 1}]: Routing...")
            
                        # Chat completion call to Groq
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",  
                messages=messages,
                tools=groq_tools,
                tool_choice="auto",
                temperature=0.2
            )
            
            # Safely grab the first message choice from the completion structure
            response_message = response.choices[0].message
            
            # Convert to dictionary payload for the agent message history tracking
            messages.append({
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": [tc.model_dump() for tc in response_message.tool_calls] if response_message.tool_calls else None
            })

            
            # Check if the LLM wants to talk to a tool or finish the report
            if not response_message.tool_calls:
                print("\n✅ Final Synthesis Achieved!")
                print(f"\n📊 **FINAL REPORT**:\n{response_message.content}")
                break
                
            # Execute tool paths requested by the agent loop
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"🛠️ [Executing Tool]: `{function_name}` with arguments: {function_args}")
                
                # Execute the actual tool function inside your local server.py
                tool_result = await session.call_tool(function_name, arguments=function_args)
                
                # Extract text payload content from tool response blocks
                result_text = "".join([content.text for content in tool_result.content if hasattr(content, 'text')])
                
                print(f"📥 [Observation]: Tool output received successfully ({len(result_text)} characters).")
                
                # Feed the observation facts back to the LLM context path
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": result_text
                })

if __name__ == "__main__":
    # Get a quick input prompt from your command prompt window
    prompt = input("Enter research question: ")
    asyncio.run(run_research_agent(prompt))
