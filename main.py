from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    

llm = ChatAnthropic(model_name="claude-3-5-sonnet-20241022", timeout=60, stop=[])
parser = PydanticOutputParser(pydantic_object=ResearchResponse)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())
tools = [search_tool, wiki_tool, save_tool]
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a research assistant that will help generate a research paper. Answer the user query and use necessary tools. Wrap the output in this format and provide no other text",
    response_format=ResearchResponse,
)
query = input("What can I help you research? ")
raw_response = agent.invoke({"messages": [{"role": "user", "content": query}]})
print(raw_response)

try:
    text = raw_response['messages'][-1].content[0]['text']
    structured_response = parser.parse(text)
    print(structured_response)
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)
