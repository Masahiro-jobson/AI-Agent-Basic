from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from tools import save_tool, search_tool, wiki_tool

# load LLM API keys from .env file
load_dotenv()


class ResearchResponse(BaseModel):
    """Pydantic model for parsing the research response."""
    topic: str
    summary: str
    source: list[str]
    tool_used: list[str]


# llm variable contains the LLM model to use for the agent.
llm = ChatOpenAI(model="gpt-3.5-turbo")
llm2 = ChatAnthropic(model="claude-3-opus-20241022")


# Parser is a variable that contains an instance of the PydanticOutputParser class, which is used
# to parse the output of the LLM into a structured format defined by the ResearchResponse Pydantic model.
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Make a prompt variable that contains the prompt template for the agent.
# The prompt template is a string that contains placeholders for the input variables.
prompt = ChatPromptTemplate.from_messages(
    [
        # The system message s ets the context for the agent, let them know what it is supposed to do.
        ("system",
         "You are a research assistant that will help generate a research paper."
         "Answer the user query and use necessary tools."
         "Wrap the output in this format and provide no other text \n{format_instructions}"),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())


tools = [search_tool, wiki_tool, save_tool]

# This is a brain of the agent, which is created using the create_tool_calling_agent function. It takes in the LLM, tools, and prompt
# as arguments and returns an agent that can be executed using the AgentExecutor class.
agent = create_tool_calling_agent(
    llm=llm, tools=tools, prompt=prompt)

# AgentExecuter is a brain (class) of the agent, which is created
# using the create_tool_calling_agent function. It takes in the LLM, tools, and prompt
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What can I help you research ?")
# Run the agent executor with a sample query.
raw_response = agent_executor.invoke(
    {"query": query},)

# print(raw_response)


# Parse the raw response using the parser and print the structured response.
# If there is an error during parsing, print the error message and the raw response for debugging purposes.
try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print(structured_response)

except Exception as e:
    print("Error parsing response:", e, "Raw Response -", raw_response)
