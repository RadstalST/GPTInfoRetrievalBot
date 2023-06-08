import os

import chainlit as cl
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
## search tools
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import WikipediaAPIWrapper


from prompt import plan_prompt, prompt # import our custom prompt template

load_dotenv()

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_MODEL = os.getenv("GPT_MODEL")
llm = ChatOpenAI(temperature=0, model=GPT_MODEL)

search = DuckDuckGoSearchRun()
wikipedia = WikipediaAPIWrapper()
# define search tools with description for the model to understand which tools it would need to efficiently solve the task
# Web Search Tool
search_tool = Tool(
    name="Web Search",
    func=search.run,
    description="A web search tool is a software application or service that enables users to search for information on the internet. It is valuable for swiftly accessing a vast array of data and is widely used for research, learning, entertainment, and staying informed. With features like filters and personalized recommendations, users can easily find relevant results. However, web search tools may struggle with complex or specialized queries that require expert knowledge and can sometimes deliver biased or unreliable information. It is crucial for users to critically evaluate and verify the information obtained through web search tools, particularly for sensitive or critical topics.",
)
# Wikipedia Tool
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia.run,
    description="Wikipedia is an online encyclopedia that serves as a valuable web search tool. It is a collaborative platform where users can create and edit articles on various topics. Wikipedia provides a wealth of information on a wide range of subjects, making it a go-to resource for general knowledge and background information. It is particularly useful for getting an overview of a topic, understanding basic concepts, or exploring historical events. However, since anyone can contribute to Wikipedia, the accuracy and reliability of its articles can vary. It is recommended to cross-reference information found on Wikipedia with other reliable sources, especially for more specialized or controversial subjects.",
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
plan_chain = ConversationChain(
    llm=llm,
    memory=memory,
    input_key="input",
    prompt=plan_prompt,
    output_key="output",
)

# Initialize Agent
agent = initialize_agent(
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    tools=[search_tool, wikipedia_tool],
    llm=llm,
    verbose=True, # verbose option is for printing logs (only for development)
    max_iterations=10,
    prompt=prompt,
    memory=memory,
)



@cl.langchain_run
def run(agent, input_str):
    # Plan execution
    plan_result = plan_chain.run(input_str)
    print(plan_result)
    # Agent execution
    res = agent(plan_result)

    # Send message
    cl.Message(content=res["output"]).send()


@cl.langchain_factory
def factory():
    return agent