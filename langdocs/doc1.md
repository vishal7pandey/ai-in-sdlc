<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# I want all langchain, langgraph and langsmith documentation, everything - workflows, features, tips and tricks, design patterns, for Python.

Now I have comprehensive information. Let me create a detailed documentation guide.

# Complete LangChain, LangGraph, and LangSmith Documentation for Python

## Overview

This comprehensive guide covers everything you need to know about LangChain, LangGraph, and LangSmith for Python development, including workflows, features, tips and tricks, and design patterns.[^1][^2][^3]

## LangChain Python Documentation

### Installation

```python
pip install -U langchain
# Requires Python 3.10+
```


### Core Concepts

#### 1. LangChain Expression Language (LCEL)

LCEL is a declarative way to compose LangChain components using the pipe operator (`|`). It provides a standardized method for creating chains with powerful features like streaming, async support, and parallel execution.[^4][^5][^6]

**Key Features:**

- **Declarative Syntax**: Chain components using the pipe `|` operator
- **Parallel Execution**: Use `RunnableParallel` for concurrent tasks
- **Asynchronous Support**: All chains run async by default
- **Streaming**: Built-in support for token-level streaming[^4][^6]

**Basic LCEL Chain Example:**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Define components
prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
llm = ChatOpenAI(model="gpt-4")
output_parser = StrOutputParser()

# Create chain using pipe operator
chain = prompt | llm | output_parser

# Invoke the chain
response = chain.invoke({"topic": "artificial intelligence"})
```

**Parallel Execution:**

```python
from langchain_core.runnables import RunnableParallel, RunnableLambda

def to_uppercase(x): return x.upper()
def word_count(x): return len(x.split())

parallel = RunnableParallel({
    "upper": RunnableLambda(to_uppercase),
    "count": RunnableLambda(word_count)
})

result = parallel.invoke("LangChain makes AI development easier")
# {'upper': 'LANGCHAIN MAKES AI DEVELOPMENT EASIER', 'count': 5}
```


#### 2. Prompt Templates

Prompt templates provide reusable structures for building prompts with dynamic inputs.[^7][^8][^9]

**String-Based Prompts:**

```python
from langchain_core.prompts import PromptTemplate

# Create template
prompt = PromptTemplate.from_template("Tell me about {topic}")

# Format prompt
formatted = prompt.format(topic="Python")

# Use in chain
chain = prompt | llm
```

**Chat Prompts:**

```python
from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who explains {concept} clearly."),
    ("human", "{question}")
])

# Format messages
formatted = chat_prompt.format_messages(
    concept="quantum computing",
    question="What is superposition?"
)
```

**Few-Shot Prompting:**

```python
from langchain_core.prompts import FewShotPromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"}
]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Input: {input}\nOutput:",
    input_variables=["input"]
)
```


#### 3. Tools and Function Calling

Tools allow LLMs to interact with external functions and APIs.[^10][^11][^12]

**Using the @tool Decorator:**

```python
from langchain.tools import tool

@tool
def search(query: str) -> str:
    """Search for information about a query."""
    return f"Results for: {query}"

@tool
def multiply(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y
```

**Pydantic-Based Tools:**

```python
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.tools import tool

class SearchInput(BaseModel):
    query: str = Field(description="The search query")
    max_results: int = Field(default=5, description="Maximum results")

@tool(args_schema=SearchInput)
def advanced_search(query: str, max_results: int = 5) -> str:
    """Perform an advanced search."""
    return f"Found {max_results} results for {query}"
```

**Tool Binding:**

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
tools = [search, multiply]

# Bind tools to model
llm_with_tools = llm.bind_tools(tools)

# The model can now decide which tools to use
response = llm_with_tools.invoke("What is 25 times 4?")
```


#### 4. Agents

Agents use LLMs to decide which actions to take in a ReAct (Reasoning + Acting) loop.[^13][^14][^15]

**Creating a Simple Agent:**

```python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's sunny in {city}!"

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[get_weather],
    system_prompt="You are a helpful assistant"
)

# Run the agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "what is the weather in sf"}]
})
```

**ReAct Pattern:**

The ReAct pattern alternates between:

1. **Thought**: Agent reasons about what to do
2. **Action**: Agent selects and executes a tool
3. **Observation**: Agent receives results
4. **Final Answer**: Agent provides response[^14][^13]
```python
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Create agent with tools
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5  # Prevent infinite loops
)

result = agent_executor.invoke({"input": "Calculate 3 + 5 and search for AI"})
```


#### 5. Memory

Memory allows chains and agents to maintain conversation context.[^16][^17][^18][^19]

**Conversation Buffer Memory:**

```python
from langchain.memory import ConversationBufferMemory

# Simple buffer - stores all messages
memory = ConversationBufferMemory(return_messages=True)

# Save context
memory.save_context(
    {"input": "Hi, I'm Alice"},
    {"output": "Hello Alice! How can I help?"}
)

# Load memory
messages = memory.load_memory_variables({})
```

**Conversation Buffer Window Memory:**

```python
from langchain.memory import ConversationBufferWindowMemory

# Keep only last k interactions
memory = ConversationBufferWindowMemory(
    k=4,  # Keep last 4 messages
    return_messages=True
)
```

**Conversation Summary Memory:**

```python
from langchain.memory import ConversationSummaryMemory

# Summarizes conversation history
memory = ConversationSummaryMemory(
    llm=llm,
    max_token_limit=650
)
```


#### 6. Retrieval-Augmented Generation (RAG)

RAG combines retrieval with generation for context-aware responses.[^20][^21][^22]

**Document Loading and Splitting:**

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load documents
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(docs)
```

**Vector Store and Embeddings:**

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Create embeddings
embeddings = OpenAIEmbeddings()

# Create vector store
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="my_collection",
    persist_directory="./chroma_db"
)
```

**RAG Agent:**

```python
from langchain.tools import tool

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Create RAG agent
tools = [retrieve_context]
prompt = "You have access to a tool that retrieves context. Use it to help answer user queries."

agent = create_agent(model, tools, system_prompt=prompt)
```


#### 7. Output Parsers

Output parsers structure LLM responses into specific formats.[^23][^24][^25]

**Structured Output Parser:**

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Define schema
response_schemas = [
    ResponseSchema(name="answer", description="The answer to the question"),
    ResponseSchema(name="source", description="Source of information")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Get format instructions
format_instructions = output_parser.get_format_instructions()

# Use in prompt
prompt = PromptTemplate(
    template="Answer the question.\n{format_instructions}\n{question}",
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions}
)
```

**Pydantic Output Parser:**

```python
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline")

parser = PydanticOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Tell a joke.\n{format_instructions}",
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser
result = chain.invoke({})
```


#### 8. Streaming and Callbacks

LangChain supports streaming responses and custom callbacks.[^26][^27][^28]

**Streaming:**

```python
# Stream tokens
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)

# Async streaming
async for chunk in chain.astream({"topic": "AI"}):
    print(chunk, end="", flush=True)
```

**Custom Callback Handler:**

```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = ChatOpenAI(
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)
```


#### 9. Model Integrations

LangChain integrates with numerous model providers.[^29][^30][^31]

**Popular Providers:**

- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Google (Gemini, Vertex AI)
- AWS Bedrock
- Azure OpenAI
- Cohere
- Hugging Face
- Mistral
- Groq
- Together AI[^30][^29]

**Standard Interface:**

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Swap providers easily
model = ChatOpenAI(model="gpt-4")
# or
model = ChatAnthropic(model="claude-3-sonnet")

# Same interface for all
response = model.invoke("Hello!")
```


***

## LangGraph Documentation

### Overview

LangGraph is a low-level orchestration framework for building stateful, multi-actor applications with LLMs. It represents workflows as graphs with nodes and edges.[^32][^33][^34][^35]

### Installation

```python
pip install -U langgraph
```


### Core Concepts

#### 1. State Management

State is the backbone of LangGraph applications. It's shared across all nodes.[^36][^37]

**Defining State:**

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph

class State(TypedDict):
    messages: list
    count: int
    result: str
```

**State with Reducers:**

```python
from typing import Annotated
import operator

class State(TypedDict):
    # This will append new items to the list
    items: Annotated[list, operator.add]
    # This will accumulate values
    total: Annotated[int, operator.add]
```


#### 2. Nodes and Edges

**Nodes** perform work, **edges** define transitions.[^37]

**Creating Nodes:**

```python
def node_1(state: State):
    return {"count": state["count"] + 1}

def node_2(state: State):
    return {"result": f"Count is {state['count']}"}

# Build graph
graph = StateGraph(State)
graph.add_node("node_1", node_1)
graph.add_node("node_2", node_2)
```

**Adding Edges:**

```python
from langgraph.graph import START, END

# Direct edge
graph.add_edge(START, "node_1")
graph.add_edge("node_1", "node_2")
graph.add_edge("node_2", END)
```


#### 3. Conditional Edges

Conditional edges enable dynamic routing based on state.[^38][^39][^40]

```python
def route_decision(state: State) -> str:
    """Decide next node based on state"""
    if state["count"] > 10:
        return "node_a"
    else:
        return "node_b"

# Add conditional edge
graph.add_conditional_edges(
    "node_1",              # From this node
    route_decision,        # Routing function
    {
        "node_a": "node_a",  # If returns "node_a"
        "node_b": "node_b"   # If returns "node_b"
    }
)
```


#### 4. Persistence and Checkpointing

Checkpointers save graph state at every step, enabling human-in-the-loop, memory, and fault tolerance.[^41][^42][^43]

**In-Memory Checkpointer:**

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

# Use with thread_id for conversation tracking
config = {"configurable": {"thread_id": "conversation-1"}}
result = graph.invoke({"messages": [...]}, config=config)
```

**PostgreSQL Checkpointer:**

```python
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string("postgresql://...")
graph = graph_builder.compile(checkpointer=checkpointer)
```


#### 5. Human-in-the-Loop (HITL)

LangGraph supports pausing execution for human input.[^44][^45][^46][^47]

**Using interrupt():**

```python
from langgraph.types import interrupt

def ask_human(state: State):
    # Pause and wait for human input
    response = interrupt("What should I do next?")
    return {"user_input": response}
```

**Static Interrupts:**

```python
graph = builder.compile(
    interrupt_before=["node_a"],  # Pause before node_a
    interrupt_after=["node_b"],   # Pause after node_b
    checkpointer=checkpointer
)

# Run until interrupt
graph.invoke(inputs, config=config)

# Resume after human review
graph.invoke(None, config=config)
```

**Resuming with Command:**

```python
from langgraph.types import Command

# Resume with user input
response = graph.invoke(
    Command(resume="user's choice"),
    config=config
)
```


#### 6. Streaming

LangGraph supports multiple streaming modes.[^48][^49][^50]

**Stream Modes:**


| Mode | Description |
| :-- | :-- |
| `values` | Stream full state after each step |
| `updates` | Stream only state changes |
| `messages` | Stream LLM tokens |
| `custom` | Stream custom data |
| `debug` | Stream debug information |

**Using Stream Modes:**

```python
# Stream state updates
for chunk in graph.stream(
    {"messages": [...]},
    stream_mode="updates"
):
    print(chunk)

# Stream LLM tokens
for message_chunk, metadata in graph.stream(
    {"topic": "ice cream"},
    stream_mode="messages"
):
    if message_chunk.content:
        print(message_chunk.content, end="", flush=True)
```

**Multiple Stream Modes:**

```python
for mode, chunk in graph.stream(
    inputs,
    stream_mode=["updates", "custom"]
):
    print(f"{mode}: {chunk}")
```

**Custom Streaming:**

```python
from langgraph.config import get_stream_writer

def my_node(state):
    writer = get_stream_writer()

    for chunk in ["Hello", " ", "World"]:
        writer(chunk)  # Stream custom data

    return {"message": "Hello World"}
```


#### 7. Subgraphs

Subgraphs break complex workflows into modular components.[^51][^52][^53][^54]

**Basic Subgraph:**

```python
# Define subgraph
subgraph_builder = StateGraph(State)
subgraph_builder.add_node("sub_node_1", sub_node_1)
subgraph = subgraph_builder.compile()

# Add subgraph as node in parent graph
parent_builder = StateGraph(State)
parent_builder.add_node("subgraph", subgraph)
parent_builder.add_node("other_node", other_node)

graph = parent_builder.compile()
```

**Streaming from Subgraphs:**

```python
for chunk in graph.stream(
    {"foo": "input"},
    stream_mode="updates",
    subgraphs=True  # Include subgraph updates
):
    print(chunk)
```


#### 8. Map-Reduce Pattern

Map-reduce enables parallel processing of tasks.[^55][^56][^57]

```python
from langgraph.constants import Send

class OverallState(TypedDict):
    subjects: list[str]
    jokes: Annotated[list, operator.add]

class JokeState(TypedDict):
    subject: str

def generate_subjects(state: OverallState):
    # Generate list of subjects
    return {"subjects": ["cats", "dogs", "birds"]}

def generate_joke(state: JokeState):
    # Generate joke for one subject
    joke = f"Joke about {state['subject']}"
    return {"jokes": [joke]}

def continue_to_jokes(state: OverallState):
    # Map: create a node for each subject
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

# Build graph
graph = StateGraph(OverallState)
graph.add_node("generate_subjects", generate_subjects)
graph.add_node("generate_joke", generate_joke)

# Conditional edge that fans out
graph.add_conditional_edges(
    "generate_subjects",
    continue_to_jokes,
    ["generate_joke"]
)
```


#### 9. Prebuilt Components

LangGraph provides ready-to-use components for common patterns.[^58][^59][^60]

**ToolNode:**

```python
from langgraph.prebuilt import ToolNode, tools_condition

tools = [search_tool, calculator_tool]

# Automatically execute tools
tool_node = ToolNode(tools)

# Add to graph
graph.add_node("tools", tool_node)
graph.add_conditional_edges(
    "agent",
    tools_condition,  # Routes to tools if needed
    {
        "tools": "tools",
        "__end__": "__end__"
    }
)
```

**create_react_agent:**

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=checkpointer
)

result = agent.invoke(
    {"messages": [("human", "What's the weather?")]},
    config={"configurable": {"thread_id": "1"}}
)
```


### Design Patterns

#### 1. Routing Pattern

Route inputs to specialized nodes based on context.[^61]

```python
def route(state: State) -> str:
    if "pricing" in state["input"]:
        return "pricing_node"
    elif "refund" in state["input"]:
        return "refund_node"
    return "general_node"

graph.add_conditional_edges("router", route, {
    "pricing_node": "pricing_node",
    "refund_node": "refund_node",
    "general_node": "general_node"
})
```


#### 2. Orchestrator-Worker Pattern

One orchestrator delegates tasks to multiple workers.[^61]

```python
from langgraph.types import Send

def orchestrator(state: State):
    # Break down task into subtasks
    sections = planner.invoke(state["topic"])
    return {"sections": sections.sections}

def worker(state: WorkerState):
    # Complete one subtask
    result = llm.invoke(state["section"])
    return {"completed_sections": [result.content]}

def assign_workers(state: State):
    # Create worker for each section
    return [Send("worker", {"section": s}) for s in state["sections"]]

graph.add_conditional_edges("orchestrator", assign_workers, ["worker"])
```


#### 3. Multi-Agent Collaboration

**Tool Calling Pattern:**

```python
# One agent calls others as tools
supervisor_tools = [agent_a, agent_b, agent_c]

supervisor = create_agent(
    model=llm,
    tools=supervisor_tools
)
```

**Handoff Pattern:**

```python
# Agents transfer control to each other
def agent_a(state):
    if needs_specialist:
        return Command(goto="agent_b")
    return {"result": "done"}
```


### Best Practices

1. **Keep State Simple**: Use TypedDict with minimal fields[^36]
2. **Use Reducers Sparingly**: Only when you need to combine values[^36]
3. **Prefer Simple Edges**: Use conditional edges only when necessary[^36]
4. **Namespace Long-term Memory**: Organize persistent data with clear namespaces[^36]
5. **Choose Right Stream Mode**: Match streaming to UI needs (messages for chat, updates for dashboards)[^36]
6. **Implement Error Boundaries**: Handle errors at node, graph, and app levels[^36]
7. **Use Postgres for Production**: In-memory checkpointers don't persist across restarts[^62][^36]

***

## LangSmith Documentation

### Overview

LangSmith is a platform for developing, debugging, evaluating, and monitoring LLM applications.[^63][^64][^65][^66]

### Setup

```python
pip install -U langsmith

# Set environment variables
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=ls_...
export LANGSMITH_PROJECT=my-project
```


### Core Features

#### 1. Tracing

Tracing captures every step of LLM execution.[^65][^67][^68]

**Enable Tracing:**

```python
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "your-api-key"

# All LangChain calls are now traced automatically
```

**Manual Tracing:**

```python
from langsmith import traceable

@traceable
def my_function(user_input: str) -> str:
    result = llm.invoke(user_input)
    return result

# Function calls are now traced
my_function("Hello!")
```

**Wrapping OpenAI:**

```python
from langsmith.wrappers import wrap_openai
import openai

client = wrap_openai(openai.Client())

# All calls are now traced
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}]
)
```


#### 2. Datasets and Evaluation

Datasets store test cases for systematic evaluation.[^69][^70][^71][^72]

**Creating a Dataset:**

```python
from langsmith import Client

client = Client()

# Create dataset
dataset = client.create_dataset(
    dataset_name="my-test-cases",
    description="Test cases for my application"
)

# Add examples
client.create_example(
    inputs={"question": "What is 2+2?"},
    outputs={"answer": "4"},
    dataset_id=dataset.id
)
```

**Running Evaluations:**

```python
from langsmith import evaluate

def accuracy_evaluator(run, example):
    # Compare output to expected
    score = run.outputs["answer"] == example.outputs["answer"]
    return {"key": "accuracy", "score": int(score)}

results = evaluate(
    lambda inputs: my_app(inputs["question"]),
    data=client.list_examples(dataset_name="my-test-cases"),
    evaluators=[accuracy_evaluator],
    experiment_prefix="experiment-v1"
)
```

**Dataset Splits:**

```python
# Evaluate on specific splits
results = evaluate(
    target_function,
    data=client.list_examples(
        dataset_name="my-dataset",
        splits=["train", "test"]
    ),
    evaluators=[evaluator]
)
```


#### 3. Prompt Hub

Version control and manage prompts separately from code.[^73][^74][^75]

**Pushing Prompts:**

```python
from langchain import hub

# Push to hub
hub.push(
    "my-org/my-prompt:latest",
    prompt_template
)
```

**Pulling Prompts:**

```python
# Pull specific version
prompt = hub.pull("my-org/my-prompt:v1")

# Pull latest
prompt = hub.pull("my-org/my-prompt:latest")
```

**Versioning:**

Every push creates a unique commit hash, allowing rollback to any version.[^74]

#### 4. Annotation Queues

Streamline human review and feedback.[^76][^77][^78][^79]

**Creating an Annotation Queue:**

1. Navigate to "Annotation queues" in LangSmith UI
2. Click "+ New annotation queue"
3. Define name, description, and rubric
4. Add feedback keys for annotators[^77][^78]

**Adding Runs to Queue:**

```python
# Automatically via automation rules
# Or manually via UI selection
```

**Workflow:**

1. Filter runs (e.g., negative feedback)
2. Add to annotation queue
3. Annotators review and score
4. Add labeled examples to datasets
5. Use for evaluation or fine-tuning[^78][^79][^76]

#### 5. Monitoring and Production

Track application performance in production.[^80][^81][^82]

**Filtering Runs:**

- Latency: Identify slow runs
- Errors: Find breaking errors
- Feedback: Filter by user ratings
- Metadata/Tags: Segment by configuration
- Full-text search: Find specific keywords[^82]

**Monitoring Dashboard:**

View aggregate statistics:

- Latency trends
- Token usage
- Cost tracking
- Feedback scores
- Error rates[^82]

**Grouping by Metadata:**

```python
# Tag runs with metadata
metadata = {"model": "gpt-4", "env": "production"}

# Group monitoring by metadata key in UI
```


#### 6. Prompt Optimization

Promptim automates prompt improvement.[^83][^73]

**Setup:**

```bash
pip install -U promptim

promptim create task ./my-task \
    --name my-task \
    --prompt langchain-ai/starter-prompt:v1 \
    --dataset my-dataset \
    --description "Task description"
```

**Training:**

```bash
promptim train --task ./my-task/config.json
```

**How It Works:**

1. Compute baseline metrics on initial prompt
2. Loop over training examples
3. Use metaprompt to suggest improvements
4. Test improved prompt
5. Keep if better, discard if worse
6. Repeat until convergence[^83]

### Best Practices

1. **Enable Tracing Early**: Set up in development to debug issues[^67][^65]
2. **Version Datasets**: Pin dataset versions for consistent evaluation[^71]
3. **Mix Evaluation Methods**: Combine human review, heuristics, and LLM-as-judge[^71]
4. **Use Annotation Queues**: Route uncertain outputs for human review[^77][^71]
5. **Monitor Production**: Track latency, costs, and feedback in real-time[^82]
6. **Tag Strategically**: Use metadata for segmentation and comparison[^82]
7. **Close the Loop**: Feed production data back into datasets for continuous improvement[^70][^71]

***

## Advanced Tips and Tricks

### 1. Custom Tool Runtime Context

Access state and config in tools:

```python
from langchain.tools import ToolRuntime, tool

@tool
def get_user_pref(runtime: ToolRuntime):
    """Get user preference from context."""
    user_id = runtime.context.user_id
    return f"Preferences for {user_id}"
```


### 2. Dynamic Model Selection

Choose models based on conversation complexity:

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler):
    message_count = len(request.state["messages"])
    if message_count > 10:
        request.model = advanced_model
    else:
        request.model = basic_model
    return handler(request)
```


### 3. Response Format with Pydantic

Ensure structured responses:

```python
from dataclasses import dataclass

@dataclass
class ResponseFormat:
    answer: str
    confidence: float
    sources: list[str]

agent = create_agent(
    model=llm,
    tools=tools,
    response_format=ResponseFormat
)
```


### 4. Batch Processing

Process multiple inputs efficiently:

```python
# Batch invoke
results = chain.batch([
    {"topic": "AI"},
    {"topic": "ML"},
    {"topic": "DL"}
])
```


### 5. Retry Logic

Handle transient failures:

```python
from langchain.callbacks.base import BaseCallbackHandler

class RetryHandler(BaseCallbackHandler):
    def on_llm_error(self, error, **kwargs):
        # Implement retry logic
        pass
```


### 6. Cost Tracking

Monitor token usage and costs:

```python
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = chain.invoke({"topic": "AI"})
    print(f"Total cost: ${cb.total_cost}")
    print(f"Total tokens: {cb.total_tokens}")
```


### 7. Environment-Based Configuration

Separate dev/staging/prod configs:

```python
import os

config = {
    "model": os.getenv("MODEL", "gpt-3.5-turbo"),
    "temperature": float(os.getenv("TEMPERATURE", "0.7")),
    "max_tokens": int(os.getenv("MAX_TOKENS", "1000"))
}
```


***

## Common Workflows and Patterns

### 1. Simple RAG Pipeline

```python
# Load and split
loader = PyPDFLoader("document.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
chunks = splitter.split_documents(docs)

# Embed and store
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(chunks, embeddings)

# Retrieve and generate
def rag(question: str) -> str:
    docs = vector_store.similarity_search(question, k=3)
    context = "\n".join(doc.page_content for doc in docs)

    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    return llm.invoke(prompt)
```


### 2. Agentic RAG

```python
@tool
def retrieve(query: str):
    """Retrieve relevant documents."""
    return vector_store.similarity_search(query, k=2)

agent = create_agent(
    model=llm,
    tools=[retrieve],
    system_prompt="Use retrieval tool to answer questions."
)
```


### 3. Multi-Agent System

```python
# Define specialist agents
research_agent = create_agent(model, [search_tool], "Research specialist")
analyst_agent = create_agent(model, [analysis_tool], "Data analyst")
writer_agent = create_agent(model, [], "Content writer")

# Supervisor coordinates
supervisor = create_agent(
    model,
    tools=[research_agent, analyst_agent, writer_agent],
    system_prompt="Coordinate agents to complete tasks."
)
```


### 4. Human-in-the-Loop Approval

```python
def approval_node(state):
    # Pause for human review
    approved = interrupt(f"Approve this action? {state['action']}")
    return {"approved": approved}

graph.add_node("approval", approval_node)
```


### 5. Iterative Refinement

```python
def refine(state):
    for i in range(state.get("max_iterations", 3)):
        result = llm.invoke(state["draft"])
        if quality_check(result):
            return {"final": result}
        state["draft"] = improve(result)
    return {"final": state["draft"]}
```


***

## Troubleshooting Guide

### Common Issues

**1. Import Errors**

```python
# Use correct imports for your version
from langchain_core import __version__
if version.parse(__version__).minor < 3:
    from pydantic.v1 import BaseModel
else:
    from pydantic import BaseModel
```

**2. Streaming Not Working**

```python
# Ensure model supports streaming
llm = ChatOpenAI(streaming=True)

# Use correct stream mode
for chunk in chain.stream(input, stream_mode="messages"):
    print(chunk)
```

**3. Checkpointer Not Persisting**

```python
# Don't use MemorySaver in production
# Use PostgresSaver instead
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost:5432/db"
)
```

**4. Conditional Edges Not Working**

```python
# Ensure routing function returns valid node names
def route(state):
    # Must return exact node name
    return "node_a"  # Not "Node A" or "node-a"
```

**5. Tool Calls Failing**

```python
# Verify tool schema matches expectations
@tool
def my_tool(arg1: str, arg2: int) -> str:
    """Clear description is crucial."""
    return f"{arg1}: {arg2}"
```


***

This comprehensive guide covers the essential documentation, workflows, features, design patterns, and best practices for LangChain, LangGraph, and LangSmith in Python. Use it as a reference for building production-grade LLM applications with proper state management, evaluation, and monitoring.
<span style="display:none">[^100][^101][^102][^103][^104][^105][^106][^107][^84][^85][^86][^87][^88][^89][^90][^91][^92][^93][^94][^95][^96][^97][^98][^99]</span>

<div align="center">⁂</div>

[^1]: https://docs.langchain.com/oss/python/langchain/quickstart

[^2]: https://docs.langchain.com/oss/python/langchain/overview

[^3]: https://docs.langchain.com

[^4]: https://www.geeksforgeeks.org/artificial-intelligence/langchain/

[^5]: https://stephencollins.tech/posts/how-to-create-lcel-chains-in-langchain

[^6]: https://k21academy.com/ai-ml/langchain-expression-language/

[^7]: https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.prompt.PromptTemplate.html

[^8]: https://api.python.langchain.com/en/latest/core/prompts/langchain_core.prompts.prompt.PromptTemplate.html

[^9]: https://mirascope.com/blog/langchain-prompt-template

[^10]: https://blog.langchain.com/tool-calling-with-langchain/

[^11]: https://www.linkedin.com/pulse/langchain-function-calling-rutam-bhagat-scbuf

[^12]: https://docs.langchain.com/oss/python/langchain/tools

[^13]: https://docs.langchain.com/oss/python/langchain/agents

[^14]: https://airbyte.com/data-engineering-resources/using-langchain-react-agents

[^15]: https://latenode.com/blog/ai-frameworks-technical-infrastructure/langchain-setup-tools-agents-memory/langchain-react-agent-complete-implementation-guide-working-examples-2025

[^16]: https://api.python.langchain.com/en/latest/memory/langchain.memory.buffer.ConversationBufferMemory.html

[^17]: https://api.python.langchain.com/en/v0.0.354/memory/langchain.memory.buffer.ConversationBufferMemory.html

[^18]: https://www.aurelio.ai/learn/langchain-conversational-memory

[^19]: https://www.pinecone.io/learn/series/langchain/langchain-conversational-memory/

[^20]: https://docs.langchain.com/oss/python/langchain/rag

[^21]: https://docs.langchain.com/oss/python/langchain/retrieval

[^22]: https://python.langchain.com/v0.1/docs/integrations/retrievers/

[^23]: https://api.python.langchain.com/en/latest/output_parsers/langchain.output_parsers.structured.StructuredOutputParser.html

[^24]: https://www.linkedin.com/pulse/structured-outputs-from-llms-langchain-output-parsers-vijay-chaudhary-wgjqc

[^25]: https://www.youtube.com/watch?v=62CSR141VRE

[^26]: https://api.python.langchain.com/en/latest/callbacks/langchain_core.callbacks.streaming_stdout.StreamingStdOutCallbackHandler.html

[^27]: https://github.com/langchain-ai/langchain/discussions/16850

[^28]: https://docs.langchain.com/oss/python/langgraph/streaming

[^29]: https://docs.langchain.com/oss/javascript/integrations/providers/all_providers

[^30]: https://docs.langchain.com/oss/python/integrations/providers/all_providers

[^31]: https://docs.langchain.com/oss/python/langchain/models

[^32]: https://realpython.com/langgraph-python/

[^33]: https://www.datacamp.com/tutorial/langgraph-tutorial

[^34]: https://pypi.org/project/langgraph/0.0.25/

[^35]: https://www.langchain.com/langgraph

[^36]: https://www.swarnendu.de/blog/langgraph-best-practices/

[^37]: https://docs.langchain.com/oss/python/langgraph/graph-api

[^38]: https://www.youtube.com/watch?v=EKxoCVbXZwY

[^39]: https://stackoverflow.com/questions/79654297/conditional-edge-in-langgraph-is-not-working-as-expected

[^40]: https://dev.to/jamesli/advanced-langgraph-implementing-conditional-edges-and-tool-calling-agents-3pdn

[^41]: https://developer.couchbase.com/tutorial-langgraph-persistence-checkpoint/

[^42]: https://pypi.org/project/langgraph-checkpoint/

[^43]: https://docs.langchain.com/oss/python/langgraph/persistence

[^44]: https://dev.to/jamesbmour/interrupts-and-commands-in-langgraph-building-human-in-the-loop-workflows-4ngl

[^45]: https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/wait-user-input/

[^46]: https://www.youtube.com/watch?v=6t7YJcEFUIY

[^47]: https://docs.langchain.com/oss/python/langgraph/interrupts

[^48]: https://dev.to/jamesli/two-basic-streaming-response-techniques-of-langgraph-ioo

[^49]: https://docs.langchain.com/langsmith/streaming

[^50]: https://www.youtube.com/watch?v=YqBYU2_IUlA

[^51]: https://docs.langchain.com/oss/python/langgraph/use-subgraphs

[^52]: https://www.baihezi.com/mirrors/langgraph/how-tos/subgraph/index.html

[^53]: https://www.linkedin.com/posts/langchain_creating-sub-graphs-in-langgraph-for-activity-7223711413108662275-IOsv

[^54]: https://www.youtube.com/watch?v=Z8l7C031xkM

[^55]: https://www.youtube.com/watch?v=JQznvlSatPQ

[^56]: https://github.com/langchain-ai/langgraph/discussions/609

[^57]: https://www.youtube.com/watch?v=GMPFt-LrOWc

[^58]: https://www.baihezi.com/mirrors/langgraph/reference/prebuilt/index.html

[^59]: https://langchain-5e9cc07a-preview-brodyd-1754591744-fac1b99.mintlify.app/python/oss/prebuilts

[^60]: https://reference.langchain.com/python/langgraph/agents/

[^61]: https://docs.langchain.com/oss/python/langgraph/workflows-agents

[^62]: https://github.com/langchain-ai/langgraph/discussions/4375

[^63]: https://pypi.org/project/langsmith/

[^64]: https://docs.langchain.com/langsmith/home

[^65]: https://www.statsig.com/perspectives/langsmith-tracing-debug-llm-chains

[^66]: https://www.geeksforgeeks.org/data-science/debugging-and-testing-llms-in-langsmith/

[^67]: https://docs.langchain.com/langsmith/observability-quickstart

[^68]: https://www.datacamp.com/tutorial/introduction-to-langsmith

[^69]: https://docs.langchain.com/langsmith/manage-datasets

[^70]: https://www.langchain.com/evaluation

[^71]: https://www.statsig.com/perspectives/langsmith-datasets-managing-evaluation

[^72]: https://docs.langchain.com/langsmith/evaluation-concepts

[^73]: https://github.com/hinthornw/promptimizer

[^74]: https://youssefh.substack.com/p/hands-on-langsmith-course37-playground

[^75]: https://mirascope.com/blog/langsmith-prompt-management

[^76]: https://www.youtube.com/watch?v=3Ws5wOS9eko

[^77]: https://docs.langchain.com/langsmith/annotation-queues

[^78]: https://docs.langchain.com/langsmith/improve-judge-evaluator-feedback

[^79]: https://www.youtube.com/watch?v=jP24Z5Raje4

[^80]: https://www.youtube.com/watch?v=3Gcm27l-uyQ

[^81]: https://walkingtree.tech/langsmith-navigating-the-road-to-production-for-llm-apps/

[^82]: https://blog.langchain.com/langsmith-production-logging-automations/

[^83]: https://blog.langchain.com/promptim/

[^84]: https://python.langchain.com/api_reference/

[^85]: https://reference.langchain.com/python/

[^86]: https://reference.langchain.com/python/langsmith/

[^87]: https://pypi.org/project/langchain/

[^88]: https://python.langchain.com/api_reference/langchain/smith.html

[^89]: https://docs.langchain.com/langsmith/sdk

[^90]: https://github.com/langchain-ai/langsmith-docs

[^91]: https://github.com/langchain-ai/langgraph

[^92]: https://reference.langchain.com/python/langsmith/observability/sdk/

[^93]: https://www.langchain.com

[^94]: https://langchain-ai.github.io/langgraph/reference/

[^95]: https://www.datacamp.com/tutorial/building-langchain-agents-to-automate-tasks-in-python

[^96]: https://blog.langchain.com/building-langgraph/

[^97]: https://docs.langchain.com/oss/python/langchain/multi-agent

[^98]: https://datahub.io/@donbr/langgraph-unleashed/langgraph_deep_research

[^99]: https://latenode.com/blog/ai-frameworks-technical-infrastructure/langchain-setup-tools-agents-memory/langchain-python-tutorial-complete-beginners-guide-to-getting-started

[^100]: https://pub.towardsai.net/agentic-design-patterns-with-langgraph-5fe7289187e6

[^101]: https://www.datacamp.com/tutorial/langgraph-agents

[^102]: https://www.langchain.com/langsmith/observability

[^103]: https://www.youtube.com/watch?v=aHCDrAbH_go

[^104]: https://stackoverflow.com/questions/77964228/how-can-i-create-a-rag-chain-with-langchain-using-a-retriever-when-having-multip

[^105]: https://www.pinecone.io/learn/series/langchain/langchain-expression-language/

[^106]: https://www.youtube.com/watch?v=zCwuAlpQKTM

[^107]: https://www.youtube.com/watch?v=O0dUOtOIrfs
