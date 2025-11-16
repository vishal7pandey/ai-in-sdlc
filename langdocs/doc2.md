<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Think of what other documentation you might have missed, types of tooling available within langgraph.

Now I have comprehensive information about additional LangGraph tooling and features. Let me compile this into a detailed supplementary documentation section.

# Additional LangGraph Documentation - Advanced Tooling \& Features

This supplementary guide covers advanced LangGraph tooling, debugging capabilities, and deployment features that extend beyond the core functionality.

## Time Travel \& Debugging

### Time Travel Capabilities

Time travel is one of LangGraph's most powerful debugging features, allowing you to replay and fork from past executions.[^1][^2][^3]

**Two Core Actions:**

1. **Replaying**: Revisit and reproduce past agent actions
2. **Forking**: Explore alternative execution paths[^2][^1]

#### Replaying from Current State

```python
from langgraph.checkpoint.memory import MemorySaver

# Set up graph with checkpointer
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Run initial execution
config = {"configurable": {"thread_id": "1"}}
result = graph.invoke({"messages": [...]}, config=config)

# Replay from current state (pass None as input)
for event in graph.stream(None, config):
    print(event)
```


#### Replaying from Specific Checkpoint

```python
# Get all checkpoints for a thread
all_checkpoints = list(graph.get_state_history(config))

# Each checkpoint has a unique ID
for state in all_checkpoints:
    print(state.config["configurable"]["checkpoint_id"])
    print(state.next)

# Replay from specific checkpoint
checkpoint_config = {
    "configurable": {
        "thread_id": "1",
        "checkpoint_id": "xyz"  # Specific checkpoint ID
    }
}

for event in graph.stream(None, checkpoint_config):
    print(event)
```

**Key Benefit**: The graph efficiently replays by leveraging cached results from previously executed nodes instead of re-executing them.[^1]

#### Forking - Exploring Alternative Paths

```python
# Update state at a specific checkpoint
checkpoint_config = {
    "configurable": {
        "thread_id": "1",
        "checkpoint_id": "xyz"
    }
}

# This creates a new forked checkpoint
graph.update_state(checkpoint_config, {"messages": [updated_message]})

# Continue execution from the fork
# A new checkpoint ID "xyz-fork" is created
fork_config = {
    "configurable": {
        "thread_id": "1",
        "checkpoint_id": "xyz-fork"
    }
}

for event in graph.stream(None, fork_config):
    print(event)
```

**Use Cases:**

- Debug mistakes: Identify where and why errors occurred
- Understand reasoning: Analyze steps that led to successful results
- Explore alternatives: Test different execution paths[^1]


### Dynamic Breakpoints

Dynamic breakpoints allow conditional interruption based on runtime state.[^4][^5][^6]

```python
from langgraph.types import NodeInterrupt

def step_with_conditional_interrupt(state: State):
    """Node that conditionally interrupts execution"""

    # Check condition
    if len(state["input"]) > 5:
        raise NodeInterrupt(
            f"Received input longer than 5 characters: {state['input']}"
        )

    # Continue normal execution
    print("---Step executing---")
    return state
```

**Inspecting Interrupt Information:**

```python
# After interruption, inspect state
state = graph.get_state(config)

print(state.next)  # Next node to execute
print(state.tasks)  # Task information including interrupt details

# Output shows:
# {
#   "id": "...",
#   "name": "step2",
#   "interrupts": [{
#     "value": "Received input longer than 5 characters: hello world",
#     "when": "during"
#   }]
# }
```

**Resuming After Dynamic Interrupt:**

Option 1: Update state to satisfy condition

```python
# Update state to meet interrupt condition
await graph.update_state(config, {"input": "short"})

# Resume execution
for event in graph.stream(None, config):
    print(event)
```

Option 2: Skip the interrupted node entirely

```python
# Update state "as" the interrupted node to skip it
await graph.update_state(config, None, "step2")

# Continue from next node
for event in graph.stream(None, config):
    print(event)
```


## Double Texting Management

Double texting occurs when users send multiple messages before the agent completes processing the first one.[^7][^8][^9]

**Types of Double Texting:**

1. **Classic**: Multiple messages expecting one cohesive reply
2. **Rapid-fire**: Quick succession of related messages
3. **Interrupt**: New information while processing first message[^8]

### Strategies

**Reject Strategy** (LangGraph Cloud/Deployment only):

```python
# Rejects new runs and continues with original
# Available in deployment configurations
```

**Debounce Pattern:**

```python
import asyncio

class DoubleTextHandler:
    def __init__(self, debounce_time=1.0):
        self.debounce_time = debounce_time
        self.active_debounce = None

    async def process_message(self, message):
        # Cancel existing debounce
        if self.active_debounce:
            self.active_debounce.cancel()

        # Create new debounce
        self.active_debounce = asyncio.create_task(
            self._debounced_process(message)
        )

    async def _debounced_process(self, message):
        await asyncio.sleep(self.debounce_time)
        # Process accumulated messages
        return await self.graph.invoke(message)
```

**Note**: Full double-texting support with strategies like "reject", "rollback", and "interrupt" is available in LangGraph Cloud/Deployment, not in the open-source framework.[^9][^7]

## Command API

The Command API combines state updates with routing in a single return value.[^10][^11][^12]

### Basic Usage

```python
from langgraph.types import Command
from typing import Literal

def node_with_command(state: State) -> Command[Literal["node_b", "node_c", END]]:
    """Node that routes and updates state simultaneously"""

    # Decide where to go
    if state["count"] > 10:
        next_node = "node_b"
    else:
        next_node = "node_c"

    # Return command with update and routing
    return Command(
        update={"foo": state["foo"] + "_updated"},
        goto=next_node
    )
```


### Command Parameters

| Parameter | Description |
| :-- | :-- |
| `update` | State update to apply (like returning a dict) |
| `goto` | Next node(s) to execute |
| `resume` | Value to resume with (for interrupts) |
| `graph` | Target graph for multi-graph scenarios |

### Advantages Over Conditional Edges

**Traditional Approach** (separate routing and updates):

```python
def router(state):
    return "next_node"  # Can't update state here

def node(state):
    return {"value": "updated"}  # Can't route here

graph.add_conditional_edges("node", router, {...})
```

**Command Approach** (combined):

```python
def node(state):
    return Command(
        update={"value": "updated"},
        goto="next_node"
    )

# No conditional edge needed!
graph.add_node("node", node)
```


## Send API for Dynamic Parallelization

The Send API enables dynamic parallel execution with unknown object counts.[^13][^14][^15]

### Dynamic State Distribution

```python
from langgraph.types import Send

class OverallState(TypedDict):
    items: list[str]
    results: Annotated[list, operator.add]

class ItemState(TypedDict):
    item: str
    result: str

def process_item(state: ItemState):
    """Process single item - runs in parallel"""
    result = f"Processed: {state['item']}"
    return {"results": [result]}

def fan_out_to_items(state: OverallState):
    """Dynamically create parallel executions"""
    # Create a Send for each item
    return [
        Send("process_item", {"item": item})
        for item in state["items"]
    ]

# Build graph
graph = StateGraph(OverallState)
graph.add_node("process_item", process_item)

# Conditional edge that fans out
graph.add_conditional_edges(
    "generate_items",
    fan_out_to_items,
    ["process_item"]
)
```


### Use Case: Trip Booking Example

```python
class TripBookingState(TypedDict):
    reservations: list[str]  # ["hotel", "flight", "car"]
    booking_details: Annotated[list, operator.add]

def book_reservation(state):
    """Book single reservation - runs in parallel"""
    booking = llm.invoke(f"Book {state['reservation']}")
    return {"booking_details": [booking]}

def distribute_bookings(state: TripBookingState):
    """Create parallel booking tasks"""
    return [
        Send("book_reservation", {"reservation": res})
        for res in state["reservations"]
    ]
```

**Key Features:**

- Unknown object counts handled dynamically
- Each Send creates independent parallel execution
- State can differ between parent graph and sent nodes
- Results aggregated via reducers[^14][^13]


## State Reducers (Advanced)

Reducers control how state updates are combined when multiple nodes write to the same key.[^16][^17][^18]

### Default Behavior (Overwrite)

```python
class State(TypedDict):
    value: str  # Last write wins

# If two nodes run in parallel and both update "value",
# only one update survives (non-deterministic which one)
```


### Using Annotated with Reducers

```python
from typing import Annotated
import operator

class State(TypedDict):
    # Append to list instead of overwriting
    items: Annotated[list[int], operator.add]

    # Sum values instead of overwriting
    total: Annotated[int, operator.add]

# Now parallel nodes can safely write to same keys
def node_a(state):
    return {"items": [1, 2], "total": 5}

def node_b(state):
    return {"items": [3, 4], "total": 10}

# Result: {"items": [1, 2, 3, 4], "total": 15}
```


### Built-in Message Reducer

```python
from langgraph.graph.message import add_messages
from typing import Annotated

class State(TypedDict):
    messages: Annotated[list, add_messages]

# add_messages intelligently handles message deduplication
# and updates by ID
```


### Custom Reducers

```python
def custom_reducer(left: list, right: int | None) -> list:
    """Custom logic for combining values"""
    if right is not None:
        return left + [right]
    return left

class State(TypedDict):
    values: Annotated[list, custom_reducer]
```

**When to Use Reducers:**

- Parallel node execution writing to same keys
- Accumulating results (lists, sums, etc.)
- Message history management
- Map-reduce patterns[^17][^19][^16]


## Visualization

### Mermaid Diagrams

LangGraph provides multiple visualization options.[^20][^21][^22]

#### Generate PNG

```python
from IPython.display import Image, display

# Using Mermaid.ink API (default, no dependencies)
image = graph.get_graph().draw_mermaid_png()
display(Image(image))

# Save to file
with open("graph.png", "wb") as f:
    f.write(image)
```


#### With Custom Styling

```python
from langchain_core.runnables.graph import (
    CurveStyle,
    NodeColors,
    MermaidDrawMethod
)

image = graph.get_graph().draw_mermaid_png(
    curve_style=CurveStyle.LINEAR,
    node_colors=NodeColors(
        start="#ffdfba",
        end="#baffc9",
        other="#fad7de"
    ),
    wrap_label_n_words=9,
    background_color="white",
    padding=10,
    draw_method=MermaidDrawMethod.API
)
```


#### ASCII Visualization

```python
# Simple text-based graph
print(graph.get_graph().draw_ascii())

# Output:
#     +-----------+
#     | __start__ |
#     +-----------+
#           *
#           *
#           *
#       +------+
#       | node |
#       +------+
#           *
#           *
#           *
#     +---------+
#     | __end__ |
#     +---------+
```


#### Mermaid Code Export

```python
# Get Mermaid syntax for external rendering
mermaid_code = graph.get_graph().draw_mermaid()
print(mermaid_code)

# Paste output into https://mermaid.live for web rendering
```


## LangGraph Studio

Studio is a specialized IDE for developing and debugging LangGraph applications.[^23][^24][^25][^26]

### Setup and Launch

```bash
# Install LangGraph CLI
pip install -U "langgraph-cli[inmem]"

# Start development server
langgraph dev

# Output:
# > Ready!
# > - API: http://localhost:2024
# > - Docs: http://localhost:2024/docs
# > - LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```


### Key Features

#### 1. Hot Reload

Edit code and Studio automatically detects changes and reloads.[^25][^26][^27]

#### 2. Replay from Any Node

```python
# In Studio UI:
# 1. Click on any node in execution history
# 2. Click "Replay from here"
# 3. Graph re-executes from that point
```


#### 3. Edit State and Re-run

Modify agent state directly in Studio and continue execution.[^27]

#### 4. Breakpoints for Iteration

```python
# Set breakpoint in Studio UI before a node
# Run graph, it pauses at breakpoint
# Edit prompt in code
# Re-run node to test changes
# Continue or restart as needed
```


#### 5. Attach Debugger

```bash
# Install debugpy
pip install debugpy

# Start with debug port
langgraph dev --debug-port 5678
```

**VS Code `launch.json`:**

```json
{
  "name": "Attach to LangGraph",
  "type": "debugpy",
  "request": "attach",
  "connect": {
    "host": "0.0.0.0",
    "port": 5678
  }
}
```


#### 6. Pull Production Traces

```python
# In Studio, connect to production LangSmith traces
# Pull down specific traces to debug locally
# Replay them with full state inspection
```


### Debugging Workflows

**Workflow 1: Isolate and Fix Errors**

1. Run graph and identify failing node
2. Click on failed node to see error details
3. Fix code in editor
4. Click "Rerun from node" to test fix
5. Continue execution[^25]

**Workflow 2: Iterative Improvement**

1. Set breakpoint before target node
2. Run until breakpoint
3. Inspect state and outputs
4. Edit prompt or logic in code
5. Rerun node to test changes
6. Repeat until satisfied[^25]

## Scheduled Tasks \& Cron Jobs

LangGraph deployments support cron-based scheduling.[^28][^29][^30]

### Creating Cron Jobs

```python
from langgraph_sdk import get_client

client = get_client(url="<DEPLOYMENT_URL>")
assistant_id = "agent"

# Create thread
thread = await client.threads.create()

# Schedule cron job on thread
cron_job = await client.crons.create_for_thread(
    thread["thread_id"],
    assistant_id,
    schedule="27 15 * * *",  # 3:27 PM daily
    input={"messages": [{"role": "user", "content": "Daily summary"}]}
)
```


### Stateless Cron Jobs

```python
# Create cron without specific thread
cron_job = await client.crons.create(
    assistant_id,
    schedule="0 9 * * MON",  # 9 AM every Monday
    input={"messages": [{"role": "user", "content": "Weekly report"}]}
)
```


### Managing Cron Jobs

```python
# List all crons
crons = await client.crons.list()

# Delete cron
await client.crons.delete(cron_job["cron_id"])
```


### Cron Expression Examples

| Schedule | Expression | Description |
| :-- | :-- | :-- |
| Every day 3 PM | `0 15 * * *` | Daily at 15:00 |
| Every Monday 9 AM | `0 9 * * MON` | Weekly |
| Every hour | `0 * * * *` | Hourly |
| Every 15 minutes | `*/15 * * * *` | Frequent |

### Use Cases

- **Ambient agents**: Background tasks running at set times
- **Daily digests**: News summaries, reports
- **Monitoring**: Health checks, data validation
- **Scheduled research**: Long-running analysis tasks[^29][^31]

**Important**: Cron jobs run on the deployment infrastructure, not your local machine. Shutting down your local system doesn't affect scheduled jobs.[^32]

## Deployment Options

LangGraph offers multiple deployment strategies.[^33][^34][^35]

### 1. Cloud (Fully Managed)

- Control plane and data plane in LangChain cloud
- Connect GitHub repos and deploy from UI
- Automatic CI/CD
- Simplest option[^34]


### 2. Hybrid

- Data plane in your cloud
- Control plane managed by LangChain
- Build Docker images locally
- Deploy via control plane UI
- Supported: Kubernetes, ECS (coming)[^36][^33][^34]

```bash
# Build Docker image
langgraph build -t my-agent:v1

# Push to your registry
docker push your-registry/my-agent:v1

# Deploy from control plane UI
```


### 3. Self-Hosted

**Full Platform** (control plane + data plane):

- Run entire platform in your infrastructure
- Requires Enterprise license[^34]

**Standalone Data Plane** (lightweight):

```bash
# Build image
langgraph build

# Docker Compose setup
```

**`docker-compose.yml` example:**

```yaml
services:
  langgraph-redis:
    image: redis:6

  langgraph-postgres:
    image: postgres:16

  langgraph-api:
    image: my-agent:latest
    environment:
      - REDIS_URI=redis://langgraph-redis:6379
      - POSTGRES_URI=postgresql://user:pass@langgraph-postgres:5432/db
    ports:
      - "8123:8000"
```


### Kubernetes Deployment

```bash
# Install via Helm
helm install langgraph-agent ./langgraph-helm-chart \
  --set image.repository=your-registry/my-agent \
  --set image.tag=v1 \
  --set langsmith.apiKey=<YOUR_KEY>
```

**Requirements:**

- PostgreSQL (only supported database)
- Redis (task queue)
- LangSmith API key (monitoring)[^37]


## Async \& Batch Processing

LangChain Runnables support async and batch operations natively.[^38][^39][^40]

### Batch Processing

```python
# Batch multiple inputs
results = chain.batch([
    {"topic": "AI"},
    {"topic": "ML"},
    {"topic": "DL"}
])

# Async batch
results = await chain.abatch([...])
```


### Async Methods

```python
# All standard methods have async equivalents
result = await chain.ainvoke({"topic": "AI"})

# Async streaming
async for chunk in chain.astream({"topic": "AI"}):
    print(chunk)

# Async batch with completion tracking
async for idx, result in chain.abatch_as_completed([...]):
    print(f"Completed: {idx}")
```


### Custom Async Implementation

```python
from langchain_core.runnables import RunnableLambda

def sync_function(x: int) -> int:
    return x + 1

async def async_function(x: int) -> int:
    await asyncio.sleep(0.1)
    return x + 1

# Provide both sync and async
runnable = RunnableLambda(
    sync_function,
    afunc=async_function
)

# Uses sync_function
runnable.invoke(1)

# Uses async_function
await runnable.ainvoke(1)
```


## Prebuilt LangGraph Components

LangGraph provides high-level prebuilt components.[^41][^42][^43]

### ToolNode

```python
from langgraph.prebuilt import ToolNode

# Automatically execute tool calls
tools = [search_tool, calculator_tool]
tool_node = ToolNode(tools)

# Invoke with tool call messages
result = tool_node.invoke({
    "messages": [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "search",
                "args": {"query": "weather"},
                "id": "1"
            }]
        )
    ]
})
```


### ValidationNode

```python
from langgraph.prebuilt import ValidationNode
from pydantic import BaseModel, field_validator

class SelectNumber(BaseModel):
    a: int

    @field_validator("a")
    def a_must_be_valid(cls, v):
        if v != 37:
            raise ValueError("Only 37 is allowed")
        return v

validation_node = ValidationNode([SelectNumber])

# Validates tool calls against schema
result = validation_node.invoke({
    "messages": [
        AIMessage(
            "",
            tool_calls=[{
                "name": "SelectNumber",
                "args": {"a": 42},
                "id": "1"
            }]
        )
    ]
})
```


### create_react_agent

```python
from langgraph.prebuilt import create_react_agent

# Quickly create a ReAct agent
agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=checkpointer,
    state_modifier="You are a helpful assistant"
)

# Invoke with standard interface
result = agent.invoke(
    {"messages": [("human", "What's the weather?")]},
    config={"configurable": {"thread_id": "1"}}
)
```


### tools_condition

```python
from langgraph.prebuilt import tools_condition

# Automatically route to tools or end
graph.add_conditional_edges(
    "agent",
    tools_condition,  # Routes based on tool calls
    {
        "tools": "tools",
        "__end__": "__end__"
    }
)
```


***

## Summary of Additional Tooling

**Debugging \& Development:**

- Time travel (replay \& fork) for advanced debugging
- Dynamic breakpoints for conditional interrupts
- LangGraph Studio for visual debugging and hot reload
- Attach debugger support for step-through debugging

**Advanced Execution:**

- Send API for dynamic parallel execution
- State reducers for parallel write safety
- Command API for combined routing and updates
- Double texting management strategies

**Production Features:**

- Cron-based scheduling for background tasks
- Multiple deployment options (Cloud, Hybrid, Self-hosted)
- Kubernetes and Docker support
- Async and batch processing

**Visualization \& Monitoring:**

- Mermaid diagram generation (PNG, ASCII, code)
- Real-time graph visualization
- Production trace replay
- State inspection at any checkpoint

This comprehensive tooling makes LangGraph production-ready with enterprise-grade debugging, deployment, and monitoring capabilities.[^12][^23][^28][^1]
<span style="display:none">[^44][^45][^46][^47][^48][^49][^50][^51][^52][^53][^54][^55][^56][^57][^58][^59][^60][^61][^62][^63][^64][^65][^66][^67][^68][^69][^70][^71][^72][^73][^74][^75][^76][^77][^78][^79]</span>

<div align="center">⁂</div>

[^1]: https://langchain-ai.github.io/langgraphjs/concepts/time-travel/

[^2]: https://dragonforest.in/time-travel-in-langgraph/

[^3]: https://galileo.ai/blog/mastering-agents-langgraph-vs-autogen-vs-crew

[^4]: https://langchain-ai.github.io/langgraphjs/how-tos/dynamic_breakpoints/

[^5]: https://changelog.langchain.com/announcements/langgraph-python-dynamic-breakpoints-error-tracking-in-checkpointer-and-custom-configs

[^6]: https://www.youtube.com/watch?v=LvkzIS3OV9w

[^7]: https://docs.langchain.com/langsmith/double-texting

[^8]: https://www.reddit.com/r/LangChain/comments/1kzz9kr/solving_the_double_texting_problem_that_makes/

[^9]: https://dev.to/sreeni5018/langgraph-uncovered-building-stateful-multi-agent-applications-with-llms-part-i-p86

[^10]: https://github.com/langchain-ai/langgraph/discussions/2654

[^11]: https://langchain-ai.github.io/langgraphjs/reference/classes/langgraph.Command.html

[^12]: https://blog.langchain.com/command-a-new-tool-for-multi-agent-architectures-in-langgraph/

[^13]: https://dev.to/sreeni5018/leveraging-langgraphs-send-api-for-dynamic-and-parallel-workflow-execution-4pgd

[^14]: https://forum.langchain.com/t/parallel-execution-with-supervisor-pattern/1665

[^15]: https://github.com/langchain-ai/langgraph/discussions/609

[^16]: https://www.youtube.com/watch?v=UrVno_5wB08

[^17]: https://www.reddit.com/r/LangChain/comments/1hxt5t7/help_me_understand_state_reducers_in_langgraph/

[^18]: https://dragonforest.in/define-state-in-langgraph/

[^19]: https://github.com/langchain-ai/langgraph/discussions/2975

[^20]: https://kitemetric.com/blogs/visualizing-langgraph-workflows-with-get-graph

[^21]: https://www.baihezi.com/mirrors/langgraph/how-tos/visualization/index.html

[^22]: https://www.youtube.com/watch?v=Xn7pPopFK1s

[^23]: https://changelog.langchain.com/announcements/langgraph-studio-v2-run-and-debug-production-traces-locally

[^24]: https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/

[^25]: https://www.youtube.com/watch?v=5vEC0Y4sV8g

[^26]: https://docs.langchain.com/oss/python/langgraph/studio

[^27]: https://www.datacamp.com/tutorial/langgraph-studio

[^28]: https://docs.langchain.com/langsmith/cron-jobs

[^29]: https://www.youtube.com/watch?v=9DRn9RpR2vA

[^30]: https://forum.langchain.com/t/what-are-the-best-way-to-schedule-langgraph-workflows-on-windows/1747

[^31]: https://x.com/LangChainAI/status/1880300661533012194

[^32]: https://github.com/langchain-ai/langgraph/discussions/5357

[^33]: https://www.cohorte.co/blog/navigating-langgraphs-deployment-landscape-picking-the-right-fit-for-your-ai-projects

[^34]: https://langchain-5e9cc07a-preview-an07au-1754595026-9c8a87e.mintlify.app/langgraph-platform/deployment-options

[^35]: https://docs.langchain.com/langsmith/deploy-with-control-plane

[^36]: https://docs.langchain.com/langsmith/hybrid

[^37]: https://konghq.com/blog/engineering/build-a-multi-llm-ai-agent-with-kong-ai-gateway-and-langgraph

[^38]: https://reference.langchain.com/python/langchain_core/runnables/

[^39]: https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.base.Runnable.html

[^40]: https://mirascope.com/blog/langchain-runnables

[^41]: https://pypi.org/project/langgraph-prebuilt/

[^42]: https://www.baihezi.com/mirrors/langgraph/reference/prebuilt/index.html

[^43]: https://langchain-5e9cc07a-preview-brodyd-1754591744-fac1b99.mintlify.app/python/oss/prebuilts

[^44]: https://www.linkedin.com/posts/chiefj_how-to-build-a-simple-chatbot-using-langgraph-activity-7245326361488183297-Jlpo

[^45]: https://docs.langchain.com/langsmith/reject-concurrent

[^46]: https://docs.langchain.com/oss/python/langgraph/interrupts

[^47]: https://docs.langchain.com/oss/python/langgraph/use-time-travel

[^48]: https://forum.langchain.com/t/streaming-subgraphs-results-in-duplicate-messages-as-subgraph-updates-parent/1659

[^49]: https://github.com/langchain-ai/langgraphjs/issues/708

[^50]: https://www.langchain.com/langgraph

[^51]: https://github.com/langchain-ai/langgraph/issues/3062

[^52]: https://docs.langchain.com/oss/python/langgraph/graph-api

[^53]: https://docs.langchain.com/oss/python/langgraph/persistence

[^54]: https://aiproduct.engineer/tutorials/langgraph-tutorial-parallel-tool-execution-state-management-unit-23-exercise-1

[^55]: https://www.youtube.com/watch?v=5Autf3g1NMs

[^56]: https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.graph_mermaid.draw_mermaid_png.html

[^57]: https://mermaid.live

[^58]: https://docs.mermaidchart.com/mermaid-oss/syntax/flowchart.html

[^59]: https://docs.langchain.com/langsmith/deploy-standalone-server

[^60]: https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph

[^61]: https://www.kaggle.com/code/ksmooi/langgraph-dynamic-chart-generator-agent-teams

[^62]: https://langfuse.com/guides/cookbook/integration_langgraph

[^63]: https://langchain-ai.github.io/langgraphjs/reference/classes/langgraph.StateGraph.html

[^64]: https://blog.gopenai.com/building-parallel-workflows-with-langgraph-a-practical-guide-3fe38add9c60

[^65]: https://docs.langchain.com/oss/python/langgraph/overview

[^66]: https://docs.langchain.com/oss/python/langgraph/pregel

[^67]: https://github.com/langchain-ai/langgraph

[^68]: https://github.com/langchain-ai/langgraph/discussions/2212

[^69]: https://reference.langchain.com/python/langgraph/graphs/

[^70]: https://docs.langchain.com/oss/python/langgraph/quickstart

[^71]: https://langchain-ai.github.io/langgraph/how-tos/create-react-agent-manage-message-history/

[^72]: https://forum.langchain.com/t/does-langgraph-have-plans-to-eliminate-its-use-of-annotated-in-state-property-types/2072

[^73]: https://docs.langchain.com/langsmith/quick-start-studio

[^74]: https://api.python.langchain.com/en/latest/core/runnables/langchain_core.runnables.base.RunnableBinding.html

[^75]: https://langchain-opentutorial.gitbook.io/langchain-opentutorial/01-basic/07-lcel-interface

[^76]: https://www.youtube.com/watch?v=o9CT5ohRHzY

[^77]: https://langchain-ai.github.io/langgraphjs/reference/functions/langgraph.task.html

[^78]: https://python.langchain.com/v0.2/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html

[^79]: https://blog.langchain.com/langgraph-cloud/
