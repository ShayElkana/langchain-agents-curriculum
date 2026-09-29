# LangChain, LangGraph, and Deep Agents

A learning guide for building LLM features in Python. Read it once from top to bottom. After that, use the vocabulary table and the "which layer" table as references.

The ideas stack. Each part assumes the one before it.

## How to use this guide

1. Read [Part 0](#part-0-vocabulary) once. Come back whenever a word feels fuzzy.
2. Read [Parts 1–4](#part-1-langchain) to decide *which tool* a problem needs.
3. Read [Part 5](#part-5-integrating-into-an-existing-python-monolith) before you put any of this in a real app.
4. Do [Part 6](#part-6-hands-on-curriculum) using the files below.

Work in order. Debug any file with Python: Current File (.env).

Set `OPENAI_API_KEY` in a `.env` file before you run a lesson that calls the model. Step 9's tests must not need that key.

| Step | Lesson | Exercise |
| --- | --- | --- |
| 1 | `steps/step1_simple_llm_call.py` | `steps/step1_exercise_chat_repl.py` |
| 2 | `steps/step2_structured_output.py` | `steps/step2_exercise_code_review.py` |
| 3 | `steps/step3_tool_calling.py` | `steps/step3_exercise_two_tools.py` |
| 4 | `steps/step4_simple_agent.py` | `steps/step4_exercise_refund_limits.py` |
| 5 | `steps/step5_langgraph_workflow.py` | `steps/step5_exercise_urgent_route.py` |
| 6 | `steps/step6_checkpoints_and_hitl.py` | `steps/step6_exercise_threads_and_resume.py` |
| 7 | `steps/step7_deep_agent.py` | `steps/step7_exercise_deep_vs_plain.py` |
| 8 | `steps/step8_monolith_facade.py` | `steps/step8_exercise_service_adapter.py` |
| 9 | `steps/step9_production_checklist.py` | `steps/step9_exercise_ci_tests.py` |

Run a lesson with `uv run python <path>`. Run Step 5's routing tests with `uv run pytest steps/step5_exercise_urgent_route.py -q`, and Step 9 with `uv run pytest tests -q`.

### What each step teaches

| Step | What you learn |
| --- | --- |
| 1 | A call is messages in, one `AIMessage` out. The model is stateless. "Memory" is you resending the history, and that resend is why input tokens climb. |
| 2 | A Pydantic schema plus `with_structured_output` returns a validated object. Field descriptions are the prompt. A valid shape is not a correct judgment. |
| 3 | The model never runs a tool. `@tool` publishes a schema, you execute it, and you reply with a `ToolMessage`. Failures should be messages the model can read. |
| 4 | `create_agent` is that loop until the model answers in text. The model picks the path. Always set a recursion limit. A dollar cap belongs in the tool, not only in the prompt. |
| 5 | A workflow is a `StateGraph` where your code chooses the next node. Conditional edges are plain Python, so you can unit-test the route with no model. |
| 6 | A checkpointer keyed by `thread_id` is session memory and crash recovery. `interrupt()` pauses for a human; `Command(resume=...)` continues. Put side effects after the interrupt. |
| 7 | A deep agent adds files, a todo list, and subagents so a long task does not fill one context window. A short lookup is usually cheaper as a plain `create_agent`. |
| 8 | The rest of the app calls a facade and gets plain types. LangChain stays inside `myapp/ai/`. Tools call the service layer, so auth and audit stay in one place. |
| 9 | CI uses a scripted chat model, not the API. `thread_id` comes from a domain id. Every `invoke` gets a recursion limit. The rest of production is a checklist, not a new framework. |

## Contents

- [Part 0: Vocabulary](#part-0-vocabulary)
- [Part 1: LangChain](#part-1-langchain)
- [Part 2: LangGraph](#part-2-langgraph)
- [Part 3: Deep Agents](#part-3-deep-agents)
- [Part 4: Architecture](#part-4-architecture--how-it-all-fits)
- [Part 5: Integrating into a monolith](#part-5-integrating-into-an-existing-python-monolith)
- [Part 6: Hands-on curriculum](#part-6-hands-on-curriculum)
  - [Step 1 — A simple LLM call](#step-1--a-simple-llm-call)
  - [Step 2 — Structured output](#step-2--structured-output)
  - [Step 3 — Tool calling](#step-3--tool-calling)
  - [Step 4 — A simple agent](#step-4--a-simple-agent)
  - [Step 5 — A LangGraph workflow](#step-5--a-langgraph-workflow)
  - [Step 6 — Checkpoints and human approval](#step-6--checkpoints-and-human-approval)
  - [Step 7 — Deep agents](#step-7--deep-agents)
  - [Step 8 — A facade in a monolith](#step-8--a-facade-in-a-monolith)
  - [Step 9 — Production checklist](#step-9--production-checklist)

---

## Part 0: Vocabulary

Terms are defined here so later parts can use them without stopping. Come back to this table whenever a word feels fuzzy.

| Term | Meaning |
| --- | --- |
| **LLM** | A function that takes text in and produces text out, one token (word fragment) at a time. It is **stateless**. It remembers nothing between calls. Every "conversation" is an illusion created by re-sending the entire history on every call. |
| **Prompt** | The text you send to the model. A **system prompt** is a special instruction block that frames behavior ("You are a support assistant for Acme Corp..."). |
| **Context / context window** | Everything the model can see in a single call: the system prompt, the conversation history, retrieved documents, tool results. It has a hard size limit. Deciding what goes into it is called **context engineering**. |
| **Tool** | A regular function (yours) that the model is allowed to *request*. The model cannot execute anything. It can only emit a structured message such as "please call `get_order_status(order_id='A123')`". Your code runs the function and sends the result back. |
| **Tool calling** | The protocol above. You describe your functions to the model (name, description, JSON schema of arguments). The model responds with plain text, or with a request to invoke one or more tools. |
| **Agent** | A loop: call the model, and if it requested tools, run them, feed the results back, and repeat until the model answers with plain text. The defining trait is that the **model decides the control flow at runtime**. |
| **Workflow** | You decide the control flow in code. The model only fills in steps. Contrast this with an agent. |
| **Orchestration** | The machinery that sequences model calls, tool executions, branches, and retries. LangGraph is an orchestration framework. |
| **State** | The data that flows through a multi-step process: the message history, plus anything else you accumulate (a draft, extracted fields, a retry counter). |
| **Memory** | An overloaded word. **Short-term memory** usually means the message history within one conversation. **Long-term memory** means facts persisted across conversations (for example, "this user prefers terse answers"), stored in a database. |
| **Checkpoint** | A saved snapshot of a graph's state after each step, written to storage. It enables resuming after crashes, pausing for human approval, and "time travel" debugging. |
| **Human-in-the-loop (HITL)** | Pausing an automated process so a human can approve, edit, or reject an action before it happens (for example, before an agent sends a refund). |
| **RAG** | Retrieval-Augmented Generation. Before calling the model, search a knowledge base (usually via vector similarity over embeddings) for relevant documents and paste them into the prompt. That is how a model answers questions about your data without retraining. It is its own topic; this guide mentions it where it matters and does not teach it. |
| **Thread** | In LangGraph, one persistent conversation or session, identified by a `thread_id`. Checkpoints are stored per thread. |

---

## Part 1: LangChain

**After this part you can answer:** what LangChain is for, when the raw provider SDK is enough, and which names to learn first.

### What is it?

LangChain is a Python (and JavaScript) library of standard interfaces and building blocks for LLM applications: a uniform way to call any model provider, define tools, get structured output, and compose these into applications. As of v1.0 it also ships the standard agent implementation, `create_agent`.

### What problem does it solve?

Three real ones:

1. **Provider lock-in and API divergence.** OpenAI, Anthropic, Google, Bedrock, and Ollama all have different SDKs, message formats, and tool-calling syntaxes. LangChain normalizes them behind one interface, so switching `model="openai:gpt-..."` to `model="anthropic:claude-..."` is a one-line change.
2. **Boilerplate you would otherwise hand-roll.** Converting Python functions into tool schemas, parsing model output into validated Pydantic objects, retrying transient failures, and streaming are solved, tested code.
3. **The agent loop.** Call the model, run tools, repeat. It sounds trivial and has many edge cases: parallel tool calls, malformed arguments, message trimming, error recovery. `create_agent` is a hardened implementation.

### Why not just call the LLM API directly?

You can, and for a single one-off call you should. Do it once as a learning exercise.

The raw API becomes painful when you need several of these:

- multiple providers, or the option to switch later
- tool calling (hand-written JSON schemas and a dispatch loop)
- validated structured output
- conversation persistence
- streaming with intermediate steps
- observability across a multi-step pipeline

That is the code you would end up writing yourself, under deadline. Think of LangChain the way you think of `requests` / `httpx` versus hand-rolling HTTP over sockets: standardized plumbing.

> **The honest counterargument.** Older LangChain (0.x) accumulated abstraction bloat, and "just use the SDK" became a meme. The 1.0 release cut the namespace down to the essentials and moved legacy pieces into `langchain-classic`. The modern core is lean. If a colleague warns you about LangChain, ask which version they mean.

### Concepts to learn first

Learn these, in this order. The later ones depend on the earlier ones.

| Concept | What it is |
| --- | --- |
| Chat models and `init_chat_model` | The uniform model interface. |
| Messages | `SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`. The shared language of everything that follows. |
| Tools | The `@tool` decorator. How a Python function becomes a schema the model sees. |
| Structured output | Binding a Pydantic model so the response is a validated object. |
| `create_agent` | The standard agent loop. |
| Middleware | Hooks that run before or after the model, or wrap tool calls. This is the 1.x way to customize agents. |

### What to skip at first

| Skip | Why |
| --- | --- |
| `langchain-classic` and 0.x tutorials: `LLMChain`, `ConversationChain`, `initialize_agent`, `AgentExecutor` | Deprecated. If a tutorial imports these, close the tab. |
| Deep dives on LCEL pipe syntax (`prompt \| model \| parser`) | Know it exists so you can read old code. Do not build complex logic with it. LangGraph replaced that role. |
| The integration zoo | Hundreds of document loaders and vector stores. Learn one when a task needs it. |
| RAG and vector stores | Important later, and separate from the agent skills in this guide. |

---

## Part 2: LangGraph

**After this part you can answer:** when a Python script is enough, and when you need a durable graph.

### What is it, and what problem does it solve?

LangGraph is a low-level orchestration runtime for multi-step LLM applications. You express the application as a **graph**: steps (**nodes**) connected by transitions (**edges**), operating on shared **state**, with optional persistence of every step.

Real LLM applications are rarely one call. A typical shape is: classify the request, then either look up the order or draft an escalation, then validate, maybe loop back, and pause for human approval before refunding.

Plain Python gets you a working script. It stops being enough when you also need to:

- resume after a crash mid-run
- pause for a human for three days without holding a process open
- replay a failed run from step 3
- stream intermediate progress to a UI

LangGraph gives you those properties because state transitions are explicit and checkpointed.

> **Mental model.** LangGraph is to LLM steps what Temporal or Airflow is to distributed jobs: a durable state-machine runtime. If you have written a workflow engine or a saga, you already understand most of it.

### Relationship to LangChain

`langgraph` is a separate library from the same team, designed to be used with LangChain.

- LangChain provides the **components**: models, messages, tools.
- LangGraph provides the **runtime** that sequences them.

Since v1.0, LangChain's `create_agent` is implemented as a LangGraph graph. You can use LangGraph without LangChain. In practice you use both.

### Core vocabulary

| Word | Precise meaning |
| --- | --- |
| **State** | A `TypedDict` (a typed dictionary schema) that flows through the graph. Every node receives the current state and returns a partial update. |
| **Node** | A plain Python function: `(state) -> dict`. It might call a model, call a database, or do pure logic. LangGraph does not care which. |
| **Edge** | "After node A, go to node B." A **conditional edge** is a function that inspects state and decides where to go next. That is how you get branches and loops. |
| **Reducer** | A rule for how a state update merges with existing state. The default is "overwrite the key." For message lists, use `add_messages`, which appends. |
| **Checkpoint** | After each node, the full state is snapshotted to a checkpointer (in-memory, SQLite, Postgres, or Redis), keyed by `thread_id`. |
| **Workflow vs agent** | A **workflow** is a graph whose edges are decided by your code: deterministic, testable, predictable. An **agent** is a graph with a loop where the model's tool calls decide the path. LangGraph can express both. Choosing between them is the main design decision. |

### When to use LangGraph

- Multi-step LLM processes with branching, loops, or parallel steps.
- Anything that must pause for a human and resume later.
- Anything long-running that must survive process restarts.
- When you need to inspect or replay execution history (production debugging).
- Custom agent architectures that `create_agent` does not cover.

### When to skip LangGraph

- A single LLM call, or a fixed linear two-step pipeline. Write Python functions. A graph with no branches is ceremony.
- Deterministic business logic with no LLM in it. Python already is that language.
- When `create_agent` already does the job. Do not hand-build the tool loop for its own sake.
- General job scheduling or ETL. Use Celery, Temporal, or Airflow. LangGraph is not a task queue.

Code for this part comes later in the curriculum, building from a five-line graph up to a Postgres-checkpointed, human-approved production pattern.

---

## Part 3: Deep Agents

**After this part you can answer:** why a plain agent falls apart on long tasks, and which extra tools exist to stop that.

### What are they?

`deepagents` is a separate package from the LangChain team. It is an opinionated, batteries-included agent harness built on `create_agent`, and therefore on LangGraph.

One function, `create_deep_agent`, gives you an agent that already has:

| Feature | What the agent gets |
| --- | --- |
| **Planning** | An optional `write_todos` tool. The agent keeps a visible task list for long tasks. |
| **Virtual filesystem** | `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`. Storage is pluggable: in-memory state by default; real disk, a LangGraph store, or composite routing if you configure it. The agent uses files as scratch space so the context window does not fill up. |
| **Subagents** | A built-in `task` tool that spawns short-lived child agents. Each child has a fresh context window and returns only a summary to the parent. |
| **Context management** | Automatic offloading of large tool results, and summarization of old history. |
| **Also included** | Long-term memory, skills (instruction files loaded on demand), and filesystem permissions. |

This is the same shape as a coding agent (Claude Code, Cursor): a plan, files as memory, and delegated subtasks. The package generalizes that architecture beyond coding.

### How this differs from a normal agent

A plain `create_agent` loop degrades on long tasks (dozens to hundreds of steps) for one reason: **everything piles into a single message history.** The context window fills with raw tool output, the model loses the plan, and quality collapses.

The deep-agent features are context-window strategies:

| Trick | What it fixes |
| --- | --- |
| Todo list | The plan stays visible. It is not buried 40 messages back. |
| Files | A 5,000-line research dump lives outside the conversation. The agent re-reads only what it needs. |
| Subagents | Noisy exploration stays in the child. The child can read twenty documents and return a two-paragraph summary. |

> **Mental model.** A plain agent is a contractor you phone with one job, working from memory. A deep agent is that contractor with a whiteboard (todos), a filing cabinet (filesystem), and junior staff they can delegate research to (subagents).

### When to use them

Long-running, open-ended, multi-stage tasks, for example:

- "Research this topic across many sources and write a report."
- "Analyze this codebase and propose a migration plan."
- Multi-document synthesis.
- Any task where you expect 20 or more tool calls.

### When they are over-engineering

Most of the time. A support-ticket classifier, a Q&A bot over docs, or an agent with four tools that finishes in five steps is simpler, cheaper, faster, and easier to debug as one of these:

- a plain `create_agent`
- a LangGraph workflow
- a single structured-output call

Every deep-agent feature adds tokens, latency, and surface area.

> **Design rule.** Start at the simplest layer. Move up only when you have *seen* the failure that the next layer solves. Example: you watch the agent forget its plan mid-task. That "escalate only on observed failure" rule is the best design guidance in this ecosystem.

---

## Part 4: Architecture — how it all fits

**After this part you can answer:** which layer a given problem belongs on.

Each layer is built on the one below. You can enter at any layer.

```text
┌─────────────────────────────────────────────────────────┐
│  deepagents          Opinionated agent harness          │
│                      (planning, files, subagents)        │
├─────────────────────────────────────────────────────────┤
│  langchain.agents    create_agent — the standard        │
│                      tool-calling loop + middleware     │
├─────────────────────────────────────────────────────────┤
│  langgraph           Orchestration runtime: graphs,     │
│                      state, checkpoints, interrupts     │
├─────────────────────────────────────────────────────────┤
│  langchain-core      Primitives: models, messages,      │
│  + provider packages tools, structured output           │
│  (langchain-openai, langchain-anthropic, ...)           │
└─────────────────────────────────────────────────────────┘
```

### Which layer for which problem

| Problem shape | Use |
| --- | --- |
| One model call: classify, summarize, extract | `langchain-core`: model + structured output. No graph, no agent. |
| Fixed multi-step pipeline. Your code decides the branches. | LangGraph `StateGraph` (a workflow) |
| The model must choose among tools, about 3–15 steps | `create_agent` |
| Long-horizon, open-ended, research-like tasks | `deepagents` |
| Pause for a human, crash recovery, or resume later | Whatever layer you are already on, plus a LangGraph checkpointer. All of these layers support it, because they run on LangGraph. |

> **Common mistake.** Entering too high: reaching for an agent when a structured-output call would do. More autonomy means less predictability, and harder testing. Spend autonomy like a budget.

---

## Part 5: Integrating into an existing Python monolith

**After this part you can answer:** where the AI code lives, what it is allowed to call, and what you must have in place before production.

The correct answer is boring. The AI layer is another module with a narrow interface. Treat it the way you would treat a payment-provider integration.

### Where the AI layer lives

The rest of the monolith never imports LangChain types.

```text
myapp/
├── domain/              # existing business logic — no LangChain imports
├── services/            # existing service layer
├── api/                 # existing routes
└── ai/                  # everything new lives here
    ├── client.py        # facade: the plain-Python API the monolith calls
    ├── models.py        # model factory (init_chat_model + config)
    ├── tools/           # thin adapters wrapping existing services as tools
    ├── graphs/          # LangGraph workflow definitions
    ├── agents/          # create_agent / deep agent definitions
    ├── prompts/         # prompts as versioned files or constants
    └── evals/           # golden datasets and eval scripts
```

### Two boundary rules

These two rules prevent most of the architectural pain.

**1. Inward boundary.** The monolith calls `ai.client.triage_ticket(ticket) -> TriageResult`. Plain functions. Plain dataclasses or Pydantic in, plain objects out. No `AIMessage` ever crosses this line. LangChain stays swappable, and its fast-moving API stays inside one package.

**2. Outward boundary.** Tools never touch the ORM or raw SQL. Each tool is a short adapter that calls your existing service layer. An agent then gets the same permission checks, validation, and audit logging as any other caller. The agent is another client of your services, and it is an untrusted one.

### Introduce it in phases

| Phase | What you add | Autonomy |
| --- | --- | --- |
| **1** | One LLM call. Low-risk, high-tolerance work: summarize a ticket, draft a reply a human reviews, tag or classify content. Structured output, validated, a human sees the result. | None. This phase builds config, keys, logging, and evals. |
| **2** | A LangGraph workflow. Multi-step, and you control the flow: classify, then route, then draft, then validate. | Still none. |
| **3** | `create_agent` with tools that can only look things up. | Read-only. |
| **4** | Write-capable tools behind human approval. Interrupts before anything that mutates data. | Writes, with a person in the loop. |
| **5** | Autonomous writes, or deep agents. Only where phases 1–4 proved value, and only with evals and monitoring already in place. | Full. Often you never need this phase. |

### Operational checklist

#### Dependencies

The ecosystem releases quickly. Pin exact versions (`uv` or Poetry lockfile). Isolate them in an optional dependency group. Read changelogs before bumping. Treat `langchain*` upgrades like framework upgrades.

Minimal set:

- `langchain`
- `langgraph`
- one provider package, for example `langchain-openai`
- `langgraph-checkpoint-postgres` when you need durable checkpoints
- `deepagents` only if you reach that layer

#### Configuration

Model name, temperature, timeouts, and token limits are config, not code. Put them in your existing settings system (for example `pydantic-settings`).

You will want a different model per environment (a cheap model in dev and CI, a capable one in production) and per use case. One factory function owns the model string. Do not scatter hardcoded `"gpt-..."` strings through the codebase.

#### Logging and observability

Failures here are often semantic: the model did something unhelpful, and there is no stack trace. You need a full trace of every prompt, every tool call, every token count, latency, and cost, per run.

Options:

- [LangSmith](https://www.langchain.com/langsmith) — first-party, set up with environment variables
- self-hosted [Langfuse](https://langfuse.com/)
- OpenTelemetry

Also log a correlation id (`thread_id` or run id) into your normal application logs so the two views join. Track cost per feature from day one.

#### Testing

Test in layers. Only the last layer calls a real model.

| Layer | How |
| --- | --- |
| Tools | They are plain functions. Normal unit tests. No LLM. |
| Graph routing | Unit-test nodes and conditional-edge functions with hand-built state dicts. |
| Agent loop wiring | A fake chat model. `langchain-core` ships fakes you script with canned tool-call responses. Deterministic, free, fine in CI. |
| Evals | A golden dataset of input → expected outcome, run against the real model. Score with exact match for classifications, or an LLM-as-judge for prose. Run nightly or before deploy, not on every commit. |

Assert structure and outcomes. Do not assert exact LLM strings in CI.

#### Error handling and retries

Three different failure kinds, three different responses:

| Level | Examples | What to do |
| --- | --- | --- |
| Transport | Rate limits, timeouts | Provider classes have built-in `max_retries`. Also set an explicit timeout. |
| Semantic | Invalid tool arguments | Structured-output validation catches these. Retry with the error appended so the model can correct itself. Cap at 2–3 attempts. |
| Step | A node in a graph fails | LangGraph retry policies per node. Checkpointing means a crashed run resumes from the last completed node. |

Also configure model fallbacks (`.with_fallbacks([...])`) so a provider outage degrades the feature instead of taking it down.

#### Persistence

In production, the checkpointer is Postgres (`PostgresSaver`). You likely already run Postgres.

- Map `thread_id` to a real domain entity, for example `f"ticket-{ticket.id}"`.
- Store that mapping in your own tables.
- Plan retention and cleanup. Checkpoints are conversation data. GDPR and PII rules apply.

### Architectural mistakes

These show up in real systems. Avoid them.

1. **LangChain types leak through the codebase.** See the inward boundary.
2. **Tools bypass the service layer and hit the database.** No authorization, no audit trail. An LLM-controlled SQL path is a security incident waiting. Treat all model output as untrusted user input.
3. **An agent for everything.** Debug time scales with autonomy. Use a workflow or a single call when that is enough.
4. **Mutating tools without idempotency keys.** Agents retry. `issue_refund` must be safe to call twice.
5. **No cost, latency, or step budget.** A looping agent spends real money. Always set a recursion or step limit.
6. **A 30-second LLM call inside a web request handler.** Run long graphs in your background-job system and poll or notify, the same way you would any slow job.
7. **Prompts as untracked strings.** Version them and review them in pull requests. A prompt change is a behavior change. It needs an eval run the way a code change needs tests.

---

## Part 6: Hands-on curriculum

Work in order. Debug any file with Python: Current File (.env).

Read the lesson, run the file, then do the exercise. The file map and the "what you learn" table are at the top of this guide.

### Setup

Python 3.10 or newer. This repo is already initialized. From the project root:

```bash
uv add langchain langchain-openai langgraph
```

Put the key in `.env` (this repo loads it with `python-dotenv`):

```bash
OPENAI_API_KEY=sk-...
```

The examples use OpenAI. Swapping the provider string for Anthropic, Google, or Ollama is the point of the abstraction.

### Step 1 — A simple LLM call

**Concept.** One request, one response.

**Why it exists.** It is the atom everything else is built from. It teaches the two facts behind all later machinery:

1. The model is stateless.
2. Everything is messages.

> **Analogy.** You are texting a knowledgeable stranger who has amnesia. Each text must contain everything they need. They remember nothing from the previous text.

```python
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage

model = init_chat_model("openai:gpt-5", temperature=0)
response = model.invoke([
    SystemMessage("You are a concise assistant for the engineering team at Acme."),
    HumanMessage("Explain idempotency in one paragraph."),
])
print(response.content)         # the text
print(response.usage_metadata)  # token counts — this is your cost
```

**Line by line.**

| Line | What it does |
| --- | --- |
| `init_chat_model` | Provider-agnostic factory. The string picks the provider package and the model. |
| `temperature=0` | Minimizes randomness. Use it for anything you will test. |
| `.invoke([...])` | Takes a list of messages and returns an `AIMessage`. |

Internally, your messages are serialized into the provider's HTTP format, sent to the API, and parsed back into a common shape.

The runnable version in this repo adds a timeout, retries, and a model name read from the environment. Open `steps/step1_simple_llm_call.py` and compare it to the snippet above.

**Statelessness demo.** Run this:

```python
model.invoke([HumanMessage("My name is Shay.")])
r = model.invoke([HumanMessage("What's my name?")])
print(r.content)  # it has no idea
```

"Memory" is you re-sending history. Append each `AIMessage` and the next `HumanMessage` to a growing list. Every chat product you have used works this way.

**Production use.** Summarization, classification, and drafting. This is also where you set `timeout` and `max_retries` on the model.

**Pitfalls.**

- Forgetting that history is manual, until checkpointers show up later.
- Unbounded history growth. You re-send the whole conversation every turn, so token cost grows fast.
- Putting raw user input straight into the system prompt. That is a prompt-injection path.

**Exercise.** Build a terminal chat REPL that keeps a message list, and print cumulative token usage after each turn. Watch the input-token number climb.

Run `uv run python steps/step1_exercise_chat_repl.py`. The three TODOs in that file are the exercise: append the user message, call the model with the full history, append the reply, and print running token totals.

---

### Step 2 — Structured output

**Concept.** A model's native output is prose. Production code cannot branch on prose. Hand the model a schema and get a validated Python object back.

**Why it exists.** Classification, extraction, and routing all need a shape your `if` statements can trust.

> **Analogy.** Prose is a colleague describing a ticket in a Slack message. Structured output is the same colleague filling in a form with a dropdown, a 1–5 score, and a checkbox. You can route the form. You cannot reliably route the Slack message.

Run `uv run python steps/step2_structured_output.py`.

The schema is the lesson. The docstring and every `Field(description=...)` are sent to the model. They are instructions, not comments.

```python
from typing import Literal
from pydantic import BaseModel, Field

class TicketTriage(BaseModel):
    """Triage assessment of a customer support ticket."""

    category: Literal["billing", "bug", "feature_request", "other"] = Field(
        description="The single best-fitting category for the ticket."
    )
    severity: int = Field(
        ge=1, le=5,
        description="Impact on the customer: 1=trivial, 3=degraded, 5=critical outage.",
    )
    summary: str = Field(description="One-sentence summary in neutral tone.")
    needs_human: bool = Field(
        description="True if a human should review before any reply is sent."
    )
```

Three things in that class:

| Piece | What it does for the model |
| --- | --- |
| Docstring and `Field(description=...)` | Prompt engineering. Write them like docs for a new hire. |
| `Literal[...]` | Constrains the answer. The model cannot reply `"billing-ish"`. |
| `ge=1, le=5` | Gives Pydantic something to reject if the model misbehaves. |

The call replaces the `AIMessage` with your object:

```python
triager = model.with_structured_output(TicketTriage)
result = triager.invoke(f"Triage this support ticket:\n\n{ticket}")

# result is a TicketTriage. Branch on it. No regex.
if result.needs_human or result.severity >= 4:
    route_to_human(result)
```

**Under the hood.** `with_structured_output` turns the class into JSON Schema and sends that schema to the provider. Print it:

```python
print(TicketTriage.model_json_schema())
```

Pass `include_raw=True` when parsing fails in a real system. You get the raw `AIMessage`, the parsed object, and `parsing_error` in one dict.

**The lesson that matters.** Validated is not correct. An ambiguous ticket ("the export is slow, can you add a progress bar?") always comes back as a valid `TicketTriage`. Run it ten times and the category can change. The shape is guaranteed. The judgment is not. That is why evals exist (Part 5).

**Production use.** Ticket triage, extracting fields from email, classifying intent before a workflow branch.

**Pitfalls.**

- Vague field descriptions. The model reads them. Weak descriptions produce weak, inconsistent labels.
- Treating a valid object as a true one. Score it against a golden set.
- Asking for a list by passing a bare finding class. `with_structured_output` takes one schema. Wrap the list: `class CodeReview(BaseModel): findings: list[CodeReviewFinding]`.

**Exercise.** `steps/step2_exercise_code_review.py`.

1. Define `CodeReviewFinding` (file, line, severity as a `Literal`, explanation, suggested fix) and a wrapper with `findings: list[...]`.
2. Review the sample diff. Read the diff yourself first. It hides at least three real problems: a removed null check, a removed "refund exceeds total" check, and a bare `except: pass`.
3. Degradation experiment: delete the `Field` descriptions and change `severity` from `Literal` to plain `str`. Re-run. Schema quality is output quality.

---

### Step 3 — Tool calling

**Concept.** Models cannot check your database, your order system, or today's date. A tool connects language to a system.

**Why it exists.** This is the bridge from "the model talks" to "the model can look something up." Step 4 hides the loop. You should see it once with nothing hidden.

> **The line to over-learn.** The model never executes anything. It emits a structured request — "please call `get_order_status(order_id='A123')`" — and your code decides whether to run it, runs it, and sends the result back.

Run `uv run python steps/step3_tool_calling.py`.

**1. `@tool` turns a function into a schema.** The docstring is the description the model reads. Type hints become the argument schema. The model never sees your function body.

```python
from langchain.tools import tool

@tool
def get_order_status(order_id: str) -> str:
    """Look up the current status and ETA of a customer order by its ID.

    Order IDs are one letter followed by three digits, e.g. 'A123'.
    """
    order = ORDERS.get(order_id)
    if order is None:
        # A described failure. The model can read this and recover.
        return f"No order found with ID {order_id!r}. IDs look like 'A123'."
    return f"Order {order_id}: {order['status']}, ETA {order['eta']}."
```

**2. Bind the tool, then read `tool_calls`.** The model's "answer" may be a request, not text.

```python
model_with_tools = model.bind_tools([get_order_status])
ai_msg = model_with_tools.invoke([HumanMessage("Where is my order A123?")])

ai_msg.text        # often empty
ai_msg.tool_calls  # [{"name": "get_order_status", "args": {"order_id": "A123"}, "id": "..."}]
```

**3. You execute, and you answer with a `ToolMessage`.** The `tool_call_id` pairs this result with that specific request. The model may ask for several tools in one message, so loop.

```python
from langchain.messages import HumanMessage, ToolMessage

messages = [HumanMessage("Where is my order A123?")]
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)  # the request stays in the history

for tc in ai_msg.tool_calls:
    result = get_order_status.invoke(tc["args"])
    messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

final = model_with_tools.invoke(messages)  # now it can answer in text
print(final.text)
```

That loop is the whole protocol: request, execute, respond. `create_agent` in Step 4 is this loop, repeated until `tool_calls` is empty.

**4. Return error messages. Do not raise.** If the user asks for order `999`, the tool returns `"No order found..."`. The model reads that string and tells the user the id looks wrong. If the tool raised, the run would die.

**Production use.** Any lookup the model should not invent: order status, account balance, "what day is it," search over your docs.

**Pitfalls.**

- A useless docstring (`"""Tool."""`). The model chooses tools by reading descriptions. Selection falls apart when the text does.
- Hardcoding `get_order_status.invoke` when you have more than one tool. Dispatch by the name the model requested.
- Raising from a tool on a normal "not found." That kills the run. Reserve exceptions for genuine bugs in your code.
- Letting a tool touch the database directly. Call your service layer (Part 5, outward boundary).

**Exercise.** `steps/step3_exercise_two_tools.py`.

1. Add `cancel_order`. Unknown id: return a message. Status `"shipped"`: refuse in the function, not in the prompt. Otherwise delete it from `ORDERS` and confirm.
2. Replace the hardcoded call with a dispatch table: `tools_by_name = {t.name: t for t in tools}`, then `tools_by_name[tc["name"]].invoke(tc["args"])`. That table is what `create_agent` keeps internally.
3. Docstring experiment: replace both docstrings with `"Tool."` and re-run. Watch it pick the wrong tool.
4. Stretch: ask for orders `A123` and `B456` in one sentence. `ai_msg.tool_calls` can hold several requests. Your loop should already handle that.

---

### Step 4 — A simple agent

**Concept.** The Step 3 loop, repeated until the model answers with plain text.

**Why it exists.** Once you trust the loop, you should not reimplement parallel calls, malformed arguments, and "keep going" by hand. `create_agent` is that hardened loop.

> **Analogy.** Step 3 is driving a manual transmission through one intersection. An agent is cruise control on a route you did not write down. You chose the car (the model), the map (the tools), and the rules of the road (the system prompt). The model chooses the turns.

Run `uv run python steps/step4_simple_agent.py`.

```python
from langchain.agents import create_agent

agent = create_agent(
    model=MODEL,
    tools=[get_order_status, get_carrier_info, open_lost_package_claim],
    system_prompt=(
        "You are a support agent for Acme. Investigate order issues using "
        "your tools before answering. Only open claims when the carrier "
        "data suggests the package is lost."
    ),
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Order A123 hasn't arrived, help!"}]},
    {"recursion_limit": 25},
)
```

Four lessons:

| # | Lesson |
| --- | --- |
| 1 | `create_agent` is your Step 3 loop, plus "keep going until there are no tool calls." |
| 2 | `result["messages"]` is the audit log. Read it. A typical path here is status, then carrier, then claim, then a text answer. That sequence is not in your code. |
| 3 | Always set `recursion_limit`. A confused agent will loop, and each loop is a paid model call. |
| 4 | The object `create_agent` returns is a compiled LangGraph graph: a model node, a tools node, and a conditional edge. If the model message has `tool_calls`, go to tools; otherwise end. Later checkpoints and interrupts attach to this graph for that reason. |

Print the trail the way `print_trail` does in the step file. If you only print the final sentence, you cannot tell which tools ran.

**Production use.** A support assistant with a handful of read tools, finishing in a few steps. This is the `create_agent` row of the Part 4 table. It is not a deep agent, and it is not a place for unbounded writes.

**Pitfalls.**

- No step cap. See lesson 3.
- Policy that lives only in the system prompt. "Never refund more than $50" in a prompt is a suggestion. A check inside `issue_refund` is a rule.
- Skipping the message trail when debugging. The bug is usually a tool result the model misunderstood, sitting in the middle of `result["messages"]`.

**Exercise.** `steps/step4_exercise_refund_limits.py`. This one is still yours to finish.

1. Implement `issue_refund(order_id, amount)`. If `amount > 50`, return an error message and do not record a refund. Otherwise append to `REFUNDS_ISSUED` and confirm. Do not raise.
2. Build a `create_agent` whose system prompt also says never refund more than $50 without a human. Set `recursion_limit`.
3. Run both cases in `__main__`: an honest $49.99 refund, and a $500 message that tries to talk the model out of the policy.
4. Optional: delete the cap from the function, keep the prompt, and re-run the $500 ask. If the refund lands in the ledger, you have watched why a prompt is not a control.

After both cases, `REFUNDS_ISSUED` should contain only the $49.99.

---

### Step 5 — A LangGraph workflow

**Concept.** Your code chooses the next node. The model only does work inside a node.

**Why it exists.** Step 4 lets the model pick the path. A lot of production work is the opposite: you already know the process (classify, then draft the matching reply, then validate). A graph makes those rails explicit and testable.

> **Analogy.** An agent is a taxi: the driver picks the route. A workflow is a train: you laid the tracks. Take the train when you know the process.

Run `uv run python steps/step5_langgraph_workflow.py`.

State is a `TypedDict`. Each node receives it and returns a partial update. LangGraph merges the keys you return.

```python
class TicketState(TypedDict):
    ticket_text: str
    category: str
    draft_reply: str
    approved: bool

def classify(state: TicketState) -> dict:
    result = model.with_structured_output(Classification).invoke(
        f"Classify this support ticket:\n\n{state['ticket_text']}"
    )
    return {"category": result.category}

def route_by_category(state: TicketState) -> str:
    return state["category"]  # the name of the next node
```

The edges are the design:

```python
builder = StateGraph(TicketState)
builder.add_edge(START, "classify")
builder.add_conditional_edges(
    "classify",
    route_by_category,
    {"billing": "billing", "bug": "bug", "other": END},
)
builder.add_edge("billing", "validate")
builder.add_edge("bug", "validate")
builder.add_edge("validate", END)
graph = builder.compile()
```

`validate` does not call a model. It checks length and bans the word "guarantee". Policy between LLM steps belongs in ordinary Python.

Four lessons:

| # | Lesson |
| --- | --- |
| 1 | State is a `TypedDict` that flows through every node. |
| 2 | A node is `(state) -> partial update`. |
| 3 | A conditional edge is a function you can unit-test with a hand-built dict. No LLM. |
| 4 | Nodes do not have to call a model. Validators and policy checks sit between LLM steps. |

The compiled object is the same kind of graph as `create_agent`. The difference is who chooses the edges.

**Production use.** Classify, then route, then draft, then validate. Phase 2 from Part 5.

**Pitfalls.**

- Using an agent because the steps feel "AI-ish," when the sequence is already known.
- Hiding routing rules in a prompt. `route_by_category` is the rule. A prompt cannot be unit-tested the same way.
- Putting a database write in a node that may re-run. Later, checkpoints resume by re-executing. Keep side effects idempotent.

**Exercise.** `steps/step5_exercise_urgent_route.py`.

1. Add `urgent: bool` to the state and to the classification schema.
2. Add an `escalate` node with a fixed template and `approved=False`. No model in that node.
3. If `urgent`, route to `escalate`. Otherwise keep the category map. Non-urgent `"other"` still goes to `END`.
4. Prove it with plain dicts: urgent billing → `"escalate"`, non-urgent billing → `"billing"`, non-urgent other → `"other"`.

Then run both:

```bash
uv run python steps/step5_exercise_urgent_route.py
uv run pytest steps/step5_exercise_urgent_route.py -q
```

The pytest run is the point. Try that with `create_agent` and you will feel the difference.

---

### Step 6 — Checkpoints and human approval

**Concept.** A checkpointer snapshots graph state after each step, keyed by `thread_id`. That one mechanism is both memory and a pause button.

**Why it exists.** Step 1 made you resend history by hand. A checkpointer stores it. The same snapshot lets a run pause for a human, the process exit, and another worker resume days later.

> **Analogy.** A save-game. Every step auto-saves. The load slot is `thread_id`.

Run `uv run python steps/step6_checkpoints_and_hitl.py`.

**Memory.** Two `invoke` calls, one config. The second call does not include the first message. The checkpointer still has it.

```python
agent = create_agent(
    model=MODEL,
    tools=[],
    system_prompt="You are concise. Remember facts the user tells you in this thread.",
    checkpointer=InMemorySaver(),
)
config = {"configurable": {"thread_id": "user-shay"}, "recursion_limit": 10}

agent.invoke({"messages": [{"role": "user", "content": "My name is Shay. I work at Acme."}]}, config)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is my name and where do I work?"}]},
    config,
)
```

A different `thread_id` is a blank slate. Without a checkpointer, `thread_id` does nothing.

**Human approval.** `interrupt()` inside the tool freezes the run and surfaces a payload. `Command(resume=...)` continues that same thread.

```python
@tool
def issue_refund(order_id: str, amount: float) -> str:
    """Issue a customer refund. Requires a human to approve first."""
    decision = interrupt({"action": "refund", "order_id": order_id, "amount": amount})
    if decision != "approve":
        return f"Refund of ${amount} for {order_id} was rejected by a reviewer."
    return f"Refunded ${amount} for order {order_id}."
```

The first `invoke` returns with `__interrupt__` set. Nothing has been refunded yet. A later `invoke(Command(resume="approve"), config)` on the same `thread_id` finishes the tool.

Code before `interrupt()` in that tool runs again on resume. Put the real side effect after the decision, or make it idempotent.

Four lessons:

| # | Lesson |
| --- | --- |
| 1 | `thread_id` is the session name. Map it to a domain entity (`ticket-4812`). Do not reuse it across unrelated conversations. |
| 2 | Without a checkpointer, `thread_id` does nothing useful. |
| 3 | `interrupt(payload)` freezes the run. `Command(resume=...)` continues it. |
| 4 | Code before `interrupt()` re-runs on resume. Side effects go after it. |

`InMemorySaver` dies with the process. The exercise's SQLite stretch is the same API with a file, which is the idea behind `PostgresSaver` in production.

**Production use.** Multi-turn support threads, and any write that must wait for a person. Phase 4 from Part 5.

**Pitfalls.**

- One shared `thread_id` for every user. That is one shared memory.
- Refunding, then asking. The pause has to happen before the mutation.
- Assuming an in-memory saver survives a restart. It does not.

**Exercise.** `steps/step6_exercise_threads_and_resume.py`.

1. Build `create_agent` with `InMemorySaver` (or `SqliteSaver` for the stretch).
2. Tell thread `user-a` your name. Ask thread `user-b`. B must not know it. Ask A again. A must.
3. Two refund threads. Resume one with `"approve"` and one with `"reject"`. Print both tool outcomes.
4. Stretch: swap in `SqliteSaver`, quit, run again with the same `thread_id`, and confirm A still remembers the name.

---

### Step 7 — Deep agents

**Concept.** `create_deep_agent` is `create_agent` plus a filesystem, optional todos, and subagents, aimed at tasks long enough to bury the plan in one message history.

**Why it exists.** You have seen the failure mode this fixes only if a plain agent lost the plot after many tool calls. This step lets you watch the harness, then compare it to the plain loop on a task that is too small to need it.

> **Analogy.** A plain agent works from memory. A deep agent has a whiteboard, a filing cabinet, and junior staff.

Run `uv run python steps/step7_deep_agent.py`.

The lesson uses a fake `knowledge_search` tool so you do not need a second API. A subagent named `researcher` gets that tool and a fresh context. The parent is told to write `/comparison.md`.

```python
agent = create_deep_agent(
    model=MODEL,
    tools=[knowledge_search],
    system_prompt=(
        "You are a research analyst. For comparison questions: look up "
        "each topic, write short notes with your file tools, then put a "
        "5-line comparison in /comparison.md."
    ),
    subagents=[researcher],
)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Compare LangGraph, Temporal, and create_agent..."}]},
    {"recursion_limit": 40},
)
```

`result["files"]` is the virtual filesystem when the agent wrote something. The compiled object is still a LangGraph graph. These runs are slower and more expensive than Step 4. The recursion cap is 40 on purpose, and it is still a cap.

**Production use.** Research across many sources, a migration plan over a large codebase, multi-document synthesis. Not a five-step support agent.

**Pitfalls.**

- Reaching for this because the package exists. On a three-bullet lookup, `create_agent` usually wins on time and cost.
- No recursion limit. A deep agent has more ways to wander.
- Letting the agent run shell commands when the task only needed the search tool.

**Exercise.** `steps/step7_exercise_deep_vs_plain.py`.

1. `create_deep_agent` with `knowledge_search` and one subagent. Ask it to write `/report.md` with three bullets.
2. Print the virtual file if it is there, otherwise the final message, and print elapsed seconds.
3. Run the same question with plain `create_agent` and only the search tool. Compare latency, trail length, and answer quality.

The likely result on this toy is that the plain agent is enough. That is the correct conclusion.

---

### Step 8 — A facade in a monolith

**Concept.** The rest of the app calls a plain function and gets a plain type. LangChain types stay inside `myapp/ai/`.

**Why it exists.** Parts 1–7 are the AI machinery. This step is the seam so the machinery can be swapped without rewriting routes, jobs, or domain code.

The call site you want:

```python
from myapp.ai.client import triage_ticket

result = triage_ticket(ticket_id=4812, text=raw)  # TriageResult, not an AIMessage
```

Run `uv run python steps/step8_monolith_facade.py`.

The toy layout:

```text
myapp/domain.py            # plain result types
myapp/services/orders.py   # existing business logic — no LangChain
myapp/ai/client.py         # the facade
myapp/ai/models.py         # model factory
myapp/ai/graphs/           # workflows
myapp/ai/tools/            # adapters over services
```

Two checks the lesson prints:

- `triage_ticket` returns a `TriageResult`.
- `lookup_order` answers a question by calling `get_order` in the service layer, not by reading a raw dict inside the agent.

Confirm the boundary:

```bash
rg langchain myapp --glob '*.py'
```

Hits belong under `myapp/ai/` only.

**Production use.** Every feature from Parts 1–7, once something outside the learning scripts needs to call it.

**Pitfalls.**

- Importing `AIMessage` from a route or a service.
- A tool that opens its own database connection and skips `myapp.services`.
- A facade that returns graph state "just this once." Callers will start depending on it.

**Exercise.** `steps/step8_exercise_service_adapter.py`. This file must not import LangChain.

1. `myapp/services/notes.py`: `add_note(order_id, body)`. Unknown orders raise `OrderNotFound` from the orders service. Store notes in a module-level dict.
2. `myapp/ai/tools/notes.py`: `@tool add_order_note` catches `OrderNotFound` and returns an error message. Do not raise across the tool boundary.
3. `myapp.ai.client.add_note_via_agent(question) -> str` builds a `create_agent` with that one tool.
4. Call the facade from the exercise for order `A123` and for the fake id `Z999`.
5. Re-run `rg "langchain" myapp --glob "*.py"`. `services/` and `domain.py` stay clean.

---

### Step 9 — Production checklist

**Concept.** The pieces from Steps 1–8, named as habits a real app needs on day one: a fake model in CI, a `thread_id` taken from a primary key, and a recursion cap on every invoke.

**Why it exists.** The lessons call a real model so you can see behavior. A pull request cannot do that. A scripted model runs the real `create_agent` loop for free and the same way every time.

Run:

```bash
uv run python steps/step9_production_checklist.py
uv run pytest tests/test_fake_tool_loop.py -q
```

The fake yields two messages you wrote: a tool call, then the final sentence. `create_agent` still executes `get_order_status` and appends a `ToolMessage`. The assertion is on message types, not on prose.

```python
fake = ScriptedChatModel(messages=iter([
    AIMessage(content="", tool_calls=[{
        "name": "get_order_status",
        "args": {"order_id": "A123"},
        "id": "call_fake_1",
        "type": "tool_call",
    }]),
    AIMessage(content="Order A123 has shipped."),
]))
```

Part 2 is one line of identity: ticket `4812` becomes `thread_id` `"ticket-4812"`. Store that mapping in your tables. Checkpoints are conversation data.

Part 3 prints the checklist. It is the same list as Part 5 of this guide, tied back to the steps:

| Habit | Where you practiced it |
| --- | --- |
| Facade, domain types only | Step 8 |
| Pick the lowest layer that works | Steps 1, 2, 4, 5, 7 |
| Tools as service adapters; errors as messages | Steps 3 and 8 |
| Checkpointer, `thread_id` = domain id | Step 6 |
| Interrupt before a consequential write | Step 6 |
| `timeout`, `max_retries`, `recursion_limit` | Steps 1, 4, 6 |
| Fake-model loop in CI; routers tested with dicts | Steps 5 and 9 |
| Caps in code, model output treated as untrusted | Step 4 |

**Exercise.** `steps/step9_exercise_ci_tests.py`. No API key.

1. `uv run pytest tests/test_fake_tool_loop.py -q` already passes. Read it.
2. Add `tests/test_route_ticket.py`. Import `route_ticket` from `myapp.ai.graphs.triage`. A hand-built state with category `"other"` routes to `"other"`.
3. Add `tests/test_recursion_limit.py`. `ScriptedChatModel` always emits the same `get_order_status` call. `create_agent` with `recursion_limit=4` raises `langgraph.errors.GraphRecursionError`.

```bash
uv run pytest tests -q
```

---

### How the nine steps stack

| You are here | You can do this next |
| --- | --- |
| 1–2 | Call a model and branch on a typed object. |
| 3–4 | Let the model request tools, with a cap and with rules in code. |
| 5 | Take the path choice back when the process is known. |
| 6 | Remember a thread, and pause a write for a person. |
| 7 | Add files and subagents only after a plain agent loses the plan. |
| 8–9 | Hide the stack behind a facade, and test the loop without paying for a model. |
