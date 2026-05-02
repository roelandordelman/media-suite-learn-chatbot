# Agentic RAG — Background and Relevance to Ask Media Suite

This document summarises the concept of Agentic RAG, its current state of the
art, and how it relates to the Ask Media Suite chatbot. It is intended as
background reading for contributors and as a planning reference for future
development. It does not describe anything currently implemented — see
[ROADMAP.md](../ROADMAP.md) for the planned adoption path.

---

## What standard RAG cannot do

The current Ask Media Suite pipeline is a fixed sequence:

```
question → expand → embed → retrieve → generate → answer
```

It runs once, in one direction, with no ability to evaluate whether it worked.
The model has no agency over the process — it takes whatever chunks and graph
facts come back from retrieval and does its best with them. For simple,
well-formed questions this works well. For complex, multi-part, or ambiguous
questions it has a structural weakness: a single retrieval pass is unlikely to
cover everything the question needs, and there is no mechanism to detect or
recover from a poor retrieval.

---

## What Agentic RAG is

Agentic RAG replaces the fixed pipeline with a **reasoning loop**. The LLM
becomes an agent that actively decides:

- whether to retrieve at all, or answer from what it already knows
- what query to use for retrieval
- whether the retrieved results are good enough to answer the question
- whether to try a different retrieval strategy
- whether the question needs multiple retrievals to answer fully
- when it has enough information to stop and generate an answer

```
question
   │
   ▼
┌─────────────────────────────────────┐
│  Agent reasons about the question   │
│                                     │
│  "Do I need to retrieve?"    Yes ───┼──► retrieve(query_1)
│         │                           │         │
│        No                           │    good enough?
│         │                           │    No ──┼──► retrieve(query_2)
│         ▼                           │         │
│  answer from context                │    good enough?
│                                     │    Yes ─┼──► generate answer
└─────────────────────────────────────┘
```

The agent decides the path through the system rather than the system dictating
the path to the agent. The term "agentic" comes from AI agent theory — a system
that perceives its environment, reasons about it, and takes actions to achieve
a goal. In agentic RAG the environment is the knowledge base and conversation,
the actions are retrieval calls and generation steps, and the goal is a correct,
grounded answer.

---

## A concrete example

**Standard RAG handling a complex question:**

> "I'm researching post-war Dutch radio broadcasts and I want to annotate
> specific fragments with my own tags. What do I need and what are the
> limitations?"

Standard RAG embeds this whole question, retrieves the top-k chunks, and tries
to answer from whatever comes back. The question spans at least three topics:
collection access, annotation tools, and licensing restrictions. The retrieved
chunks are unlikely to cover all three well in a single pass.

**Agentic RAG handling the same question:**

```
Agent plan:
  Step 1 — What collections cover post-war Dutch radio?
    → retrieve("Dutch radio collections post-war")
    → evaluate: good result, found Sound & Vision radio collection

  Step 2 — What are the access requirements for this collection?
    → retrieve("Sound and Vision radio collection access login")
    → evaluate: good result, found access requirements

  Step 3 — What annotation tools are available?
    → retrieve("annotation tools Media Suite fragment level")
    → evaluate: partial result, found tools but not fragment-level detail

  Step 4 — Are there licensing restrictions on annotation?
    → retrieve("annotation restrictions licensing Sound and Vision")
    → evaluate: weak result, low confidence
    → flag as uncertain in final answer

  Synthesise: combine steps 1–4 into a structured answer,
  noting that step 4 was uncertain and suggesting the researcher
  check the documentation directly for licensing details.
```

The answer is more complete, more honest about uncertainty, and grounded in
multiple targeted retrievals rather than one broad one.

---

## Current state of the art

Several patterns are in active use in production systems. They are listed here
roughly in order of complexity and capability.

### CRAG — Corrective RAG

The simplest form of agentic behaviour. After retrieval, a lightweight evaluator
scores chunk relevance. Below a threshold: discard and try a reformulated query.
Above threshold: proceed to generation. This adds a single quality gate to the
standard pipeline without a full reasoning loop.

This is the most practical first step for Ask Media Suite — it addresses the
"low confidence answer" problem without a major architectural change.

### ReAct — Reason + Act

The most widely used pattern for full agentic RAG. The agent alternates between
explicit reasoning steps ("I need to find out about access requirements for this
collection") and action steps ("retrieve: Sound and Vision access login"). Each
reasoning step is visible in the output, which makes the system explainable —
contributors and researchers can see why each retrieval decision was made.

Developed by Google in 2022, now implemented in most agent frameworks including
LangChain and LlamaIndex, and replicable without either framework using a
reasoning prompt and a tool-calling loop.

### Plan-and-Execute

The agent first generates a complete plan — a list of sub-questions to answer —
then executes each step in sequence. Better than ReAct for complex multi-part
questions because the full scope of the answer is defined upfront. The weakness
is that if the plan is wrong, everything downstream is wrong. For research
assistance use cases where questions are genuinely complex, this pattern is
worth evaluating.

### Self-RAG

The model is trained to generate special tokens that signal when to retrieve,
how to evaluate retrieved content, and how confident it is in its answer. More
powerful than prompt-based approaches but requires a specially fine-tuned model.
Not applicable to Ask Media Suite's current stack (general-purpose llama3.1:8b)
without significant additional work.

---

## Three properties that define agentic behaviour

**Tool use** — retrieval is a tool the LLM calls, not a step the pipeline forces
it through. The LLM decides when to call it, with what query, and how many times.
It can also call other tools — a SPARQL endpoint, a web search, a citation lookup
— when vector search is not the right approach. In the Ask Media Suite context,
both ChromaDB and Fuseki become tools the agent can choose between rather than
both being called unconditionally for every question.

**Self-evaluation** — the agent assesses its own outputs. After retrieving, it
asks: "does this actually answer the question?" After generating, it asks: "is
this grounded in what I retrieved, or am I filling in gaps from general
knowledge?" If the answer to either is no, it tries again rather than returning
a poor answer silently.

**Planning** — for complex questions, the agent decomposes the question into
sub-questions, retrieves for each, and synthesises the results. This is the
capability that most clearly separates agentic RAG from standard RAG for
humanities research use cases, where questions are often genuinely multi-part.

---

## Tradeoffs

Agentic RAG is not strictly better than standard RAG — it trades simplicity and
speed for capability and quality on complex questions.

| | Standard RAG | CRAG | ReAct / Agentic |
|---|---|---|---|
| Latency | Fast | Slightly slower | Noticeably slower |
| Complexity | Simple pipeline | Minimal addition | Reasoning loop, harder to debug |
| Simple questions | Good | Good | Good but overkill |
| Complex questions | Poor | Better | Much better |
| LLM calls per question | 1–2 | 1–3 | 3–8+ |
| Predictability | High | High | Lower |
| Explainability | Low | Low | High (ReAct shows reasoning) |
| Local model feasibility | Yes | Yes | Needs testing |

The latency row is the most important consideration for Ask Media Suite
specifically. Running llama3.1:8b locally, a standard RAG answer takes a few
seconds. A ReAct loop with 3–4 retrieval steps and reasoning between each could
take 20–40 seconds — noticeable and potentially frustrating in a chat interface.
This needs to be measured before committing to a full agentic architecture.

---

## Relation to conversational search

Agentic RAG and conversational RAG are complementary, not alternatives.
Conversation history tells the agent what the researcher is trying to accomplish
across multiple turns. The agent uses that understanding to plan how to retrieve
within each turn.

```
conversation history
        │
        ▼
agent understands research context and goal
        │
        ▼
plans retrieval strategy for this question
        │
   ┌────┴────┐
   ▼         ▼
retrieve   retrieve        ← multiple targeted calls
   │         │
   └────┬────┘
        ▼
evaluate completeness
        │
   ┌────┴────┐
   ▼         ▼
generate   ask follow-up
answer     question
```

This is essentially what a good reference librarian does — maintains context
about what you are researching, plans a search strategy, evaluates whether what
they found actually helps, and asks targeted clarifying questions only when
genuinely needed.

---

## Relation to the dual-path architecture

The current Ask Media Suite architecture already contains a primitive form of
agentic behaviour: both retrieval paths run for every question, and the LLM
synthesises from whatever both paths return. This is better than routing, because
routing adds a failure mode — a misclassified question never reaches the right
path.

Full agentic RAG would make this more sophisticated: instead of always running
both paths unconditionally, the agent decides which path to invoke and with what
query, based on reasoning about the question. ChromaDB and Fuseki become tools
in the agent's toolkit rather than fixed pipeline stages.

This is a natural evolution of the current architecture rather than a replacement
of it. The named SPARQL query catalogue, the query expansion logic, and the
parallel path design all remain valuable — they become the tools the agent uses,
not the structure the agent replaces.

---

## Relevance to the knowledge graph layer

The knowledge graph is particularly well suited to agentic retrieval. Vector
search is good at finding relevant text but poor at answering precise relational
questions ("which tools support annotation AND work with the Oral History
collection?"). SPARQL is good at exactly these questions but needs to know which
query template to use.

An agent can reason about which type of question it is facing and choose
accordingly — vector search for narrative, explanation-seeking questions; SPARQL
for structural, enumeration, or filter questions. This is more nuanced than the
current embedding-similarity routing and can handle questions that genuinely
need both.

---

## Planned adoption path for Ask Media Suite

The roadmap proposes a staged approach rather than jumping directly to full
agentic RAG:

**Stage 1 — CRAG (next)**
Add a relevance scoring step after retrieval. If the top-k chunks score below
a threshold, reformulate the query and try once more before generating. This
is a small addition to `api/rag.py` and gives meaningful improvement on
ambiguous or poorly phrased questions.

**Stage 2 — Hybrid routing**
Route simple, well-formed questions to the standard pipeline (fast) and
complex, multi-part questions to a ReAct loop (thorough). A lightweight
classifier — or simply a question complexity heuristic — does the routing.
This avoids paying the latency cost of agentic reasoning for questions that
don't need it.

**Stage 3 — Full ReAct agent**
Replace the fixed pipeline with a reasoning loop that treats ChromaDB,
Fuseki, and any future MCP-connected sources as tools. Implement once the
knowledge base is stable, the query catalogue is comprehensive, and latency
on local hardware is understood.

The dependency order matters: embedding-based SPARQL routing should be
completed before Stage 2, and the knowledge graph should be substantially
complete before Stage 3. Building agentic behaviour on top of an unstable
or sparse knowledge base produces more elaborate failures, not better answers.

---

## Further reading

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — the foundational ReAct paper
- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)
- [Corrective Retrieval Augmented Generation (CRAG)](https://arxiv.org/abs/2401.15884)
- [LangChain conversational RAG documentation](https://python.langchain.com/docs/tutorials/qa_chat_history/)
- [LlamaIndex agentic RAG patterns](https://docs.llamaindex.ai/en/stable/use_cases/agents/)
