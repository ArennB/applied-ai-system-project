# AI Interactions Log

> **Stretch features only.** Only fill in the sections that apply to stretch features you attempted. If you did not attempt a stretch feature, leave its section blank or delete it. This file is not required for the core project.

---

## Agent Workflow (SF7)

> Document your experience using an AI agent (e.g., Cursor Agent, Claude, Copilot) to make multi-step changes autonomously.

**What task did you give the agent?**

Add an applied-AI feature fully integrated into PawPal+'s main logic. After
discussing options (RAG, agentic workflow, fine-tuning, reliability testing), we
chose an agent that uses RAG as one of its tools, backed by the Gemini API, with
TF-IDF retrieval and a new "Plan with AI" section in the Streamlit app.

**What did the agent do?**

1. Read `pawpal_system.py` and `app.py` and proposed a plan (agent + RAG, with the
   existing `Scheduler` kept as the source of truth).
2. Wrote the care-guide knowledge base in `knowledge/` (10 markdown files).
3. Wrote `pawpal_rag.py` (heading-based chunking and TF-IDF search).
4. Wrote `pawpal_agent.py` (validated tools, the plan-act-check loop, the
   independent post-check, and the Gemini adapter).
5. Wrote `tests/test_rag.py` and `tests/test_agent.py` with a scripted fake LLM,
   then ran the whole suite.
6. Added the "Plan with AI" section to `app.py` and drove the app headlessly
   with Streamlit's `AppTest` to check the no-key path and a full agent run.
7. Updated `README.md`, `requirements.txt` and `diagrams/uml_final.mmd`.

**What did you have to verify or fix manually?**

<!-- Describe anything the agent got wrong or that required human review -->

---

## Prompt Comparison (SF11)

> Compare two different prompts (or two different models) on the same task.

| | Option A | Option B |
|-|----------|----------|
| **Model / tool used** | | |
| **Prompt** | | |
| **Response summary** | | |
| **What was useful** | | |
| **Problems noticed** | | |
| **Decision** | | |

**Which approach did you use in your final implementation and why?**

<!-- Your conclusion -->
