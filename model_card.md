# PawPal+ Model Card

A responsible-AI reflection on PawPal+, an AI pet care planner. For what the
system does and how it's built, see the [README](README.md).

## System at a glance

| | |
|---|---|
| **AI components** | A planning agent using Gemini (`gemini-3.8-flash` by default, falling back to lighter "flash-lite" models when it's overloaded); a scikit-learn task classifier trained on 219 labelled tasks |
| **Intended use** | Helping a pet owner turn a description of their day into a schedule of routine care tasks that fits their available time |
| **Not intended for** | Diagnosis, medication dosing, emergencies, or anything that should go to a vet |
| **Who decides** | Gemini proposes tasks, but the original deterministic scheduler places them. The classifier only flags disagreements. The owner can edit everything. |

---

## Limitations and biases

**The agent relies on Gemini's general knowledge.** No reference material
backs its choices of duration and priority. It can be generic, or simply
wrong for a particular breed, age or health condition. In the live
evaluation, it rated a 2-year-old husky's only walk as "medium" priority,
so the walk was the task skipped. The trained model flagged it, but only as
advice the owner has to act on.

**The training data reflects my own judgement, not ground truth.** The 219
labelled tasks were written with AI help and reviewed by me. Wherever
priority is a matter of opinion, the model learns my opinion. For example,
the data labels a morning walk "high" and an evening walk "medium". The
model can't tell that's a convention rather than a fact.

**The classifier is biased toward the common answers.**
- Half the training labels are "high" priority, and it finds only 29% of
  the genuinely "low" tasks.
- On duration, it never correctly predicted a 45, 60 or 90 minute task in
  testing. So it under-estimates long tasks, and those are exactly the long
  exercise sessions that high-energy dogs need.
- It sees only three species, dog, cat and "other", so a rabbit, a snake and
  a goldfish all look the same to it.

**Species coverage is uneven.** The dataset has 91 dog tasks, 68 cat tasks
and 60 for every other animal combined. Owners of reptiles, birds or fish
get less reliable second opinions.

**The safety checks are keyword lists, and English only.**
- The emergency banner fires on words like "poison", "seizure" or "collapsed".
  While writing this card, I found that "my cat has been vomiting blood" and
  "my dog is limping and won't eat" don't trigger it.
- The prompt-injection screen catches common phrasings like "ignore previous
  instructions", but not a reworded attempt like "pretend the rules don't
  apply".
- None of this is tuned for languages other than English.

**The checks judge structure, not wisdom.** The post-check confirms the plan
fits the time and has no clashes. It can't confirm that 15-minute walks are
enough for three dogs that each "need an hour", which is a choice the agent
made in the live evaluation. That judgement is left to the owner and the
human rating.

**It depends on an outside service that isn't always available.** Planning
needs Google's servers. During testing, every full Gemini Flash model was
overloaded and returning errors. PawPal now falls back to lighter
"flash-lite" models, so it keeps working, but:
- The same request can be answered by a different, less capable model
  depending on Google's load that day, so plan quality isn't consistent
  from run to run.
- The status box shows which model answered, but nothing judges whether the
  lighter model's plan is as good.
- With no key, or with every model down, Plan with AI is off. The
  scheduler, quick add and task editor still work.

**A plan takes many round trips, so it can be slow.** A plan is 4 to 15
separate Gemini calls. In normal conditions a step takes a few seconds; when
Google is busy, one slow call holds up the whole plan.

---

## Misuse and how it's prevented

| Possible misuse | What PawPal+ does about it | What's still open |
|---|---|---|
| **Treating it as a vet**, e.g. relying on it in an emergency or for medication | Emergency wording shows a "call your vet now" banner. The agent is told not to invent medical advice and to use the owner's own times for medication, and every task stays editable. | The emergency keywords miss some phrasings (see above). The no-medical-advice rule is in the prompt, so nothing in code enforces it. |
| **Prompt injection**, to make the model ignore its rules or leak its prompt | Input is screened for injection patterns before any model call. The agent can only act through 7 validated tools, which reject bad values, so a successful injection can still only add or edit pet tasks in that user's own session. | The pattern list is deliberately narrow, so creative rewordings get through. The tool design is the real protection. |
| **Running up the API bill**, by scripting the button or pasting huge inputs | 10 AI calls per session per 5 minutes; a 2,000-character cap on plan requests; at most 15 agent steps per plan. | The rate limit is per browser session, so someone opening many sessions could get around it. A public deployment would need a per-user or per-IP limit. |
| **Leaking the API key** | The key is read from the environment or a git-ignored `.env` file, never from code. Tests never use a real key. | — |
| **Exposing what users typed** | Logs store only a short preview of user text, the `logs/` folder is git-ignored, and nothing is stored between sessions. | Previews can still mention a pet's health condition, so on a shared machine the log file should be treated as private. |

---

## What surprised me while testing reliability

- **The second model earned its place.** I added the trained classifier as
  a backup check, expecting it to matter rarely. In the first live run after
  removing the care guide, it caught the most important mistake: a husky's
  walk rated too low and then skipped. A different kind of model doesn't
  share the LLM's blind spots.
- **Confidence scores turned out to be meaningful, but not reassuring.** When
  the classifier was at least 70% sure of a priority, it was right 72% of
  the time, versus 53% below that. That justified the 70% bar for flagging.
  It also showed that a "confident" prediction is still wrong about 1 time
  in 4, which is why the model only flags and never overrides.
- **More tuning didn't help the classifier.** Other settings, an SVM, and
  different feature sets all moved accuracy by only 1 to 2 points. The limit
  is the labels, not the algorithm: priority is partly a judgement call.
- **The bugs were in the plain code, not the AI.** Tests found a stray space
  left by the text cleaner, and the quick-add parser turning "half an hour
  of play" into a task called "Of play".
- **The gaps showed up as soon as I tried new phrasings.** "Vomiting blood"
  doesn't trigger the emergency banner. It wasn't in my original test cases;
  I found it by deliberately asking things the system shouldn't handle.
- **The hardest reliability problem was availability, not accuracy.** I
  expected the model's mistakes to be the main risk. Instead, the first
  live runs failed before the model said anything:
  - "503 high demand" errors from Google, then plans that ran for minutes.
  - My first fix, a time limit, turned endless waiting into "504 deadline
    exceeded" errors, which fail faster but still fail.
  - Testing each model directly showed every full Flash model was
    overloaded while the "lite" models answered in under a second. So I
    added an automatic fallback. The next real run fell back after one
    timeout and finished the plan in about 2 seconds.

  The lesson was that "does it work?" includes "is the service there?", and
  that the logs, which time every step, are what made the cause visible.

---

## Collaboration with AI

I built this with an AI coding agent (Claude Code), alongside my own design
and testing. I designed the original classes and scheduler myself from my
UML. Then I used the agent for multi-step work:
- the agent loop and the Gemini adapter
- tests with a scripted fake model
- the Streamlit integration
- documentation drafts

The most useful prompts were specific and limited to one file, like "add
conflict detection to Scheduler in `scheduler.py`". Asking "what edge cases
could break this?" was also useful. I checked every change by running the
test suite, running `main.py`, and using the app myself. The interaction log
is in [`ai_interactions.md`](ai_interactions.md).

### A suggestion I accepted

**Keep the original scheduler as the authority, and let Gemini act only
through validated tools.** When I asked for an AI planning feature, the
agent proposed that Gemini shouldn't write the schedule itself. Instead, it
would call tools like `add_task` and `build_plan`, and my existing,
already-tested `Scheduler` would decide the times. I accepted this for three
reasons:
- It kept 42 passing scheduler tests meaningful instead of throwing them
  away.
- It made the AI's mistakes containable, because a tool can reject a bad
  value, but a free-text schedule can't be checked as easily.
- It made the agent testable with a fake model.

The cost was more round trips to the API per plan. I wrote that trade-off
down in the README rather than hiding it.

### A suggestion I did not accept as written

**Retrieval (RAG) alongside the agent.** The AI's original proposal was an
agent *plus* RAG. It wrote a care guide of about 40 passages, a TF-IDF
search over it, a separate "Ask the care guide" Q&A feature with citations,
and checks that every number in an answer appeared in its source. It worked
and was well tested. But it doubled what a reader had to understand: two AI
features, two kinds of checks, and a knowledge base to explain. Most of it
had nothing to do with scheduling, which is what the project is about. I
decided it was over-engineered for this system and removed all of it. I
kept the agent, the scheduler and the checks that protect plans. The cost
is real: the agent now picks durations from general knowledge. The live
evaluation showed it under-rating a husky's walk, and the trained model
catching it.

A smaller example of the same kind: the early design gave `Owner` a list of
"preferred task types" that the scheduler never used. I removed it rather
than keep a field that suggested a feature that didn't exist.

### What I'd tell someone starting out

Design the boundaries yourself, then hand the AI small, well-defined jobs
inside them. The AI was fast and usually right. It was also confident when
it was wrong, and willing to add things that looked impressive but that the
project didn't need.
