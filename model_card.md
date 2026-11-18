# PawPal+ Model Card

A responsible-AI reflection on PawPal+, an AI pet care planner. For what the
system does and how it's built, see the [README](README.md).

## System at a glance

| | |
|---|---|
| **AI components** | A planning agent and a care-guide Q&A, both using Gemini (`gemini-3.8-flash` by default); TF-IDF retrieval over a hand-written care guide; a scikit-learn task classifier trained on 219 labelled tasks |
| **Intended use** | Helping a pet owner turn a description of their day into a schedule of routine care tasks, and answering common care questions from a small, cited guide |
| **Not intended for** | Diagnosis, medication dosing, emergencies, or anything that should go to a vet |
| **Who decides** | Gemini proposes tasks, but the original deterministic scheduler places them. The classifier only flags disagreements. The owner can edit everything. |

---

## Limitations and biases

**The care guide is small and narrow.** It's about 40 passages across 10
Markdown files that I wrote. It covers common care for dogs, cats and a few
small animals. Anything outside it gets "the guide doesn't cover that", which
is safe but often unhelpful. The advice is general. It doesn't know breeds
beyond a few examples, regional differences, or an individual pet's medical
history.

**The training data reflects my own judgement, not ground truth.** The 219
labelled tasks were written with AI help, following my care guide. Wherever
priority is a matter of opinion, the model learns my opinion. For example,
the data labels a morning walk "high" and an evening walk "medium". The
model can't tell that's a convention rather than a fact.

**The classifier is biased toward the common answers.** Half the training
labels are "high" priority. It finds only 29% of the genuinely "low" tasks.
On duration, it never correctly predicted a 45, 60 or 90 minute task in
testing. So it under-estimates long tasks, and those are exactly the long
exercise sessions that high-energy dogs need. It also sees only three
species, dog, cat and "other", so a rabbit, a snake and a goldfish all look
the same to it.

**Species coverage is uneven.** The dataset has 91 dog tasks, 68 cat tasks
and 60 for every other animal combined. The guide follows the same pattern.
Owners of reptiles, birds or fish get thinner advice.

**Retrieval matches words, not meaning.** A question worded differently from
the guide can miss the best passage, or rank it below a weaker one. The
husky exercise question ranks the right passage third.

**The safety checks are keyword lists, and English only.**
- The emergency banner fires on words like "poison", "seizure" or "collapsed".
  While writing this card, I found that "my cat has been vomiting blood" and
  "my dog is limping and won't eat" don't trigger it.
- The prompt-injection screen catches common phrasings like "ignore previous
  instructions", but not a reworded attempt like "pretend the rules don't
  apply".
- None of this is tuned for languages other than English.

**Gemini can still be wrong in ways the checks can't see.** The checks
confirm that an answer cites a passage and that its numbers appear in that
passage. They can't confirm that a sentence without numbers is a fair
summary of what it cites.

---

## Misuse and how it's prevented

| Possible misuse | What PawPal+ does about it | What's still open |
|---|---|---|
| **Treating it as a vet**, e.g. asking for drug doses or relying on it in an emergency | The guide has no dosing information, and the Q&A must answer from the guide or say it isn't covered. Health answers always get a "follow your vet's advice" note, added in code rather than requested in the prompt. Emergency wording shows a "call your vet now" banner. | The emergency keywords miss some phrasings (see above). Dosing questions are only kept out because the model says "not covered"; retrieval doesn't stop them. |
| **Prompt injection**, to make the model ignore its rules or leak its prompt | Input is screened for injection patterns before any model call. The agent can only act through 8 validated tools, which reject bad values, so a successful injection can still only add or edit pet tasks in that user's own session. | The pattern list is deliberately narrow, so creative rewordings get through. The tool design is the real protection. |
| **Running up the API bill**, by scripting the buttons or pasting huge inputs | 10 AI calls per session per 5 minutes; length caps on every input (2,000 characters for a plan request, 300 for a question); at most 15 agent steps per plan. | The rate limit is per browser session, so someone opening many sessions could get around it. A public deployment would need a per-user or per-IP limit. |
| **Leaking the API key** | The key is read from the environment or a git-ignored `.env` file, never from code. Tests never use a real key. | — |
| **Exposing what users typed** | Logs store only a short preview of user text, the `logs/` folder is git-ignored, and nothing is stored between sessions. | Previews can still mention a pet's health condition, so on a shared machine the log file should be treated as private. |

---

## What surprised me while testing reliability

- **A score threshold wasn't enough to keep retrieval on topic.** "Can my dog
  drive a car?" retrieved passages that scored as high as real questions,
  just from the word "dog". I added a second check: a passage must share a
  specific topic word with the question, not just a pet word.
- **Confidence scores turned out to be meaningful, but not reassuring.** When
  the classifier was at least 70% sure of a priority, it was right 72% of the
  time, versus 53% below that. That justified the 70% bar for flagging. It
  also showed that a "confident" prediction is still wrong about 1 time in 4,
  which is why the model only flags and never overrides.
- **More tuning didn't help the classifier.** Other settings, an SVM, and
  different feature sets all moved accuracy by only 1 to 2 points. The limit
  is the labels, not the algorithm: priority is partly a judgement call.
- **The bugs were in the plain code, not the AI.** Tests found a stray space
  left by the text cleaner, and the quick-add parser turning "half an hour of
  play" into a task called "Of play".
- **The gaps showed up as soon as I tried new phrasings.** A veterinary
  dosing question ("What dose of ibuprofen can I give my dog?") gets past the
  retrieval check, so only the model's "not covered" reply stops it.
  "Vomiting blood" doesn't trigger the emergency banner. Neither was in my
  original test cases. Both came from deliberately asking things the system
  shouldn't handle.
- **I had to resist changing a test to make it pass.** A test that required
  the best husky passage to rank first failed. The honest fix was to test
  for the top 3, which is what the model actually sees, and record the
  limitation, not to tune the retrieval until that one example passed.

---

## Collaboration with AI

I built this with an AI coding agent (Claude Code), alongside my own design
and testing. I designed the original classes and scheduler myself from my
UML. Then I used the agent for multi-step work:
- writing the care guide and the retrieval module
- the agent loop and the Gemini adapter
- tests with a scripted fake model
- the Streamlit integration
- documentation drafts

The most useful prompts were specific and limited to one file, like "add
conflict detection to Scheduler in `pawpal_system.py`". Asking "what edge
cases could break this?" was also useful. I checked every change by running
the test suite, running `main.py`, and using the app myself. The
interaction log is in [`ai_interactions.md`](ai_interactions.md).

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

**The "preferred task types" field on `Owner`.** The early AI-assisted
design gave `Owner` a list of preferred task types, which looked reasonable
for a planner. But the scheduler never used it. Keeping it would have
suggested a feature that didn't exist, both to users and to anyone reading
the code. I removed it rather than build a preference system I didn't need.
For the same reason, I cut AI-drafted documentation that was longer than it
needed to be.

### What I'd tell someone starting out

Design the boundaries yourself, then hand the AI small, well-defined jobs
inside them. The AI was fast and usually right. It was also confident when
it was wrong, and willing to add things that looked useful but weren't
needed.
