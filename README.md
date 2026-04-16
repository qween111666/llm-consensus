# llm-consensus

**Multi-model LLM consensus engine — run a prompt through multiple models simultaneously and converge on a unanimous answer.**

Implements a cross-feed arbitration loop: each model sees all other models' answers and must either agree or revise until unanimous consensus is reached (or max rounds exceeded).

---

## How it works

```
Round 0:  all models answer independently
Round 1+: each model sees all others' answers → must reply AGREE: <reason> or REVISE: <new answer>
Done:     all answers match (string normalisation or judge-model check)
```

Supports mixing providers and multiple instances per provider — e.g. 3x Claude + 2x GPT-4 + 2x Gemini (7 models total).

---

## Install

```bash
pip install -r requirements.txt
```

Set API keys in environment:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=AI...
```

---

## CLI

```bash
python -m llm_consensus "What is the time complexity of merge sort?"
python -m llm_consensus "Best approach for rate-limiting an async queue?" --instances 3 --rounds 5 --verbose
```

```
Round 0 — 6 models queried
  [claude-3-5-sonnet-instance-0]  O(n log n) in all cases ...
  [gpt-4o-instance-0]             O(n log n) time, O(n) space ...
  ...
Round 1 — checking consensus
  [claude-3-5-sonnet-instance-0]  AGREE: same conclusion
  ...
Consensus reached in 1 round(s).
Answer: O(n log n) in all cases, O(n) auxiliary space.
```

---

## Python API

```python
from llm_consensus import ConsensusEngine, build_default_pool

models = build_default_pool(
    openai_instances=2,
    anthropic_instances=2,
    gemini_instances=2,
)

engine = ConsensusEngine(models, max_rounds=5)
result = engine.run(
    prompt="What is the best data structure for a priority queue?",
    system="You are a computer science expert. Be concise.",
)

print(result.answer)          # final consensus answer
print(result.rounds)          # how many rounds it took
print(result.consensus)       # True if unanimous, False if max_rounds hit
```

---

## Run tests (mock models, no API keys needed)

```bash
pip install pytest
pytest tests/ -v
```

---

## License

MIT
