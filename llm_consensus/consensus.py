"""Core multi-model consensus engine."""
from __future__ import annotations
import concurrent.futures
import textwrap
from dataclasses import dataclass, field
from typing import Optional
from .models import ModelResponse

CROSS_FEED_SYSTEM = """You are participating in a multi-model consensus process.
Given the original question and other models\' responses, reply with:
  AGREE: <exact answer you agree with>
or if you disagree:
  REVISE: <revised answer>
  REASON: <one sentence why>
Be concise. No pleasantries."""

@dataclass
class ConsensusResult:
    answer: str
    rounds: int
    total_calls: int
    history: list = field(default_factory=list)
    converged: bool = True

class ConsensusEngine:
    def __init__(self, models, max_rounds=5, judge_model=None, verbose=False):
        self.models = models
        self.max_rounds = max_rounds
        self.judge_model = judge_model
        self.verbose = verbose
        self._total_calls = 0

    def run(self, prompt: str, system: Optional[str] = None) -> ConsensusResult:
        history = []
        messages = self._build_initial(prompt, system)
        responses = self._ask_all(messages, 0)
        history.append(responses)
        self._log(0, responses)

        for round_idx in range(1, self.max_rounds + 1):
            if self._is_unanimous(responses):
                return ConsensusResult(self._extract(responses), round_idx-1, self._total_calls, history, True)
            messages = self._build_crossfeed(prompt, responses, system)
            responses = self._ask_all(messages, round_idx)
            history.append(responses)
            self._log(round_idx, responses)

        return ConsensusResult(self._majority(responses), self.max_rounds, self._total_calls, history, self._is_unanimous(responses))

    def _ask_all(self, messages, round_num):
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.models)) as ex:
            futures = {ex.submit(m.ask, messages, round_num): m for m in self.models}
            results = []
            for f in concurrent.futures.as_completed(futures):
                try:
                    results.append(f.result()); self._total_calls += 1
                except Exception as e:
                    print(f"[WARN] {futures[f].model_id} failed: {e}")
        return results

    def _build_initial(self, prompt, system):
        msgs = []
        if system: msgs.append({"role":"system","content":system})
        msgs.append({"role":"user","content":prompt})
        return msgs

    def _build_crossfeed(self, prompt, responses, system):
        others = "\n\n".join(f"--- {r.model_id} ---\n{r.content}" for r in responses)
        return [
            {"role":"system","content":CROSS_FEED_SYSTEM},
            {"role":"user","content":f"ORIGINAL:\n{prompt}\n\nOTHER RESPONSES:\n{others}"},
        ]

    def _is_unanimous(self, responses):
        if not responses: return False
        if self.judge_model: return self._judge(responses)
        return len(set(self._norm(r.content) for r in responses)) == 1

    def _judge(self, responses):
        combined = "\n\n".join(f"{r.model_id}:\n{r.content}" for r in responses)
        msgs = [{"role":"system","content":"Reply UNANIMOUS or DIVERGENT. One word only."},
                {"role":"user","content":combined}]
        try:
            r = self.judge_model.ask(msgs); self._total_calls += 1
            return r.content.upper().startswith("UNANIMOUS")
        except: return False

    def _norm(self, text):
        t = text.strip().lower()
        for p in ("agree:","revise:","reason:"):
            if t.startswith(p): t = t[len(p):].strip()
        return t

    def _extract(self, responses):
        raw = responses[0].content
        if raw.upper().startswith("AGREE:"): return raw[6:].strip()
        if raw.upper().startswith("REVISE:"):
            lines = [l for l in raw.split("\n") if not l.upper().startswith("REASON:")]
            return "\n".join(lines).replace("REVISE:","").strip()
        return raw

    def _majority(self, responses):
        from collections import Counter
        return Counter(self._norm(r.content) for r in responses).most_common(1)[0][0]

    def _log(self, round_num, responses):
        if not self.verbose: return
        print(f"\n{'='*60}\n  ROUND {round_num}  ({len(responses)} responses)\n{'='*60}")
        for r in responses:
            print(f"  [{r.model_id}] {r.content[:100].replace(chr(10),' ')}...")
