"""Unit tests using mock models — no API calls needed."""
import pytest
from llm_consensus.consensus import ConsensusEngine
from llm_consensus.models import ModelResponse

class MockModel:
    def __init__(self, model_id, answer): self.model_id=model_id; self.provider="mock"; self._answer=answer
    def ask(self, messages, round_num=0): return ModelResponse(self.model_id,"mock",self._answer,round_num)

def make_engine(answers, **kw):
    return ConsensusEngine([MockModel(f"m{i}",a) for i,a in enumerate(answers)], **kw)

def test_unanimous_round0():
    r = make_engine(["42","42","42"]).run("question")
    assert r.converged and r.rounds == 0 and "42" in r.answer

def test_max_rounds_no_consensus():
    r = make_engine(["red","blue","green"], max_rounds=2).run("colour")
    assert not r.converged and r.answer in ("red","blue","green")

def test_call_count():
    r = make_engine(["yes","yes","yes"]).run("agree?")
    assert r.total_calls == 3
