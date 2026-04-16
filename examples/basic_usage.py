"""
Basic usage: ask a question and get a consensus answer from multiple LLMs.

Requires API keys:
  OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
"""

import os
from llm_consensus import ConsensusEngine, build_default_pool


def main():
    models = build_default_pool(
        openai_instances=2,
        anthropic_instances=2,
        gemini_instances=2,
    )
    print(f"Pool: {len(models)} model instances")
    for m in models:
        print(f"  {m.model_id}")

    engine = ConsensusEngine(models, max_rounds=4, verbose=True)

    result = engine.run(
        prompt="What is the time complexity of quicksort in the average case, and why?",
        system="You are a computer science expert. Answer in 2-3 sentences.",
    )

    print("\n" + "="*60)
    print(f"Consensus reached: {result.consensus}")
    print(f"Rounds taken:      {result.rounds}")
    print(f"\nFinal answer:\n{result.answer}")


if __name__ == "__main__":
    main()
