"""CLI: python -m llm_consensus "Your question here" """
import argparse, sys
from .models import build_default_pool
from .consensus import ConsensusEngine

def main():
    p = argparse.ArgumentParser(description="Multi-model unanimous consensus engine.")
    p.add_argument("prompt", nargs="?", help="Question (use - for stdin)")
    p.add_argument("--system", default=None)
    p.add_argument("--instances", type=int, default=2, metavar="N")
    p.add_argument("--rounds", type=int, default=5, metavar="N")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    prompt = sys.stdin.read().strip() if (not args.prompt or args.prompt == "-") else args.prompt
    if not prompt: p.print_help(); sys.exit(1)

    print(f"[consensus] {args.instances} instances x 3 providers = {args.instances*3} models")
    pool = build_default_pool(args.instances, args.instances, args.instances)
    engine = ConsensusEngine(pool, max_rounds=args.rounds, judge_model=pool[0], verbose=args.verbose)
    print(f"[consensus] Running (max {args.rounds} rounds)...\n")
    result = engine.run(prompt, args.system)
    print(f"\n{'='*60}\n  RESULT — converged={result.converged} rounds={result.rounds} calls={result.total_calls}\n{'='*60}\n{result.answer}\n")

if __name__ == "__main__": main()
