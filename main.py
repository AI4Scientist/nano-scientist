#!/usr/bin/env python3
"""Autonomous Scientist — budget-aware research agent.

Usage:
    python main.py "Your research topic" --budget 0.05 --env .env.prod
    python main.py topic.md --budget 1.00
    python main.py --list-skills
"""

import argparse
import os
import shutil
import sys
import tempfile

from pathlib import Path
from src.utils import init_env, load_skill_index, filter_skill_index, detect_api_keys
from src.flow import create_scientist_flow

# Resolve project paths
SKILLS_DIR = "skills"


class _Tee:
    """Write to both the original stdout and a file once the path is known."""

    def __init__(self, original):
        self._orig = original
        self._file = None

    def set_path(self, path: Path):
        if self._file is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("w", encoding="utf-8", buffering=1)

    def write(self, data):
        self._orig.write(data)
        if self._file:
            self._file.write(data)

    def flush(self):
        self._orig.flush()
        if self._file:
            self._file.flush()

    def fileno(self):
        return self._orig.fileno()

    def close(self):
        if self._file:
            self._file.close()

def list_skills():
    """Print available skills (from skills.json index)."""
    index = load_skill_index(str(SKILLS_DIR))
    print(f"\nAvailable Skills ({len(index)}):")
    print("-" * 50)
    for name, desc in sorted(index.items()):
        short = desc[:80] + "..." if len(desc) > 80 else desc
        print(f"  {name}: {short}")
    print()


def run(topic: str, budget: float, output_dir: str):
    """Run the autonomous scientist flow."""
    # Load lightweight skill index (descriptions only, not full SKILL.md)
    skill_index = load_skill_index(str(SKILLS_DIR))

    # Detect available API keys so the scientist knows its capabilities
    api_keys = detect_api_keys()
    available = [k for k, v in api_keys.items() if v]
    missing = [k for k, v in api_keys.items() if not v]
    print(f"API keys available: {', '.join(available) if available else 'NONE'}")
    if missing:
        print(f"API keys missing: {', '.join(missing)}")

    # Filter out skills whose required API keys are not available
    skill_index = filter_skill_index(str(SKILLS_DIR), skill_index, api_keys)
    print(f"Loaded {len(skill_index)} skills (after filtering unavailable)")

    # Initialize shared store
    shared = {
        "topic": topic,
        "budget_dollars": budget,
        "budget_remaining": budget,
        "cost_log": [],
        "skill_index": skill_index,       # lightweight: {name: description}
        "skills_dir": str(SKILLS_DIR),    # path for lazy-loading SKILL.md
        "output_dir": output_dir,
        "api_keys": api_keys,             # {key_name: is_set} for capability awareness
    }

    # Create and run flow
    flow = create_scientist_flow()

    print(f"\n{'='*50}")
    print(f"Autonomous Scientist")
    print(f"Topic: {topic}")
    print(f"Budget: ${budget:.2f}")
    print(f"{'='*50}\n")

    tmp_fd, tmp_traj_str = tempfile.mkstemp(suffix=".traj.txt")
    os.close(tmp_fd)
    tmp_traj = Path(tmp_traj_str)

    tee = _Tee(sys.stdout)
    tee.set_path(tmp_traj)
    sys.stdout = tee  # type: ignore[assignment]
    try:
        flow.run(shared)
    finally:
        sys.stdout = tee._orig
        tee.close()

    # Move temp traj file into the run's output directory
    if shared.get("output_path") and tmp_traj.exists():
        traj_dest = Path(shared["output_path"]) / "traj.txt"
        shutil.move(str(tmp_traj), traj_dest)
    elif tmp_traj.exists():
        tmp_traj.unlink()

    return shared


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Scientist — budget-aware research agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "What are CRISPR off-target effects?" --budget 1.00
  python main.py "Latest in quantum error correction" --budget 0.05
  python main.py topic.md --budget 1.00
  python main.py --list-skills
        """,
    )
    parser.add_argument(
        "topic", nargs="?",
        help="Research topic string OR path to a .md file",
    )
    parser.add_argument(
        "--budget", "-b", type=float, default=5.0,
        help="Budget in USD for LLM inference (default: $5.00)",
    )
    parser.add_argument(
        "--output", "-o", default="outputs",
        help="Output directory (default: outputs/)",
    )
    parser.add_argument(
        "--list-skills", action="store_true", help="List available skills",
    )
    parser.add_argument(
        "--env", "-e", default=".env",
        help="Path to .env file for API keys (default: .env)",
    )

    args = parser.parse_args()

    # Load environment variables before anything else
    init_env(args.env)

    if args.list_skills:
        list_skills()
        return

    if not args.topic:
        parser.print_help()
        sys.exit(1)

    try:
        # Detect file mode: if topic argument is a path to an existing file,
        # read its content as the research topic
        topic = args.topic
        topic_path = Path(args.topic)
        if topic_path.exists() and topic_path.is_file():
            topic = topic_path.read_text(encoding="utf-8").strip()
            print(f"Loaded research topic from {args.topic}")

        shared = run(topic, args.budget, args.output)
        # Exit 0 if we got output
        if shared.get("output_path"):
            sys.exit(0)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
