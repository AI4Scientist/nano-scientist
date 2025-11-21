"""
Minimal example - the simplest possible usage.
"""

from msr_scientist import create_research_agent

# One-liner to create agent (make sure HF_TOKEN is set in environment)
agent = create_research_agent()

# Just give it a research task and let it evolve
agent.research("""
Investigate whether Python list comprehensions are faster than traditional loops.
Run experiments, collect data, and draft a short paper with your findings.
""")

print("✅ Done! Check outputs/ directory for results.")
