"""Minimal example"""
from mini_researcher_agent import ResearchAgent

# Set HF_TOKEN in .env first
agent = ResearchAgent()

agent("Compare bubble sort vs quicksort: implement, benchmark on arrays [100,1000,10000], draft paper with findings")
