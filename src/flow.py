"""Flow wiring for the Autonomous Scientist agent.

Pipeline:
  Initializer → PlanInitialExecutor → PlanDrivenExecutor (loop)
              → ReviewExecutor → [execute / compile]
              → CompileTeX ↔ FixTeX → Finisher

PlanDrivenExecutor executes each plan step in order (research or write).
ReviewExecutor appends revision steps to the plan tail and loops back.
LaTeX compilation happens exactly once, as the final PDF generation step.
"""

from pocketflow import Flow

from .nodes import (
    Initializer,
    PlanInitialExecutor,
    PlanDrivenExecutor,
    ReviewExecutor,
    CompileTeX,
    FixTeX,
    Finisher,
)


def create_scientist_flow() -> Flow:
    init     = Initializer()
    planner  = PlanInitialExecutor(max_retries=2, wait=3)
    executor = PlanDrivenExecutor(max_retries=2, wait=5)
    review   = ReviewExecutor(max_retries=2, wait=5)
    compile  = CompileTeX(max_retries=1, wait=0)
    fix_tex  = FixTeX(max_retries=2, wait=3)
    finisher = Finisher()

    init     - "research" >> planner
    planner  - "execute"  >> executor

    executor - "execute"  >> executor   # loop through plan steps
    executor - "review"   >> review

    review   - "execute"  >> executor   # revision steps appended to plan
    review   - "compile"  >> compile

    compile  - "fix"      >> fix_tex
    compile  - "done"     >> finisher
    fix_tex  - "compile"  >> compile
    fix_tex  - "done"     >> finisher

    return Flow(start=init)
