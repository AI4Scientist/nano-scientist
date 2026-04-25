Analyze real-world AI agent behavior using the AIDev dataset (https://huggingface.co/datasets/hao-li/AIDev), which contains 933k pull requests authored by five AI agents — OpenAI Codex, GitHub Copilot, Cursor, Devin, and Claude Code — across 116k GitHub repositories (December 2024–August 2025).

Using the `all_pull_request`, `pr_task_type`, `pr_timeline`, `pr_commits`, and `pr_reviews` tables, conduct the following analyses:

1. **Acceptance rate by agent and task type**: Compare merge rates (`merged_at` not null) across agents and Conventional Commits task categories (`feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`). Identify which agents succeed most on which task types.

2. **Review dynamics**: Using `pr_reviews` and `pr_review_comments_v2`, measure how often agent PRs receive change requests vs. immediate approval, and whether rejection rates differ by agent or task type.

3. **Cycle time analysis**: Compute time-to-merge (`merged_at - created_at`) and time-to-first-review from `pr_timeline`, stratified by agent and task type. Test whether autonomous agents (Devin, Claude Code) have longer or shorter cycles than IDE-paired agents (Copilot, Cursor).

4. **Code churn**: Using `pr_commit_details`, measure average files touched and patch size per PR per agent. Correlate churn with acceptance outcome.

5. **Cross-agent comparison on the AIDev-pop subset** (repos >100 stars): Compare behavioral profiles of all five agents on the same high-quality repository pool to control for repository quality effects.
