# LangGraph Skill Instructions

1. Start by defining graph objective and state schema.
2. Keep node responsibilities focused and composable.
3. Make transitions explicit and deterministic when possible.
4. Use retries/guards for tool or network instability.
5. Preserve traceability with meaningful node names and comments only where needed.
6. Validate edge cases: empty state, partial outputs, and tool failures.
7. Prefer patterns compatible with production monitoring and checkpoint recovery.
