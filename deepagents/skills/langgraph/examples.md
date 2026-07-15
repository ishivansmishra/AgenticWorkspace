# LangGraph Skill Examples

## Example 1: Basic Workflow
User: "Create a LangGraph with `plan -> execute -> summarize`."
Assistant behavior: defines state type, three nodes, and deterministic edges.

## Example 2: Human-in-the-loop
User: "Pause for approval before sending email."
Assistant behavior: inserts an approval checkpoint/interrupt node before side-effect step.

## Example 3: Failure Handling
User: "Tool calls fail intermittently."
Assistant behavior: adds guarded retry policy, fallback branch, and explicit error state.
