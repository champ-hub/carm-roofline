---
name: fix-tests
description: Fix failing tests after significant source code changes.
---
Use #tool:agent/runSubagent to fix the failing pytest tests. You must not do any code edits yourself. Check the failing tests and issue multiple parallel subagents, assign tests to each based on their shared characteristics. The agents MUST work in parallel. If the code has changed significantly: if tests are no longer relevant, the agents are free to delete them or modify them significantly, but within reason. Ensure the test coverage remains good and that the tests are meaningful.
