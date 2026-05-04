# Paper Operator Start Checklist

This checklist is planning documentation only. No process was started, stopped, polled, or managed.

## Required Before Any Future Start

- Confirm the paper process executor plan status is `ready`.
- Confirm the source paper execution request status is `ready` and still references the same strategy.
- Confirm the reviewed start command exactly matches the execution request command.
- Confirm process metadata, status snapshot, stdout, stderr, and paper metrics paths are local workspace paths.
- Confirm stop and cleanup artifacts have been reviewed before startup.
- Confirm a separate explicit user request authorizes the exact future start action.

## Startup Boundaries

- Do not use API keys, secrets, private environment values, or credential-bearing configs.
- Do not start live trading, canary live trading, exchange order placement, leverage above 1.0, or shorting.
- Preserve stdout, stderr, process metadata, status snapshots, paper metrics, and source-of-truth artifacts.

- Plan status: blocked
- Strategy: LongOnlyFreqAIStrategy
- Run ID: phase3_paper_executor_plan_20260503
