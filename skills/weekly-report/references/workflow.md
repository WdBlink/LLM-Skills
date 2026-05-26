# Weekly Report Workflow

Use this reference when generating, debugging, or adapting a weekly report in a
new environment. The skill is self-contained and must not assume access to a
specific local checkout or external project entrypoint.

## Inputs

- `--log-dir`: directory containing recent `.md`, `.txt`, or `.docx` work logs.
- `--logs-file`: a single log file when the user provides one source file.
- `--template`: optional `.docx` template. If omitted, the script creates a
  basic weekly report document.
- `--output-dir`: destination for the generated `.docx`.

By default, directory mode only includes files containing `#工作日志`. Use
`--include-untagged` when the user's source material does not use that marker.

## LLM Configuration

The script can call any OpenAI-compatible chat-completions endpoint. It checks
these environment variables in order:

- API key: `WEEKLY_REPORT_API_KEY`, `MINIMAX_API_KEY`, `OPENAI_API_KEY`,
  `SILICONFLOW_API_KEY`
- Base URL: `WEEKLY_REPORT_BASE_URL`, `MINIMAX_BASE_URL`, `OPENAI_BASE_URL`,
  `SILICONFLOW_BASE_URL`
- Model: `WEEKLY_REPORT_MODEL`, `MINIMAX_MODEL`, `OPENAI_MODEL`,
  `SILICONFLOW_MODEL`

Use `--no-api` for deterministic fallback generation or smoke tests.

## Output Style

Keep "计划完成情况" aligned one-to-one with the extracted plan/task list. Avoid
generic AI-style report phrases and do not invent metrics, completion ratios,
or experimental results that are not present in the logs.

Avoid these phrases unless they appear in source evidence and are genuinely
needed: 扎实推进, 持续赋能, 显著提升, 形成闭环, 关键路径, 抓手, 落地见效, 稳步推进,
阶段性成果, 进一步加强, 不断优化.

## Template Filling

When a template is provided, the script looks for table headers similar to:

- 本周工作计划 / 工作计划 / 本周计划
- 主要进展及成果 / 计划完成情况 / 完成情况 / 主要进展
- 存在的问题 / 存在问题 / 问题
- 下周工作计划 / 下周计划

If a template does not expose recognizable table headers, the script appends a
plain "周报内容" section instead of failing.
