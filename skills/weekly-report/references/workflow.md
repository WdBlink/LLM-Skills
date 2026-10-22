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

The script selects one configuration group by the first nonempty API key in
this order: `WEEKLY_REPORT`, `MINIMAX`, `OPENAI`, `SILICONFLOW`. It reads that
group's `*_API_KEY`, `*_BASE_URL`, and `*_MODEL` together; it never takes a key
from one group and an endpoint or model from another.

| Prefix | Default API base | Model |
| --- | --- | --- |
| `WEEKLY_REPORT` | Required | Required |
| `MINIMAX` | `https://api.minimaxi.com/v1` | Existing default `MiniMax-M2.5`; override with `MINIMAX_MODEL` |
| `OPENAI` | `https://api.openai.com/v1` | Set `OPENAI_MODEL` explicitly |
| `SILICONFLOW` | `https://api.siliconflow.cn/v1` | Set `SILICONFLOW_MODEL` explicitly |

A custom `WEEKLY_REPORT_BASE_URL` or `WEEKLY_REPORT_MODEL` requires its own
`WEEKLY_REPORT_API_KEY`. Missing selected settings fail locally and use the
fallback summary; the script does not try a different provider. To change
providers when several keys are installed, supply a complete `WEEKLY_REPORT`
group or unset the higher-priority key for that invocation.

`*_BASE_URL` accepts an API base or a full `/chat/completions` URL. The script
adds `/chat/completions` only when absent, rejects embedded credentials/query
strings/fragments, and does not follow HTTP redirects. Explicit same-prefix
endpoint overrides support an authorized compatible gateway or local service.

`WEEKLY_REPORT_MAX_TOKENS` and `WEEKLY_REPORT_TEMPERATURE` override generation
settings. Otherwise use the selected group's matching settings, then defaults
of 1200 tokens and 0.4. The script does not borrow these values from another
provider.

Endpoint references: [OpenAI Chat Completions](https://developers.openai.com/api/reference/resources/chat),
[SiliconFlow Chat Completions](https://docs.siliconflow.cn/docs/api/chat-completions-post),
and [MiniMax's OpenAI-compatible API](https://platform.minimax.io/docs/api-reference/text-openai-api).
The MiniMax default preserves this skill's existing domestic endpoint and
model; use same-prefix overrides for another region or model. Account/model
availability must be checked separately before real use.

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
