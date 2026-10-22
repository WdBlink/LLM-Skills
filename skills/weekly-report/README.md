# Weekly Report

![Reserved](https://img.shields.io/badge/rights-reserved-blueviolet)
![Version](https://img.shields.io/badge/version-0.1.1-blue)

Portable Codex skill for generating Chinese weekly report DOCX files from work
logs. It does not require a separate source-project checkout.

## Install

```bash
mkdir -p ~/.codex/skills/weekly-report
rsync -a skills/weekly-report/ ~/.codex/skills/weekly-report/
```

## Use

```bash
python3 ~/.codex/skills/weekly-report/scripts/generate_weekly_report.py --dry-run
python3 ~/.codex/skills/weekly-report/scripts/generate_weekly_report.py --log-dir /path/to/logs --output-dir /path/to/reports --days 7
python3 ~/.codex/skills/weekly-report/scripts/generate_weekly_report.py --logs-file /path/to/log.md --template /path/to/template.docx --output-dir /path/to/reports --no-api
```

## API configuration

Credentials, endpoint and model are selected together from one environment-variable
prefix. The priority is `WEEKLY_REPORT`, `MINIMAX`, `OPENAI`, then `SILICONFLOW`.
Settings from other prefixes are ignored. For OpenAI and SiliconFlow, set the
matching `*_MODEL`; a key alone does not choose a model.

For a custom endpoint, set all three: `WEEKLY_REPORT_API_KEY`,
`WEEKLY_REPORT_BASE_URL`, and `WEEKLY_REPORT_MODEL`. A partial custom configuration
fails locally and uses the existing fallback summary without sending a request.
The URL may be an API base ending in `/v1` or a full `/chat/completions` endpoint.
See [configuration and defaults](references/workflow.md#llm-configuration).

Run offline credential-routing checks with:

```bash
python3 -m unittest discover -s skills/weekly-report/tests -v
```
