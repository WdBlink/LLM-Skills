# Weekly Report

![Private](https://img.shields.io/badge/Private-blueviolet)
![Version](https://img.shields.io/badge/version-0.1.0-blue)

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
