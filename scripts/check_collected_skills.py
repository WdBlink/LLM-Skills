#!/usr/bin/env python3
"""Run offline regressions for the skills collected or repaired in this release."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
SUITES = [
    ['-m', 'unittest', 'discover', '-s', 'tests', '-v'],
    ['-m', 'unittest', 'discover', '-s', 'skills/ms-qwen-vl/tests', '-v'],
    ['-m', 'unittest', 'discover', '-s', 'skills/weekly-report/tests', '-v'],
    ['-m', 'unittest', 'discover', '-s', 'skills/mind-map/tests', '-v'],
    ['-m', 'unittest', 'discover', '-s', 'skills/human-understanding/tests', '-v'],
    ['-m', 'unittest', 'discover', '-s', 'skills/workflow-skill-creator/tests', '-v'],
    ['-m', 'unittest', 'discover', '-s', 'skills/workflow-skill-creator/evals', '-p', 'test_*.py', '-v'],
    ['skills/workflow-skill-creator/assets/evidence-brief/scripts/check.py'],
    ['skills/project-handoff/scripts/check.py'],
    ['skills/research-context/scripts/check.py'],
    ['skills/resource2skill-distiller/scripts/test_adapter.py'],
]


def main():
    for suite in SUITES:
        print('Running:', ' '.join(suite), flush=True)
        subprocess.run([sys.executable, '-B', *suite], cwd=ROOT, check=True, timeout=90)
    print('Offline regressions passed. Live model, external services and UI quality were not tested.')


if __name__ == '__main__':
    main()
