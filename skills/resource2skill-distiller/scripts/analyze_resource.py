"""Distill extracted webpage text, Markdown, or a transcript with Codex CLI."""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import os
import sys
from urllib.parse import urlsplit

from codex_runner import run_codex


def source_url(value):
    url = urlsplit(value)
    if url.scheme != 'https' or not url.hostname or url.username or url.password or url.port:
        raise ValueError('Provide a public HTTPS source URL without credentials or a port.')
    return value


def distiller_prompt(repo, domain=None, prompt_file=None):
    if domain:
        sys.path.insert(0, str(repo.expanduser().resolve()))
        from core import get_distiller_prompt, list_domains, load_domain
        if domain not in list_domains():
            raise ValueError('Unknown Resource2Skill domain; supply a matching prompt file.')
        return get_distiller_prompt(load_domain(domain))
    path = prompt_file or Path(__file__).resolve().parents[1] / 'references/method-distiller.md'
    return path.expanduser().read_text()


def common_arguments(parser):
    parser.add_argument('--source-url', required=True, type=source_url)
    parser.add_argument('--goal', required=True)
    parser.add_argument('--repo', type=Path, default=Path(os.environ.get('RESOURCE2SKILL_HOME', '~/.local/share/resource2skill')))
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--domain')
    group.add_argument('--prompt-file', type=Path)
    parser.add_argument('--model', help='Optional Codex model override; defaults to the local base configuration.')
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--output-dir', type=Path, required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common_arguments(parser)
    parser.add_argument('--text-file', required=True, type=Path)
    args = parser.parse_args()
    path = args.text_file.expanduser().resolve()
    if not path.is_file() or not 0 < path.stat().st_size <= 1_500_000 or not args.goal.strip():
        parser.error('Provide a nonempty text file up to 1.5 MB and a nonempty goal.')
    content = path.read_text()
    prompt = distiller_prompt(args.repo, args.domain, args.prompt_file)
    metadata = {'url': args.source_url, 'local_text': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'domain': args.domain, 'goal': args.goal, 'coverage': 'supplied text only',
                'created_at': datetime.now(timezone.utc).isoformat()}
    run_codex(prompt + '\n\n提炼目标：' + args.goal + '\n来源：' + args.source_url + '\n\n来源正文：\n' + content,
              args.output_dir, model=args.model, timeout=args.timeout, metadata=metadata)
    print('Codex extraction completed; content and skill validation remain separate.')


if __name__ == '__main__':
    main()
