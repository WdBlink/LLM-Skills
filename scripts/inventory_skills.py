#!/usr/bin/env python3
"""Read-only skill inventory. Installation origin is not proof of authorship."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def inventory(roots, lock=None, packages=None, owner='WdBlink', catalog=None):
    records = {}
    groups = {}
    registered = (lock or {}).get('skills', {})
    reviewed = (catalog or {}).get('skills', {})
    package_paths = {}
    for label, root in (packages or {}).items():
        for entry in Path(root).glob('*/SKILL.md'):
            package_paths[str(entry.parent.resolve())] = label
    missing = []
    for root in roots:
        root = Path(root).expanduser()
        if not root.is_dir():
            missing.append(str(root))
            continue
        # Gather every alias before interpreting installation metadata.
        for folder in sorted(root.iterdir()):
            if (folder / 'SKILL.md').is_file():
                groups.setdefault(str(folder.resolve()), set()).add(str(folder))
    for canonical, locations in sorted(groups.items()):
        name = Path(canonical).name
        names = sorted({name} | {Path(location).name for location in locations})
        digest = hashlib.sha256((Path(canonical) / 'SKILL.md').read_bytes()).hexdigest()
        sources = sorted({str(registered[n]['source']) for n in names
                          if n in registered and registered[n].get('source')})
        matches = [reviewed[n] for n in names if n in reviewed
                   and digest == reviewed[n].get('skill_md_sha256')]
        origin, evidence = 'unknown', []
        if canonical in package_paths:
            origin = 'third_party_package'
            evidence.append('resolved package member: ' + package_paths[canonical])
        elif len(sources) > 1:
            origin = 'conflicting_install_records'
        elif sources:
            origin = 'owner_repository' if sources[0].lower().startswith(owner.lower() + '/') else 'third_party_install'
        evidence.extend('name-matched installer source (package content unverified): ' + source for source in sources)
        if matches:
            classifications = {decision['classification'] for decision in matches}
            if origin == 'unknown':
                origin = next(iter(classifications)) if len(classifications) == 1 else 'conflicting_reviews'
            for decision in matches:
                evidence.extend(decision.get('evidence', []))
        records[canonical] = {
            'name': name, 'names': names, 'canonical_path': canonical,
            'locations': sorted(locations), 'skill_md_sha256': digest,
            'origin': origin, 'installer_source': sources[0] if len(sources) == 1 else None,
            'installer_sources': sources,
            'installer_match': 'name_only_not_content_verified' if sources else None,
            'evidence': sorted(set(evidence)),
            'review_matches_entry': bool(matches),
        }
    rows = sorted(records.values(), key=lambda row: (row['name'], row['canonical_path']))
    return {
        'schema_version': 1,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'Top-level SKILL.md packages; symlinks deduplicated by resolved path. Separate physical copies remain separate.',
        'limits': 'Installer records are matched by name, not by installation path or package content; a local same-named replacement may have a different origin. Records do not prove authorship or redistribution rights. Review hashes cover SKILL.md only. Missing metadata means unknown.',
        'roots': [str(Path(root).expanduser()) for root in roots],
        'missing_roots': missing,
        'counts': dict(Counter(row['origin'] for row in rows)),
        'skills': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', action='append', type=Path, help='Repeat for each skill root')
    parser.add_argument('--lock', type=Path, default=Path.home() / '.agents/.skill-lock.json')
    parser.add_argument('--package', action='append', default=[], metavar='NAME=PATH', help='Known third-party package skill root; resolve membership by path')
    parser.add_argument('--owner', default='WdBlink', help='GitHub owner used only for installer source classification')
    parser.add_argument('--catalog', type=Path, help='Optional reviewed provenance JSON, bound to entry hashes')
    parser.add_argument('--out', type=Path, help='Local JSON output; default stdout. May contain private paths.')
    args = parser.parse_args()
    packages = {}
    for value in args.package:
        label, separator, path = value.partition('=')
        if not label or not separator or not path:
            parser.error('--package requires NAME=PATH')
        packages[label] = Path(path).expanduser()
        if not packages[label].is_dir():
            parser.error(f'package root does not exist: {path}')
    roots = args.root or [Path.home()/'.codex/skills', Path.home()/'.agents/skills', Path.home()/'.claude/skills']
    report = inventory(roots, read_json(args.lock) if args.lock.exists() else {}, packages,
                       args.owner, read_json(args.catalog) if args.catalog else {})
    content = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.out:
        target = args.out.resolve()
        protected = [args.lock.resolve()] + ([args.catalog.resolve()] if args.catalog else [])
        if target in protected or any(target == Path(root).expanduser().resolve() or Path(root).expanduser().resolve() in target.parents for root in roots):
            parser.error('--out must not overwrite source metadata or write into an installed skill root')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(content, encoding='utf-8')
        print(json.dumps({'output': str(args.out), 'counts': report['counts']}, ensure_ascii=False))
    else:
        print(content, end='')


if __name__ == '__main__':
    main()
