#!/usr/bin/env python3
"""Prepare source snapshots, accept a host explanation, publish and bind its review.

No model, browser, network or shell execution. Runs are immutable; revisions use
new output directories. Source freshness and content hashes prevent stale reuse,
not forgery by a party able to replace all inputs and hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from judgment_handoff import (HandoffError, MAX_JSON_BYTES, canonical, nonempty_string, protect_inputs,
                              read_bounded, read_json, require, same_content, split_units,
                              write_json_atomic)
from check_evidence import validate as check_evidence
from render_explanation import render_evidence_markdown, render_html, render_markdown

ROOT = Path(__file__).resolve().parents[1]
FORMATS = ('auto', 'text', 'diagram', 'html')
KINDS = ('observation', 'inference', 'missing', 'conflicting', 'recommendation', 'explanation', 'demonstration')
CHECKS = ('facts', 'language', 'understanding', 'visual', 'interaction')
MAX_CHARS = 50_000


def digest(data):
    return hashlib.sha256(data).hexdigest()


def identity(value):
    return 'sha256:' + digest(canonical(value))


def fields(value, required, optional=()):
    require(isinstance(value, dict) and set(required) <= set(value) <= set(required) | set(optional),
            'Expected fields: ' + ', '.join(required) + '; optional: ' + ', '.join(optional))


def strings(value, name):
    require(isinstance(value, list) and len(value) <= 1000, name + ' must be an array of at most 1000 strings')
    for item in value:
        nonempty_string(item, name, MAX_CHARS)
    return value


def rules_hash():
    names = ['SKILL.md', 'references/explanation-methods.md', 'references/workflow-contract.md',
             'scripts/run.py', 'scripts/judgment_handoff.py', 'scripts/render_explanation.py',
             'scripts/check_evidence.py']
    return identity({name: digest((ROOT / name).read_bytes()) for name in names})


def snapshot(path, index):
    path = path.resolve()
    raw = read_bounded(path, 4 * MAX_CHARS + 3)
    text = raw.decode('utf-8-sig')
    require(len(text) <= MAX_CHARS, f'{path}: more than {MAX_CHARS} characters; no truncation performed')
    sid = f'S{index}'
    units = [{'id': f'{sid}:P{i}', 'text': unit['text']} for i, unit in enumerate(split_units(text), 1)]
    return {'id': sid, 'path': str(path), 'sha256': digest(raw), 'units': units}, len(text)


def evidence_view(path):
    path = path.resolve()
    bundle = read_json(path)
    report = check_evidence(bundle, path.parent)
    require(report['status'] != 'fail', 'Evidence failed: ' + '; '.join(report['errors']))
    for source in bundle['sources']:
        if source.get('path'):
            source['path'] = str((path.parent / source['path']).resolve())
    return {'path': str(path), 'sha256': digest(path.read_bytes()), 'bundle': bundle, 'report': report}


def prepare(args):
    require(1 <= args.max_chars <= MAX_CHARS, f'--max-chars must be 1..{MAX_CHARS}')
    task = {'question': nonempty_string(args.question, 'question'),
            'reader': nonempty_string(args.reader, 'reader'),
            'scope': nonempty_string(args.scope, 'scope'), 'format': args.format}
    evidence = evidence_view(args.evidence) if args.evidence else None
    paths = [p.resolve() for p in args.source]
    if evidence:
        paths += [Path(s['path']) for s in evidence['bundle']['sources'] if s['status'] == 'available']
    paths = list(dict.fromkeys(paths))
    protect_inputs(args.out, paths + ([args.evidence] if args.evidence else []))
    sources, count = [], 0
    for i, path in enumerate(paths, 1):
        source, size = snapshot(path, i)
        sources.append(source)
        count += size
    require(count <= args.max_chars, f'Combined sources have {count} characters; cap {args.max_chars}; no truncation performed')
    core = {'version': 1, 'task': task, 'sources': sources, 'gaps': strings(args.missing, 'missing'),
            'evidence': evidence, 'rules_sha256': rules_hash()}
    request = {**core, 'request_id': identity(core)}
    require(len(json_bytes(request)) <= MAX_JSON_BYTES, 'Prepared request exceeds 1 MiB; reduce the material scope explicitly')
    write = write_json_atomic(args.out, request)
    return {'status': 'needs_judgment', 'request': str(args.out.resolve()), 'request_id': request['request_id'],
            'characters': count, 'source_count': len(sources), 'write': write,
            'rules': str(ROOT / 'references/explanation-methods.md'),
            'contract': str(ROOT / 'references/workflow-contract.md')}


def load_request(path):
    request = read_json(path)
    fields(request, ('version', 'task', 'sources', 'gaps', 'evidence', 'rules_sha256', 'request_id'))
    require(type(request['version']) is int and request['version'] == 1, 'Unsupported version')
    core = {k: v for k, v in request.items() if k != 'request_id'}
    require(request['request_id'] == identity(core), 'Request hash mismatch; prepare again')
    require(request['rules_sha256'] == rules_hash(), 'Skill implementation/rules changed; prepare and judge again')
    fields(request['task'], ('question', 'reader', 'scope', 'format'))
    for key in ('question', 'reader', 'scope'):
        nonempty_string(request['task'][key], key)
    require(request['task']['format'] in FORMATS, 'Unknown requested format')
    strings(request['gaps'], 'gaps')
    require(isinstance(request['sources'], list), 'sources must be an array')
    total = 0
    for i, source in enumerate(request['sources'], 1):
        fields(source, ('id', 'path', 'sha256', 'units'))
        nonempty_string(source['path'], 'source.path')
        current, size = snapshot(Path(source['path']), i)
        require(current == source, 'Source changed or snapshot invalid; prepare and judge again: ' + source['path'])
        total += size
    require(total <= MAX_CHARS, 'Combined sources exceed hard character cap')
    if request['evidence'] is not None:
        fields(request['evidence'], ('path', 'sha256', 'bundle', 'report'))
        current = evidence_view(Path(nonempty_string(request['evidence']['path'], 'evidence.path')))
        require(current == request['evidence'], 'Evidence changed; prepare and judge again')
    return request


def validate_decision(decision, request):
    require(isinstance(decision, dict), 'Decision must be an object')
    require(decision.get('request_id') == request['request_id'], 'Stale or unrelated decision request_id')
    if decision.get('status') == 'needs_input':
        fields(decision, ('request_id', 'status', 'missing'))
        require(bool(strings(decision['missing'], 'missing')), 'needs_input requires missing material')
        return
    fields(decision, ('request_id', 'status', 'format', 'format_reason', 'title', 'answer', 'limitations', 'sections'))
    require(decision['status'] == 'ready', 'Decision status must be ready or needs_input')
    require(decision['format'] in FORMATS[1:], 'Unknown output format')
    require(request['task']['format'] in ('auto', decision['format']), 'Requested format must be preserved')
    for key in ('format_reason', 'title', 'answer'):
        nonempty_string(decision[key], key, MAX_CHARS)
    strings(decision['limitations'], 'limitations')
    sections = decision['sections']
    require(isinstance(sections, list) and 1 <= len(sections) <= 100, 'sections must contain 1..100 items')
    unit_ids = {u['id'] for s in request['sources'] for u in s['units']}
    evidence_ids = {e['id'] for e in request['evidence']['bundle']['evidence']} if request['evidence'] else set()
    seen = set()
    for section in sections:
        fields(section, ('id', 'title', 'kind', 'text', 'scope', 'unit_ids', 'evidence_ids', 'limitations'), ('example',))
        for key in ('id', 'title', 'text', 'scope'):
            nonempty_string(section[key], key, MAX_CHARS)
        require(section['id'] not in seen, 'Duplicate section id')
        seen.add(section['id'])
        require(section['kind'] in KINDS, 'Unknown section kind')
        for key, known in (('unit_ids', unit_ids), ('evidence_ids', evidence_ids)):
            refs = strings(section[key], key)
            require(len(refs) == len(set(refs)) and set(refs) <= known, 'Duplicate or unknown ' + key)
        refs = section['unit_ids'] + section['evidence_ids']
        require(section['kind'] != 'observation' or refs, 'Observation needs a source reference')
        require(section['kind'] != 'conflicting' or len(refs) >= 2, 'Conflict needs at least two source references')
        strings(section['limitations'], 'section.limitations')
        if 'example' in section:
            example = section['example']
            fields(example, ('kind', 'input', 'rule', 'intermediate', 'output'))
            require(example['kind'] in ('source', 'demonstration'), 'Example kind must be source or demonstration')
            require(example['kind'] != 'source' or refs, 'Source example requires a reference')
            for key in ('input', 'rule', 'intermediate', 'output'):
                nonempty_string(example[key], 'example.' + key, MAX_CHARS)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def publish(directory, contents, inputs):
    # Preflight all outputs before publishing; a receipt is always published last.
    # shortcut: single-writer runs, not a multi-file transaction; rerun identical content after interruption.
    from tempfile import NamedTemporaryFile
    import os
    for name, data in contents.items():
        require(len(data) <= 10 * 1024 * 1024, f'Output exceeds 10 MiB: {name}')
        path = directory / name
        protect_inputs(path, inputs)
        require(not path.is_symlink(), f'Refuse symlink output: {path}')
        require(not path.exists() or same_content(path, data), f'Output conflict: {path}; use a new run directory')
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in contents.items():
        path = directory / name
        if same_content(path, data):
            continue
        temporary = None
        try:
            with NamedTemporaryFile(dir=directory, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                require(same_content(path, data), f'Concurrent output conflict: {path}')
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)


def source_paths(request):
    return [Path(s['path']) for s in request['sources']] + ([Path(request['evidence']['path'])] if request['evidence'] else [])


def apply(args):
    request = load_request(args.request)
    decision = read_json(args.decision)
    validate_decision(decision, request)
    inputs = [args.request, args.decision] + source_paths(request)
    if args.artifact:
        inputs.append(args.artifact)
    if decision['status'] == 'needs_input':
        require(not args.artifact, 'needs_input cannot publish an artifact')
        publish(args.out, {'pending.json': json_bytes(decision)}, inputs)
        return {'status': 'needs_input', 'pending': str((args.out / 'pending.json').resolve()), 'missing': decision['missing']}
    contents = {'explanation.md': render_markdown(request, decision).encode(),
                'evidence.md': render_evidence_markdown(request, decision).encode()}
    if args.artifact:
        suffix = args.artifact.suffix.lower()
        allowed = ('.html',) if decision['format'] == 'html' else ('.svg', '.png', '.jpg', '.webp', '.mmd')
        require(decision['format'] in ('html', 'diagram') and suffix in allowed, 'Artifact type must match diagram/html format')
        name = 'artifact' + suffix
        contents[name] = read_bounded(args.artifact, 10 * 1024 * 1024)
        require(bool(contents[name]), 'Artifact is empty')
        contents['explanation.md'] += f'\n产物：[打开解释]({name})。图或页面中的说明也须遵守同一证据与表达规则。\n'.encode()
    elif decision['format'] == 'diagram':
        raise HandoffError('diagram needs --artifact; use an available diagram tool and inspect its output')
    elif decision['format'] == 'html':
        contents['explanation.html'] = render_html(request, decision).encode()
    hashes = {name: digest(data) for name, data in contents.items()}
    review_request = {'request_id': request['request_id'], 'artifact_hashes': hashes,
                      'format': decision['format'], 'required_checks': list(CHECKS)}
    contents['review-request.json'] = json_bytes(review_request)
    receipt = {'status': 'needs_review', **review_request, 'decision_sha256': digest(args.decision.read_bytes()),
               'evidence_check': request['evidence']['report'] if request['evidence'] else 'not_applicable',
               'semantic_review': 'not_checked', 'interaction_review': 'not_checked'}
    contents['receipt.json'] = json_bytes(receipt)
    publish(args.out, contents, inputs)
    return {'status': 'needs_review', 'output': str(args.out.resolve()), 'review_request': str((args.out / 'review-request.json').resolve()),
            'artifact_hashes': hashes, 'semantic_review': 'not_checked'}


def finish(args):
    request = load_request(args.request)
    pending = read_json(args.dir / 'review-request.json')
    fields(pending, ('request_id', 'artifact_hashes', 'format', 'required_checks'))
    require(pending['request_id'] == request['request_id'], 'Review request is stale')
    require(pending['format'] in FORMATS[1:], 'Invalid review format')
    require(pending['required_checks'] == list(CHECKS), 'Review checks changed')
    require(request['task']['format'] in ('auto', pending['format']), 'Review format must preserve requested format')
    receipt = read_json(args.dir / 'receipt.json')
    require(isinstance(receipt, dict) and receipt.get('status') == 'needs_review', 'Invalid apply receipt')
    require(all(receipt.get(key) == value for key, value in pending.items()),
            'Review request differs from apply receipt; apply again')
    hashes = pending['artifact_hashes']
    required = {'explanation.md', 'evidence.md'}
    require(isinstance(hashes, dict) and required <= set(hashes), 'Missing required explanation/evidence artifacts')
    extra = set(hashes) - required
    if pending['format'] == 'text':
        require(not extra, 'Text output must contain only explanation.md and evidence.md')
    else:
        allowed = ({'explanation.html', 'artifact.html'} if pending['format'] == 'html' else
                   {'artifact' + suffix for suffix in ('.svg', '.png', '.jpg', '.webp', '.mmd')})
        require(len(extra) == 1 and extra <= allowed, 'Artifact set must match review format')
    for name, sha in hashes.items():
        require(isinstance(name, str) and Path(name).name == name and name not in ('.', '..'), 'Invalid artifact filename')
        require(digest(read_bounded(args.dir / name, 10 * 1024 * 1024)) == sha, 'Artifact changed; apply and review again: ' + name)
    review = read_json(args.review)
    fields(review, ('request_id', 'artifact_hashes', 'checks'))
    require(review['request_id'] == request['request_id'] and review['artifact_hashes'] == hashes, 'Stale review binding')
    fields(review['checks'], CHECKS)
    states = []
    for name, check in review['checks'].items():
        fields(check, ('status', 'note'))
        nonempty_string(check['note'], name + '.note')
        require(check['status'] in ('passed', 'failed', 'not_run', 'not_applicable'), 'Invalid check status')
        applicable = name not in ('visual', 'interaction') or (name == 'visual' and pending['format'] != 'text') or pending['format'] == 'html'
        require(not applicable or check['status'] != 'not_applicable', name + ' is required for this format')
        states.append(check['status'])
    status = 'needs_revision' if 'failed' in states else ('complete_with_limits' if 'not_run' in states else 'complete')
    result = {**review, 'status': status, 'review_type': 'host_reported_checks',
              'note': '程序验证绑定和检查记录；不认证语义、浏览器执行或独立审查。'}
    protect_inputs(args.out, [args.request, args.review, args.dir / 'review-request.json', args.dir / 'receipt.json'] +
                   [args.dir / name for name in hashes] + source_paths(request))
    write_json_atomic(args.out, result)
    return {'status': status, 'output': str(args.out.resolve()), 'checks': review['checks']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='Snapshot UTF-8 sources and bind a host judgment request')
    prep.add_argument('--source', type=Path, action='append', default=[])
    prep.add_argument('--evidence', type=Path)
    prep.add_argument('--question', required=True)
    prep.add_argument('--reader', default='用户，背景未另说明')
    prep.add_argument('--scope', default='仅本次列出的材料与已披露缺口')
    prep.add_argument('--format', choices=FORMATS, default='auto')
    prep.add_argument('--missing', action='append', default=[])
    prep.add_argument('--max-chars', type=int, default=12_000)
    prep.add_argument('--out', type=Path, required=True)
    prep.set_defaults(handler=prepare)
    use = sub.add_parser('apply', help='Validate host judgment and assemble artifacts; no execution of custom HTML')
    use.add_argument('--request', type=Path, required=True)
    use.add_argument('--decision', type=Path, required=True)
    use.add_argument('--artifact', type=Path)
    use.add_argument('--out', type=Path, required=True, help='Output directory; different content never overwrites')
    use.set_defaults(handler=apply)
    end = sub.add_parser('finish', help='Bind host review to current artifacts; disclose unperformed checks')
    end.add_argument('--request', type=Path, required=True)
    end.add_argument('--dir', type=Path, required=True)
    end.add_argument('--review', type=Path, required=True)
    end.add_argument('--out', type=Path, required=True)
    end.set_defaults(handler=finish)
    args = parser.parse_args()
    try:
        print(json.dumps(args.handler(args), ensure_ascii=False, allow_nan=False))
        return 0
    except (HandoffError, OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(json.dumps({'status': 'failed', 'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
