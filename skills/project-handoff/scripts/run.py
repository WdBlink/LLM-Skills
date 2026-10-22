#!/usr/bin/env python3
"""Stdlib-only project handoff coordinator. The host supplies evidence and runs agents."""
import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.parse import quote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {'report'}
INPUT_FIELDS = {'project_name', 'project_root', 'request', 'generated_at', 'version',
                'coverage', 'synthetic', 'sources'}
ARTIFACTS = ('input.json', 'skeleton.json', 'plan.json', 'fill_map.json', 'draft.json', 'draft.md',
             'critique-0.json', 'patch.json', 'revised.json', 'revised.md',
             'critique-1.json', 'publication.json')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'JSON 字段重复：{key}')
        result[key] = value
    return result


def invalid_constant(value):
    raise ValueError(f'不允许非标准 JSON 常量 {value}')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique_object,
                      parse_constant=invalid_constant)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def keys(value, expected, label):
    require(isinstance(value, dict) and set(value) == set(expected),
            f'{label} 必须恰好包含字段 {sorted(expected)}')


def nonempty(value, label):
    require(isinstance(value, str) and bool(value.strip()), f'{label} 必须是非空文本')


def absolute(value, label):
    nonempty(value, label)
    require(Path(value).is_absolute(), f'{label} 必须是绝对路径')
    require('\x00' not in value, f'{label} 不允许 NUL')
    return Path(value)


def source_input(value):
    keys(value, INPUT_FIELDS, 'input')
    for field in INPUT_FIELDS - {'synthetic', 'sources'}:
        nonempty(value[field], field)
    absolute(value['project_root'], 'project_root')
    timestamp = datetime.fromisoformat(value['generated_at'].replace('Z', '+00:00'))
    require(timestamp.tzinfo is not None and timestamp.utcoffset() is not None,
            'generated_at 必须是带时区的 ISO 8601 时间')
    require(type(value['synthetic']) is bool, 'synthetic 必须为布尔值')
    require(isinstance(value['sources'], list) and value['sources'], 'sources 至少需要一条记录')
    ids = set()
    for source in value['sources']:
        keys(source, {'id', 'kind', 'locator', 'location', 'status', 'scope', 'excerpt'}, 'source')
        for field in ('id', 'kind', 'locator', 'location', 'status', 'scope'):
            nonempty(source[field], f'source.{field}')
        require(re.fullmatch(r'S[1-9][0-9]*', source['id']) is not None, '来源 id 应形如 S1')
        require(source['id'] not in ids, '来源 id 不可重复')
        ids.add(source['id'])
        require(source['kind'] in ('file', 'web', 'session', 'unknown'), '无效来源 kind')
        require(source['status'] in ('read', 'unread', 'unavailable', 'not_found'), '无效来源 status')
        require(isinstance(source['excerpt'], str), 'excerpt 必须是文本')
        if source['status'] == 'read':
            nonempty(source['excerpt'], '已读来源 excerpt')
        else:
            require(source['excerpt'] == '', '未读或不可得来源 excerpt 必须为空')
        if source['kind'] == 'file':
            absolute(source['locator'], 'file locator')
        elif source['kind'] == 'web':
            url = urlsplit(source['locator'])
            require(url.scheme in ('http', 'https') and bool(url.netloc), 'web locator 必须是 HTTP(S) URL')
            require(url.username is None and url.password is None, 'web locator 不允许用户名或密码；请先脱敏')
    return value


def prose_without_code(text):
    lines = []
    fence = None
    for line in text.splitlines(keepends=True):
        if fence is not None:
            if re.fullmatch(r' {0,3}' + re.escape(fence[0]) + '{' + str(len(fence)) + r',}[ \t]*\r?\n?', line):
                fence = None
            continue
        opening = re.match(r' {0,3}(`{3,}|~{3,})([^\n]*)', line)
        if opening and (opening[1][0] != '`' or '`' not in opening[2]):
            fence = opening[1]
        else:
            lines.append(line)
    return re.sub(r'(?<!`)(`+)(?!`)(.*?)(?<!`)\1(?!`)', '', ''.join(lines), flags=re.S)


def text_fields(value, sources, expected=FIELDS):
    keys(value, expected, '文本映射')
    readable = {source['id'] for source in sources if source['status'] == 'read'}
    for field, text in value.items():
        nonempty(text, field)
        prose = prose_without_code(text)
        require(not re.search(r'\]\s*\(|!\[[^\]\n]*\]|^\s*\[[^\]]+\]:'
                              r'|</?[A-Za-z][\w:-]*(?:\s[^<>]*)?/?>'
                              r'|<[A-Za-z][A-Za-z0-9+.-]*:[^<>\s]*>|<[^\s<>@]+@[^\s<>@]+>'
                              r'|(?:https?://|file://)', prose, re.M),
                f'{field} 不可自造链接、HTML 或 URL；用单个 [S1] 引用来源')
        for reference in re.findall(r'\[([^\]\n]*S\d[^\]\n]*)\]', text):
            require(re.fullmatch(r'S[1-9][0-9]*', reference) is not None,
                    f'{field} 来源引用须为单个 [S1]，不支持合并引用')
            require(reference in readable, f'{field} 引用了未知或未读来源 {reference}')
    return value


def writing_plan(value, sources):
    """Check structure and source IDs, not the quality of the argument."""
    keys(value, {'main_question', 'main_answer', 'open_decision', 'sections'}, 'plan')
    for field in ('main_question', 'main_answer', 'open_decision'):
        nonempty(value[field], f'plan.{field}')
    require(isinstance(value['sections'], list) and value['sections'], 'plan.sections 必须是非空列表')
    readable = {source['id'] for source in sources if source['status'] == 'read'}
    titles = set()
    for section in value['sections']:
        keys(section, {'title', 'question', 'claim', 'why_next', 'source_ids'}, 'plan.section')
        for field in ('title', 'question', 'claim', 'why_next'):
            nonempty(section[field], f'plan.section.{field}')
        title = section['title'].strip()
        require(title not in titles, 'plan.sections 标题必须唯一')
        titles.add(title)
        require(isinstance(section['source_ids'], list), 'plan.section.source_ids 必须是列表')
        require(all(isinstance(item, str) and item in readable for item in section['source_ids']),
                'plan.section 引用了未知或未读来源')
    return value


def current_format(base):
    require(isinstance(base, dict), 'skeleton 必须是对象')
    constraints = base.get('constraints')
    require(isinstance(constraints, dict) and type(constraints.get('format_version')) is int
            and constraints['format_version'] == 2,
            '旧版或不支持的运行格式；当前要求 format_version=2。旧文件已保留，请使用新 --dir 重新 prepare，不执行迁移')


def skeleton(value, output):
    return {**copy.deepcopy(value), 'output': str(output), 'constraints': {'max_revisions': 1, 'format_version': 2},
            'text': {'report': None}}


def candidate(base, text, revision, plan):
    result = {**copy.deepcopy(base), 'plan': copy.deepcopy(writing_plan(plan, base['sources']))}
    result.pop('review_token', None)
    result['text'] = text_fields(text, base['sources'])
    result['revision'] = revision
    result['review_token'] = digest(encoded(result))
    return result


def critique(value, doc):
    keys(value, {'review_token', 'verdict', 'reason', 'reader_answers', 'findings', 'logic_review'}, 'critique')
    require(value['review_token'] == doc['review_token'], '评审未绑定当前稿件 review_token')
    require(value['verdict'] in ('advance', 'revise', 'abandon'), '无效 verdict')
    nonempty(value['reason'], 'critique.reason')
    # Presence is mechanical; the host and critic must assess the reasoning itself.
    nonempty(value['logic_review'], 'critique.logic_review')
    require(isinstance(value['reader_answers'], list) and len(value['reader_answers']) == 6,
            'reader_answers 必须恰好包含六个非空字符串')
    for answer in value['reader_answers']:
        nonempty(answer, 'reader_answers')
    require(isinstance(value['findings'], list), 'findings 必须是列表')
    require((value['verdict'] == 'advance') == (len(value['findings']) == 0),
            'advance 当且仅当 findings 为空')
    ids = {source['id'] for source in doc['sources']}
    for finding in value['findings']:
        keys(finding, {'scope', 'reason', 'source_ids'}, 'finding')
        require(isinstance(finding['scope'], str) and finding['scope'] in FIELDS | {'sources'}, '无效 finding.scope')
        require(finding['scope'] != 'sources' or value['verdict'] == 'abandon', '来源问题必须 abandon')
        nonempty(finding['reason'], 'finding.reason')
        require(isinstance(finding['source_ids'], list), 'source_ids 必须是列表')
        require(all(isinstance(item, str) and item in ids for item in finding['source_ids']), 'finding 引用了未知来源')
    return value


def atomic_new(path, text):
    """Hard-link publication is atomic and never replaces even a broken symlink."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.pending-', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_new(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def literal(text):
    fence = '`' * max(3, 1 + max((len(match) for match in re.findall(r'`+', text)), default=0))
    return f'{fence}text\n{text}\n{fence}'


def label(text):
    return re.sub(r'([\\`*{}\[\]()<>#!|_])', r'\\\1', text.replace('\n', ' ').replace('\r', ' '))


def markdown(doc):
    parts = [f"# {label(doc['project_name'])} · 项目交接", f"生成时间：{label(doc['generated_at'])}  \n版本：{label(doc['version'])}"]
    if doc['synthetic']:
        parts.append('**合成演练：本文件采用虚构证据，不代表真实项目状态。**')
    parts.append(doc['text']['report'])
    parts.extend(['## 来源与取证范围', '以下为宿主提供并冻结的证据快照；程序仅检查格式、一致性和引用，不验证来源真伪。',
                  '### 项目根目录', literal(doc['project_root']), '### 原始请求与授权约束', literal(doc['request']),
                  '### 检索覆盖与限制', literal(doc['coverage'])])
    for source in doc['sources']:
        parts.append(f"### [{source['id']}] · {source['kind']} · {source['status']}")
        locator = source['locator']
        if source['kind'] in ('file', 'web'):
            target = quote(locator, safe='/:' if source['kind'] == 'file' else '/:?=&%+#@')
            parts.append(f'[打开来源](<{target}>)')
        parts.extend(['定位：', literal(locator), '位置：', literal(source['location']),
                      '归属依据、适用范围与读取深度：', literal(source['scope']),
                      '证据摘录（纯文本）：', literal(source['excerpt'] or '（未读取或不可得；无证据摘录）')])
    return '\n\n'.join(parts) + '\n'


def command(step, directory):
    return {'kind': 'command', 'step': step,
            'argv': [str(Path(sys.executable).resolve()), str(ROOT / 'scripts/run.py'), step, '--dir', str(directory)]}


def llm(step, directory, inputs, output):
    role = 'critic' if step.startswith('critic') else step
    prompt = ROOT / 'references' / f'{role}.md'
    require(prompt.is_file(), f'缺少角色提示词文件：{prompt}')
    return {'kind': 'llm_subagent', 'step': step, 'prompt': str(prompt),
            'inputs': [str(directory / name) for name in inputs], 'output': str(directory / output),
            'notes': '由宿主派发无会话历史的新子 Agent；只传本 action 的 prompt、inputs、output。'
                     '仅写 output，校验 JSON 后原子排他提交；程序本身不调用 Agent。'}


def output_path(value, directory):
    path = absolute(value, 'output')
    # Resolve the parent only: the requested name may already be a broken symlink.
    path = path.parent.resolve() / path.name
    require(not path.is_relative_to(directory), '输出目标不得位于运行目录内或覆盖运行产物')
    return path


def inspect_run(directory):
    path = lambda name: directory / name
    exists = lambda name: os.path.lexists(path(name))
    require(exists('input.json') and exists('skeleton.json'), '运行未准备或快照缺失；请在新目录执行 prepare --input ABS --dir ABS')
    for name in ARTIFACTS:
        require(not exists(name) or (path(name).is_file() and not path(name).is_symlink()), f'{name} 必须为普通文件')
    base = read_json(path('skeleton.json'))
    current_format(base)
    dependencies = {
        'plan.json': 'skeleton.json', 'fill_map.json': 'plan.json', 'draft.json': 'fill_map.json', 'draft.md': 'draft.json',
        'critique-0.json': 'draft.md', 'patch.json': 'critique-0.json', 'revised.json': 'patch.json',
        'revised.md': 'revised.json', 'critique-1.json': 'revised.md', 'publication.json': 'critique-0.json',
    }
    for later, earlier in dependencies.items():
        require(not exists(later) or exists(earlier), f'{later} 缺少前置产物 {earlier}')
    original = source_input(read_json(path('input.json')))
    require('output' in base, 'skeleton 缺少 output')
    output = output_path(base['output'], directory)
    require(encoded(base) == encoded(skeleton(original, output)), 'skeleton 与冻结输入或机械字段不一致')
    if not exists('plan.json'):
        return llm('planner', directory, ['skeleton.json'], 'plan.json'), None
    plan = writing_plan(read_json(path('plan.json')), original['sources'])
    if not exists('fill_map.json'):
        return llm('writer', directory, ['skeleton.json', 'plan.json'], 'fill_map.json'), None
    expected = candidate(base, read_json(path('fill_map.json')), 0, plan)
    if not exists('draft.json'):
        return command('assemble', directory), expected
    draft = read_json(path('draft.json'))
    require(encoded(draft) == encoded(expected), 'draft 与冻结输入、plan 及 fill_map 不一致')
    require(exists('draft.md') and path('draft.md').read_bytes() == markdown(draft).encode('utf-8'), 'draft.md 缺失、损坏或被改写')
    if not exists('critique-0.json'):
        return llm('critic-0', directory, ['input.json', 'draft.json', 'draft.md'], 'critique-0.json'), None
    review = critique(read_json(path('critique-0.json')), draft)
    doc = draft
    if review['verdict'] == 'revise':
        if not exists('patch.json'):
            require(not exists('publication.json'), '修订未完成却存在发布记录')
            return llm('reviser', directory, ['input.json', 'draft.json', 'draft.md', 'critique-0.json'], 'patch.json'), None
        allowed = {finding['scope'] for finding in review['findings']}
        patch = text_fields(read_json(path('patch.json')), original['sources'], allowed)
        require(all(patch[field] != draft['text'][field] for field in allowed), '每个 patch 字段都必须真实改动')
        expected = candidate(base, {**draft['text'], **patch}, 1, plan)
        if not exists('revised.json'):
            require(not exists('publication.json'), '修订未合并却存在发布记录')
            return command('merge', directory), expected
        doc = read_json(path('revised.json'))
        require(encoded(doc) == encoded(expected), 'revised 与白名单 patch 合并结果不一致')
        require(exists('revised.md') and path('revised.md').read_bytes() == markdown(doc).encode('utf-8'), 'revised.md 缺失、损坏或被改写')
        if not exists('critique-1.json'):
            require(not exists('publication.json'), '修订稿未再审却存在发布记录')
            return llm('critic-1', directory, ['input.json', 'revised.json', 'revised.md'], 'critique-1.json'), None
        review = critique(read_json(path('critique-1.json')), doc)
        require(review['verdict'] == 'advance', '一次修订已用完，再审未通过；阻塞，禁止强行发布')
    else:
        require(not any(exists(name) for name in ('patch.json', 'revised.json', 'revised.md', 'critique-1.json')), '当前 verdict 不允许修订产物')
    require(review['verdict'] == 'advance', f"评审为 {review['verdict']}，不能发布")
    if not exists('publication.json'):
        return command('render', directory), doc
    record = read_json(path('publication.json'))
    keys(record, {'output', 'sha256', 'review_token'}, 'publication')
    final = output_path(record['output'], directory)
    collision_pattern = re.escape(output.stem) + r'\.\d{8}T\d{12}Z-\d+' + re.escape(output.suffix)
    require(final.parent == output.parent and (final == output or re.fullmatch(collision_pattern, final.name)), '发布记录路径不属于指定输出目标')
    require(record['review_token'] == doc['review_token'] and record['sha256'] == digest(markdown(doc)), '发布记录未绑定当前通过评审的稿件')
    require(final.is_file() and not final.is_symlink() and final.read_bytes() == markdown(doc).encode('utf-8'), '最终文件缺失、损坏或与当前稿件不一致')
    return {'kind': 'done', 'step': 'done', 'output': str(final), 'notes': '当前稿件及发布文件完全一致；绑定评审为 advance，独立派发由宿主核实。'}, doc


def prepare(input_path, directory, requested_output):
    value = source_input(read_json(input_path))
    output = output_path(str(requested_output or Path(value['project_root']) / 'PROJECT_HANDOFF.md'), directory)
    if directory.exists() and any(directory.iterdir()):
        require((directory / 'input.json').is_file() and (directory / 'skeleton.json').is_file(), '运行目录非空或准备中断，请使用新目录')
        require(encoded(read_json(directory / 'input.json')) == encoded(value), '输入与已有运行不同，请使用新 --dir')
        base = read_json(directory / 'skeleton.json')
        current_format(base)
        require(encoded(base) == encoded(skeleton(value, output)), '输出目标或冻结快照与已有运行不同')
        return
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'input.json', value)
    write_json(directory / 'skeleton.json', skeleton(value, output))


def publish(directory, doc):
    from datetime import timezone
    output = output_path(doc['output'], directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    index = 0
    while True:
        final = output if index == 0 else output.with_name(f'{output.stem}.{stamp}-{index}{output.suffix}')
        try:
            atomic_new(final, markdown(doc))
            break
        except FileExistsError:
            index += 1
    # A crash between these two exclusive writes can leave an unrecorded final file.
    # ponytail: no multi-file transaction; use a new run after an interrupted publication.
    write_json(directory / 'publication.json', {'output': str(final), 'sha256': digest(markdown(doc)), 'review_token': doc['review_token']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='operation', required=True)
    for name in ('prepare', 'next', 'assemble', 'merge', 'render'):
        sub = subs.add_parser(name)
        sub.add_argument('--dir', required=True)
        if name == 'prepare':
            sub.add_argument('--input', required=True)
            sub.add_argument('--output')
    args = parser.parse_args()
    try:
        directory = absolute(args.dir, '--dir').resolve()
        if args.operation == 'prepare':
            input_path = absolute(args.input, '--input')
            prepare(input_path, directory, args.output)
        elif args.operation != 'next':
            action, doc = inspect_run(directory)
            if args.operation == 'render' and action['kind'] == 'done':
                pass
            else:
                require(action['kind'] == 'command' and action['step'] == args.operation,
                        f"当前应执行 {action['step']}，不可执行 {args.operation}")
                if args.operation in ('assemble', 'merge'):
                    stem = 'draft' if args.operation == 'assemble' else 'revised'
                    write_json(directory / f'{stem}.json', doc)
                    atomic_new(directory / f'{stem}.md', markdown(doc))
                else:
                    publish(directory, doc)
        action, _ = inspect_run(directory)
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        action = {'kind': 'blocked', 'step': 'validate', 'notes': str(error)}
    print(json.dumps(action, ensure_ascii=False, indent=2))
    return 2 if action['kind'] == 'blocked' else 0


if __name__ == '__main__':
    sys.exit(main())
