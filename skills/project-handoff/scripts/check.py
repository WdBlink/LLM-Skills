#!/usr/bin/env python3
"""Deterministic coordinator checks; fixtures are synthetic, not actual LLM reviews."""
import copy
import json
import runpy
from pathlib import Path
import subprocess
import sys
import tempfile

SCRIPT = Path(__file__).with_name('run.py').resolve()
FIELDS = ('report',)
FILL = {'report': '## 已知事实\n\n合成检查资料。[S1] [S1]\n\n## 未知与接手动作\n\n需要核实尚未读取的记录。'}
PLAN = {'main_question': '目前可以确定什么，接手后应先做什么？',
        'main_answer': '先理解已知事实，再补齐未知。', 'open_decision': '应优先补充哪项来源？',
        'sections': [
            {'title': '已知事实', 'question': '目前知道什么？', 'claim': '有一条合成已读记录。',
             'why_next': '界定已知后才能定位未知。', 'source_ids': ['S1']},
            {'title': '未知与接手动作', 'question': '下一步是什么？', 'claim': '未知项需要核实。',
             'why_next': '以待决策事项结束，交给接手人。', 'source_ids': []}]}


def put(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def call(directory, operation='next', *args, kind=None, step=None):
    result = subprocess.run([sys.executable, str(SCRIPT), operation, '--dir', str(directory), *map(str, args)],
                            text=True, capture_output=True)
    assert not result.stderr, result.stderr
    action = json.loads(result.stdout)
    assert result.returncode == (2 if action['kind'] == 'blocked' else 0), result
    assert kind is None or action['kind'] == kind, action
    assert step is None or action['step'] == step, action
    if action['kind'] == 'llm_subagent':
        assert Path(action['prompt']).is_absolute() and Path(action['prompt']).is_file()
        assert Path(action['output']).is_absolute()
        assert all(Path(item).is_absolute() and Path(item).is_file() for item in action['inputs'])
    if action['kind'] == 'command':
        assert all(Path(action['argv'][index]).is_absolute() for index in (0, 1, 4))
    return action


def snapshot(directory):
    return {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.rglob('*') if p.is_file()}


def fixture(parent):
    return {'project_name': '合成演练', 'project_root': str(parent / 'project'),
            'request': '整理项目交接；不得声称这些检查是真实 Agent 评审。',
            'generated_at': '2026-09-14T12:00:00+08:00', 'version': '1.0',
            'coverage': '仅合成记录；总会话数未知；读取深度为摘录。', 'synthetic': True,
            'sources': [
                {'id': 'S1', 'kind': 'file', 'locator': str(parent / 'a (draft)#%.md'), 'location': '行 1–2',
                 'status': 'read', 'scope': '合成归属记录', 'excerpt': '测试事实。\n```\n[陷阱](https://example.com)\n```'},
                {'id': 'S2', 'kind': 'session', 'locator': 'session:test', 'location': '消息未知',
                 'status': 'unavailable', 'scope': '不可访问，归属待核实', 'excerpt': ''}]}


def fresh(parent, name, output=None):
    directory = parent / name
    source = parent / f'{name}-source.json'
    put(source, fixture(parent))
    call(directory, kind='blocked')
    assert not directory.exists(), 'next must not create directory'
    options = ['--input', source]
    if output is not None:
        options += ['--output', output]
    call(directory, 'prepare', *options, kind='llm_subagent', step='planner')
    base = read(directory / 'skeleton.json')
    assert base['text'] == {'report': None}
    assert base['constraints'] == {'max_revisions': 1, 'format_version': 2}
    assert 'plan' not in base
    call(directory, 'assemble', kind='blocked')
    put(directory / 'plan.json', PLAN)
    call(directory, kind='llm_subagent', step='writer')
    put(directory / 'fill_map.json', FILL)
    call(directory, kind='command', step='assemble')
    call(directory, 'assemble', kind='llm_subagent', step='critic-0')
    return directory, source


def review(directory, verdict, revision=0):
    doc = read(directory / ('revised.json' if revision else 'draft.json'))
    return {'review_token': doc['review_token'], 'verdict': verdict, 'reason': '合成评审，仅测试编排。',
            'logic_review': '合成结构检查说明；不证明真实写作逻辑。',
            'reader_answers': [f'合成读者回答 {number}' for number in range(6)],
            'findings': [] if verdict == 'advance' else [
                {'scope': 'report', 'reason': '需要更清楚说明未知项。', 'source_ids': []}]}


def main():
    checks = []
    with tempfile.TemporaryDirectory(prefix='project-handoff-check-') as tmp:
        parent = Path(tmp)
        d, source = fresh(parent, 'normal')
        before = snapshot(d)
        call(d)
        assert before == snapshot(d), 'next must be read-only'
        put(d / 'critique-0.json', review(d, 'advance'))
        call(d, kind='command', step='render')
        done = call(d, 'render', kind='done')
        output = Path(done['output'])
        assert output.read_bytes() == (d / 'draft.md').read_bytes()
        assert '%20%28draft%29%23%25.md' in output.read_text()
        assert '````text\n测试事实。' in output.read_text(), 'excerpt must escape nested fences'
        before = snapshot(parent)
        call(d, 'render', kind='done')
        call(d, kind='done')
        assert snapshot(parent) == before, 'repeat render must not republish'
        call(d, 'assemble', kind='blocked')
        output.write_text('tampered')
        call(d, kind='blocked')
        checks.append('normal routing, read-only next, absolute actions, exact publication, idempotence, pure-text excerpts')

        d, source = fresh(parent, 'plan-validation')
        good_plan = read(d / 'plan.json')
        invalid = []
        for field, value in (('main_question', ''), ('main_answer', ''), ('open_decision', ''), ('sections', [])):
            item = copy.deepcopy(good_plan)
            item[field] = value
            invalid.append(item)
        for field, value in (('source_ids', ['S99']), ('source_ids', ['S2']), ('source_ids', 'S1'),
                             ('title', ''), ('question', ''), ('claim', ''), ('why_next', '')):
            item = copy.deepcopy(good_plan)
            item['sections'][0][field] = value
            invalid.append(item)
        item = copy.deepcopy(good_plan)
        item['sections'][1]['title'] = item['sections'][0]['title']
        invalid.append(item)
        item = copy.deepcopy(good_plan)
        del item['main_question']
        invalid.append(item)
        for item in invalid:
            put(d / 'plan.json', item)
            call(d, 'assemble', kind='blocked')
        (d / 'plan.json').write_text('{"sections":[],"sections":[]}')
        call(d, kind='blocked')
        (d / 'plan.json').unlink()
        call(d, 'assemble', kind='blocked')
        put(d / 'plan.json', good_plan)
        put(d / 'critique-0.json', review(d, 'advance'))
        changed_plan = copy.deepcopy(good_plan)
        changed_plan['main_answer'] += ' 调整写作方案。'
        put(d / 'plan.json', changed_plan)
        call(d, kind='blocked')
        runtime = runpy.run_path(str(SCRIPT))
        newer = runtime['candidate'](read(d / 'skeleton.json'), FILL, 0, changed_plan)
        assert newer['review_token'] != read(d / 'draft.json')['review_token']
        put(d / 'draft.json', newer)
        (d / 'draft.md').write_text(runtime['markdown'](newer), encoding='utf-8')
        action = call(d, kind='blocked')
        assert 'review_token' in action['notes'], action
        checks.append('planner precedes writer; strict plan schema/readable IDs; missing plan blocks assembly; plan-bound tokens reject old reviews')

        d, source = fresh(parent, 'old-format')
        put(d / 'critique-0.json', review(d, 'advance'))
        call(d, 'render', kind='done')
        old_base = read(d / 'skeleton.json')
        del old_base['constraints']['format_version']
        old_base['text'] = {'overview': None, 'approach': None}
        put(d / 'skeleton.json', old_base)
        before = snapshot(d)
        for operation, args in (('next', []), ('render', []), ('prepare', ['--input', source])):
            action = call(d, operation, *args, kind='blocked')
            assert 'format_version=2' in action['notes'], action
        assert before == snapshot(d), 'old runs must remain unchanged'
        checks.append('old format including a published run is clearly blocked and preserved, never migrated')


        d, source = fresh(parent, 'input-change')
        before = snapshot(d)
        changed = read(source)
        changed['request'] += ' 新目标'
        put(source, changed)
        call(d, step='critic-0')
        call(d, 'prepare', '--input', source, kind='blocked')
        assert before == snapshot(d), 'external input must not mutate snapshot'
        put(d / 'input.json', changed)
        call(d, kind='blocked')
        checks.append('external input changes do not change frozen run; changed prepare/snapshot rejected')

        d, _ = fresh(parent, 'revision')
        put(d / 'critique-0.json', review(d, 'revise'))
        call(d, kind='llm_subagent', step='reviser')
        for patch in ({'sources': 'changed'}, {'overview': 'unrequested'}, {'report': FILL['report']},
                      {'report': 'unknown [S99]'}, {'report': ''},
                      {'report': 'new', 'output': '/tmp/wrong.md'}):
            put(d / 'patch.json', patch)
            call(d, 'merge', kind='blocked')
            assert not (d / 'revised.json').exists()
        put(d / 'patch.json', {'report': '## 先明确接手动作\n\n需先核实未知。\n\n## 动作的依据\n\n合成已读记录支持安排核查。[S1]'})
        before = snapshot(d)
        call(d, 'merge', kind='llm_subagent', step='critic-1')
        assert all(Path(name).read_bytes() == state[0] for name, state in before.items())
        draft, revised = read(d / 'draft.json'), read(d / 'revised.json')
        assert draft['review_token'] != revised['review_token']
        assert revised['text']['report'].startswith('## 先明确接手动作')
        assert draft['plan'] == revised['plan'] == PLAN
        assert draft['sources'] == revised['sources']
        assert draft['request'] == revised['request']
        assert all(draft[key] == revised[key] for key in draft if key not in ('text', 'revision', 'review_token'))
        put(d / 'critique-1.json', review(d, 'advance'))
        call(d, kind='blocked')
        for verdict in ('revise', 'abandon'):
            put(d / 'critique-1.json', review(d, verdict, 1))
            call(d, kind='blocked')
            call(d, 'render', kind='blocked')
            assert not (d / 'publication.json').exists()
        put(d / 'critique-1.json', review(d, 'advance', 1))
        call(d, 'render', kind='done')
        checks.append('whole-report reordering, patch whitelist/real changes, immutable plan/facts, new critic, stale review rejection, one revision ceiling')

        d, _ = fresh(parent, 'refs')
        for field in FIELDS:
            for invalid in ('未知 [S99]', '未读 [S2]', '合并 [S1,S2]', '[链接](https://example.com)', '空编号 [S0]'):
                put(d / 'fill_map.json', {**FILL, field: invalid})
                call(d, kind='blocked')
        for payload in ('{"report":"one","report":"two"}', 'NaN', '[]', '{'):
            (d / 'fill_map.json').write_text(payload)
            call(d, kind='blocked')
        checks.append('body source IDs independently checked, unknown/unread/grouped citations and invented links rejected; strict JSON')

        validate_text = runpy.run_path(str(SCRIPT))['text_fields']
        sources = fixture(parent)['sources']
        for allowed in ('a < b 和 c > d；x < 3。',
                        '`GET https://example.com?q=a<b` 与 ``<div> / [x](url)``',
                        '```python\nurl = "https://example.com"\nhtml = "<div>"\n```\n比较 a < b 和 c > d。',
                        '~~~~text\n[示例](https://example.com)\n~~~~~\n比较 a < b。'):
            validate_text({**FILL, 'report': allowed}, sources)
        for forbidden in ('[链接](/absolute/path)', '![图像](/image.png)', '<https://example.com>',
                          '<user@example.com>', '<div class="x">正文</div>', '<script>alert(1)</script>',
                          '`short code` [链接](/real)', '```text\nexample\n```\n<a href="/real">link</a>'):
            try:
                validate_text({**FILL, 'report': forbidden}, sources)
            except ValueError:
                pass
            else:
                raise AssertionError(f'active markup accepted: {forbidden}')
        checks.append('plain comparisons and code examples allowed; active links, images, autolinks and HTML rejected')


        d, _ = fresh(parent, 'review-validation')
        good = review(d, 'revise')
        invalid = []
        for field, value in (('reader_answers', ['short']), ('reader_answers', [''] * 6), ('verdict', 'advance'),
                             ('review_token', 'old'), ('findings', []), ('reason', ''), ('logic_review', '')):
            item = copy.deepcopy(good)
            item[field] = value
            invalid.append(item)
        for field, value in (('scope', 'sources'), ('source_ids', ['S99']), ('scope', 'output')):
            item = copy.deepcopy(good)
            item['findings'][0][field] = value
            invalid.append(item)
        missing_logic = copy.deepcopy(good)
        del missing_logic['logic_review']
        invalid.append(missing_logic)
        for item in invalid:
            put(d / 'critique-0.json', item)
            call(d, kind='blocked')
        abandoned = review(d, 'abandon')
        abandoned['findings'][0]['scope'] = 'sources'
        put(d / 'critique-0.json', abandoned)
        call(d, kind='blocked')
        checks.append('six reader answers and logic_review presence, verdict/findings contract, source-scope abandon and finding IDs checked')

        for name in ('input.json', 'skeleton.json', 'plan.json', 'fill_map.json', 'draft.json', 'draft.md', 'critique-0.json',
                     'patch.json', 'revised.json', 'revised.md', 'critique-1.json', 'publication.json'):
            d, _ = fresh(parent, 'damage-' + name)
            put(d / 'critique-0.json', review(d, 'revise'))
            put(d / 'patch.json', {'report': '已改动，仍有未知。[S1]'})
            call(d, 'merge', step='critic-1')
            put(d / 'critique-1.json', review(d, 'advance', 1))
            call(d, 'render', kind='done')
            saved = (d / name).read_bytes()
            (d / name).write_text('{broken')
            call(d, kind='blocked')
            (d / name).write_bytes(saved)
            if name != 'publication.json':
                (d / name).unlink()
                call(d, kind='blocked')
        d, _ = fresh(parent, 'preview-bytes')
        (d / 'draft.md').write_bytes((d / 'draft.md').read_bytes().replace(b'\n', b'\r\n'))
        call(d, kind='blocked')
        checks.append('missing, orphaned or damaged artifacts blocked at every stage; previews checked byte for byte')

        for mode in ('existing', 'symlink'):
            target = parent / f'{mode}.md'
            if mode == 'existing':
                target.write_text('KEEP')
            else:
                target.symlink_to(parent / 'missing-target')
            d, _ = fresh(parent, mode, target)
            put(d / 'critique-0.json', review(d, 'advance'))
            done = call(d, 'render', kind='done')
            assert Path(done['output']) != target
            assert target.read_text() == 'KEEP' if mode == 'existing' else target.is_symlink()
        checks.append('existing outputs and broken symlinks preserved with timestamp/sequence alternate paths')

        valid = fixture(parent)
        invalid = []
        for field, value in (('project_root', 'relative'), ('synthetic', 1), ('generated_at', '2026-09-14T12:00:00'), ('sources', [])):
            item = copy.deepcopy(valid)
            item[field] = value
            invalid.append(item)
        for field, value in (('id', 'S0'), ('locator', 'relative'), ('excerpt', ''), ('kind', 'other')):
            item = copy.deepcopy(valid)
            item['sources'][0][field] = value
            invalid.append(item)
        item = copy.deepcopy(valid)
        item['sources'][1]['excerpt'] = 'not actually read'
        invalid.append(item)
        item = copy.deepcopy(valid)
        item['sources'][1]['id'] = 'S1'
        invalid.append(item)
        item = copy.deepcopy(valid)
        item['sources'][0].update(kind='web', locator='https://user:password@example.com/source')
        invalid.append(item)
        for index, item in enumerate(invalid):
            input_path = parent / f'invalid-{index}.json'
            put(input_path, item)
            directory = parent / f'invalid-run-{index}'
            call(directory, 'prepare', '--input', input_path, kind='blocked')
            assert not directory.exists()
        unknown = copy.deepcopy(valid)
        unknown['sources'] = [unknown['sources'][1]]
        input_path = parent / 'zero-read.json'
        put(input_path, unknown)
        d = parent / 'zero-read'
        call(d, 'prepare', '--input', input_path, kind='llm_subagent')
        unknown_plan = copy.deepcopy(PLAN)
        for section in unknown_plan['sections']:
            section['source_ids'] = []
        put(d / 'plan.json', unknown_plan)
        call(d, step='writer')
        put(d / 'fill_map.json', {field: '证据不足，待核实。' for field in FIELDS})
        call(d, 'assemble', step='critic-0')
        d = parent / 'forbidden-output'
        call(d, 'prepare', '--input', input_path, '--output', d / 'draft.md', kind='blocked')
        call(parent / 'relative-args', 'prepare', '--input', 'relative.json', kind='blocked')
        checks.append('input schema, path, time, source IDs/read-status independently validated; zero-read sources supported; run outputs protected')
    print(json.dumps({'status': 'passed', 'checks': checks,
                      'llm_execution': 'not tested; all writer/reviewer artifacts are synthetic test doubles'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
