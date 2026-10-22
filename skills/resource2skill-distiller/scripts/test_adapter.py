"""Offline checks for the Codex adapter; run separately from real model tests."""
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from analyze_local_video import sample_times, source_url
from analyze_resource import source_url as document_url
from codex_runner import completed_output, json_result, run_codex


def rejected(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except (ValueError, RuntimeError, FileExistsError):
        return
    raise AssertionError('Invalid input or incomplete result accepted')


def main():
    assert sample_times(30, 10, 600, 3) == [0, 10, 20]
    for args in [(30, 0, 0, 3), (30, 10, -1, 3), (31, 10, 0, 3), (float('nan'), 1, 0, 3)]:
        rejected(sample_times, *args)
    assert source_url('https://www.bilibili.com/video/BV1PHYE62Eqe/?p=2&spm_id_from=tracking').endswith('/?p=2')
    assert source_url('https://youtu.be/abcdefghijk?t=5') == 'https://www.youtube.com/watch?v=abcdefghijk'
    assert document_url('https://example.org/article') == 'https://example.org/article'
    for bad in ['http://example.org', 'https://key@example.org', 'file:///tmp/private']:
        rejected(document_url, bad)
    rejected(source_url, 'https://bilibili.com.evil.test/video/BV1PHYE62Eqe/')
    rejected(source_url, 'https://www.bilibili.com/video/BV1PHYE62Eqe/?p=0')
    events = '\n'.join(json.dumps(e) for e in [
        {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': '{"ok":true}'}},
        {'type': 'turn.completed', 'usage': {'input_tokens': 12, 'output_tokens': 4}},
    ])
    assert completed_output(events, '{"ok":true}')['usage']['output_tokens'] == 4
    for data, text in [('', 'answer'), (events, ''), (events, 'different'), (events + '\n{"type":"turn.failed"}', '{"ok":true}')]:
        rejected(completed_output, data, text)
    assert json_result('```json\n{"ok":true}\n```') == {'ok': True}
    rejected(json_result, 'unparsed text')
    rejected(json_result, 'null')
    with TemporaryDirectory() as tmp, patch('codex_runner.shutil.which', return_value='/usr/bin/codex'), patch('codex_runner.cli_settings', return_value={'model':'test-model'}):
        folder = Path(tmp)
        def launch(command, **kwargs):
            assert command[command.index('--sandbox') + 1] == 'read-only'
            assert '--ignore-user-config' in command and 'features.shell_tool=false' in command
            assert kwargs['start_new_session'] is True
            assert 'MINIMAX_API_KEY' not in kwargs['env'] and 'OPENAI_API_KEY' not in kwargs['env']
            assert command.index('-') < command.index('--image')
            assert '--output-schema' in command
            Path(command[command.index('--output-last-message') + 1]).write_text('{"ok":true}')
            return Mock(returncode=0, communicate=Mock(return_value=(events, '')))
        with patch.dict(os.environ, {'MINIMAX_API_KEY': 'fake-private', 'OPENAI_API_KEY': 'fake-private'}), patch('codex_runner.subprocess.Popen', side_effect=launch) as popen:
            attachment = folder / 'fixture.jpg'; attachment.write_bytes(b'image placeholder for command test')
            schema = {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}, 'required': ['ok'], 'additionalProperties': False}
            assert json_result(run_codex('Test supplied content', folder/'success', images=[attachment], schema=schema)) == {'ok': True}
            report = json.loads((folder/'success/validation.json').read_text())
            assert report['execution'] == 'passed' and report['actual_model'] is None
            rejected(run_codex, 'Do not overwrite', folder/'success')
            rejected(run_codex, '', folder/'empty')
            assert popen.call_count == 1
        process = Mock(pid=123, returncode=1, communicate=Mock(side_effect=[subprocess.TimeoutExpired('codex',1), ('','')]))
        with patch('codex_runner.subprocess.Popen', return_value=process), patch('codex_runner.os.killpg') as kill:
            rejected(run_codex, 'timeout fixture', folder/'timeout', timeout=1)
            kill.assert_called_once()
            assert json.loads((folder/'timeout/validation.json').read_text())['error_type'] == 'TimeoutExpired'
        process = Mock(returncode=2, communicate=Mock(return_value=('', 'CLI failed')))
        with patch('codex_runner.subprocess.Popen', return_value=process):
            rejected(run_codex, 'failure fixture', folder/'failed')
            assert json.loads((folder/'failed/validation.json').read_text())['execution'] == 'failed'
    print('PASS: source/sampling validation, isolated Codex invocation, credential exclusion, completion checks, timeout and no-overwrite handling.')


if __name__ == '__main__':
    main()
