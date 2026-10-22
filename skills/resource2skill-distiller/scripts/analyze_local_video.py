"""Extract explicit timestamped frames and optional transcript; analyze with Codex."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, urlencode, urlsplit

from analyze_resource import common_arguments, distiller_prompt
from codex_runner import run_codex


def source_url(value):
    url = urlsplit(value)
    if url.scheme != 'https' or url.username or url.password or url.port:
        raise ValueError('Use a public HTTPS video source without credentials or a port.')
    query = parse_qs(url.query)
    if url.hostname in {'www.bilibili.com', 'bilibili.com'}:
        match = re.fullmatch(r'/video/(BV[A-Za-z0-9]{10})/?', url.path)
        page = query.get('p', ['1'])
        if match and len(page) == 1 and re.fullmatch(r'[1-9][0-9]*', page[0]):
            return f'https://www.bilibili.com/video/{match[1]}/?{urlencode({"p": page[0]})}'
    elif url.hostname in {'www.youtube.com', 'youtube.com', 'm.youtube.com', 'youtu.be'}:
        video_id = url.path.strip('/') if url.hostname == 'youtu.be' else query.get('v', [''])[0]
        if (url.hostname == 'youtu.be' or url.path == '/watch') and re.fullmatch(r'[A-Za-z0-9_-]{11}', video_id):
            return f'https://www.youtube.com/watch?v={video_id}'
    raise ValueError('Use a full Bilibili BV or YouTube watch URL.')


def sample_times(duration, interval, offset, limit):
    if not all(math.isfinite(n) for n in (duration, interval, offset)) or duration <= 0 or interval <= 0 or offset < 0:
        raise ValueError('Duration and interval must be positive and source offset nonnegative, all finite.')
    if not 1 <= limit <= 100:
        raise ValueError('Frame limit must be between 1 and 100.')
    count = math.ceil(duration / interval)
    if count > limit:
        raise ValueError(f'{count} frames exceed limit {limit}; split the video or explicitly change sampling.')
    return [i * interval for i in range(count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common_arguments(parser)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--source-offset', type=float, default=0)
    parser.add_argument('--frame-interval', type=float, required=True, help='Explicit seconds between sampled frames; never silently downsample.')
    parser.add_argument('--max-frames', type=int, default=48)
    parser.add_argument('--transcript', type=Path, help='Existing UTF-8 subtitle/transcript file; no audio transcription is performed.')
    args = parser.parse_args()
    video = args.video.expanduser().resolve()
    if not video.is_file() or video.stat().st_size == 0 or args.output_dir.expanduser().exists() or not args.goal.strip():
        parser.error('Provide a nonempty local video, goal and a new output directory.')
    try:
        url = source_url(args.source_url)
        probe = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(video)],
                                         check=True, capture_output=True, text=True, timeout=30).stdout)
        if not any(s.get('codec_type') == 'video' for s in probe.get('streams', [])):
            raise ValueError('No video stream.')
        duration = float(probe.get('format', {}).get('duration', 0))
        times = sample_times(duration, args.frame_interval, args.source_offset, args.max_frames)
    except (ValueError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    transcript = args.transcript.expanduser().read_text() if args.transcript else ''
    if len(transcript.encode()) > 1_500_000:
        parser.error('Split transcripts over 1.5 MB first.')
    prompt = distiller_prompt(args.repo, args.domain, args.prompt_file)
    timeline = [{'image': f'image-{i:03}.jpg', 'clip_seconds': t, 'source_seconds': args.source_offset + t} for i, t in enumerate(times)]
    metadata = {'url': url, 'local_video': str(video), 'duration_seconds': duration, 'source_offset_seconds': args.source_offset,
                'frame_interval_seconds': args.frame_interval, 'frames': timeline, 'goal': args.goal, 'domain': args.domain,
                'transcript': str(args.transcript) if args.transcript else None,
                'transcript_sha256': hashlib.sha256(transcript.encode()).hexdigest() if transcript else None,
                'audio_analyzed': False, 'coverage': 'sampled frames plus supplied transcript, if any',
                'created_at': datetime.now(timezone.utc).isoformat()}
    prompt += '\n\n提炼目标：' + args.goal + '\n来源：' + url
    prompt += '\n输入只有以下时刻的图片和已提供的文本，没有音轨。不得将抽帧分析称为看完整个视频；未采样操作和无法辨认的字幕保持未知。'
    prompt += '\n图片顺序与时间：\n' + json.dumps(timeline, ensure_ascii=False)
    prompt += '\n提供的字幕或转写（没有则为空）：\n' + transcript
    with TemporaryDirectory(prefix='resource2skill-frames-') as work:
        frames = []
        for i, t in enumerate(times):
            frame = Path(work) / f'frame-{i:03}.jpg'
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-ss', str(t), '-i', str(video),
                            '-frames:v', '1', '-vf', 'scale=min(1280\\,iw):-2', str(frame)],
                           check=True, capture_output=True, timeout=60)
            frames.append(frame)
        run_codex(prompt, args.output_dir, images=frames, model=args.model, timeout=args.timeout, metadata=metadata)
    print('Codex sampled-frame analysis completed; audio was not analyzed.')


if __name__ == '__main__':
    main()
