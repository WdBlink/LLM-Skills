# 视频、转写与关键画面

## 环境与取材

检查 `yt-dlp`、`ffmpeg`、`ffprobe`、`whisper-cli`、`mlx_whisper` 及已有模型路径，也检查当前工作区的虚拟环境，不仅凭 PATH 缺失判断未安装。复用现有安装；缺少依赖时只安装本次会用的工具，不同时搭建多个 ASR 后端。首次获取权重说明模型和大致体积，按环境权限执行。Apple Silicon 可用 whisper.cpp 或 MLX；CPU/CUDA 环境已有 faster-whisper 时可直接复用。

下方命令中的 `VIDEO_URL`、`MEDIA_DIR`、`VIDEO_PATH`、`MODEL_PATH`、`FRAME_TIME`、`FRAME_PATH` 均先替换为已核验值。目录使用绝对路径，按来源单独建目录。通过 shell 执行时安全引用参数；不要把网页标题或字幕拼成可执行命令。命令示例按依赖关系分别执行并检查退出状态。

## 下载

先检查字幕和格式，再下载选中的单条视频；保留 B 站的 `p` 参数，不自动扩展到播放列表。默认选择不高于 1080p 的最佳视频及音频，以便辨认样片；需要更高分辨率时再调整。

```bash
yt-dlp --ignore-config --no-playlist --list-subs -- "$VIDEO_URL"
```

```bash
yt-dlp --ignore-config --no-playlist --retries 3 --fragment-retries 3 \
  --no-overwrites --write-info-json --write-subs --write-auto-subs \
  --sub-langs 'zh.*,en.*' --sub-format 'srt/vtt/best' \
  -f 'bv*[height<=1080]+ba/b[height<=1080]' \
  -P "$MEDIA_DIR" -o '%(extractor_key)s-%(id)s.%(ext)s' \
  --print after_move:filepath -- "$VIDEO_URL"
```

字幕语言按真实音轨补充，原始字幕不可覆盖。用 `after_move:filepath` 的实际结果设置 `VIDEO_PATH`，不要猜最终容器后缀。B 站分 P 若产生多个文件，逐一核对并处理，不能静默取第一个。记录实际时长、视频流、音频流与分辨率：

```bash
ffprobe -v error -show_format -show_streams -of json "$VIDEO_PATH"
```

YouTube 的完整支持可能需要 `yt-dlp-ejs` 与支持的 JavaScript runtime；按当前版本诊断，不把运行时依赖错误当作登录错误。遇到反爬、登录要求、地区限制时保存进度，依据错误采用一次有理由的修正；仍失败则接收 Downie/其他方式下载的本地文件并继续。读取浏览器 cookies 只在用户已授权使用登录态时采用，不输出或打包 cookies。默认不循环重试或自动切换到付费下载服务。

## 本地转写

如果已存在覆盖完整、语言正确、时间戳可靠的人工字幕，可直接用并注明来源；自动字幕先抽查，需要时用 ASR 校正。用户明确要求模型重新转写时执行 ASR。没有字幕才转写，不要求上游重跑。

```bash
ffmpeg -hide_banner -nostdin -n -i "$VIDEO_PATH" -map 0:a:0 \
  -vn -ar 16000 -ac 1 -c:a pcm_s16le "$MEDIA_DIR/audio.wav"
```

```bash
whisper-cli -m "$MODEL_PATH" -f "$MEDIA_DIR/audio.wav" \
  -l auto -otxt -osrt -oj -of "$MEDIA_DIR/transcript"
```

核验已安装版本的 `whisper-cli --help`。中文或中英混合内容用多语言模型，不能用 `.en` 权重；优先复用本地模型，新选择可从多语言 `small` 起步，质量不足再提高。保留原语言转写，中文整理另写，不使用翻译模式代替原稿。

若已有 MLX 格式模型，复用 `mlx-whisper`，不为切换后端重复下载权重。下面的路径和语言按实际输入设置；自动检测语言可省略 `language`：

```python
import mlx_whisper
from mlx_whisper.writers import get_writer
result = mlx_whisper.transcribe(audio_path, path_or_hf_repo=model_path,
                               language="zh", task="transcribe", word_timestamps=True)
assert result.get("segments") and result.get("text", "").strip(), "Empty transcript"
get_writer("all", output_dir)(result, "transcript")
```

检查输出内容及时间戳，不能只看进程退出码。Metal 不可用时先判断是否为运行环境限制；不要将其误判成模型缺失。日语人名、摄影术语及音乐处的幻觉文本应结合内嵌字幕核对；弹幕不是语音字幕。

音轨与截帧必须共享视频时间轴，不先删除静音再拿压缩后的时间戳定位截图。分段转写时加回每段偏移，并处理重叠重复。静音处、音乐段、品牌与人名是抽查重点；纠错保留原记录和理由。没有音轨时只做视觉笔记，标明无可转写语音。无时间戳文本不能伪造精确时间关联。

## 自动找图，而不是把每句字幕机械配图

先做低分辨率联系表，Agent 用视觉能力浏览；按解说提及的作品及视觉风格差异挑选，再提取原分辨率截图。联系表只是候选索引，不能当成精确色彩参考。

以下每约 5 秒选一帧。`showinfo` 放在 `tile` 前，日志中的 `n` 为零起始候选序号，`pts_time` 为视频时间；第 `n` 帧对应 `floor(n/16)+1` 号联系表、行 `floor((n%16)/4)+1`、列 `n%4+1`。用日志真实时间，不能按序号乘 5 猜时间。

```bash
ffmpeg -hide_banner -nostdin -n -i "$VIDEO_PATH" -map 0:v:0 -an \
  -vf "select='isnan(prev_selected_t)+gte(t-prev_selected_t,5)',showinfo,scale=320:-2,tile=4x4:padding=4:margin=4" \
  -fps_mode vfr -q:v 3 "$MEDIA_DIR/overview-%03d.jpg" \
  2> "$MEDIA_DIR/overview.log"
```

5 秒采样会漏短暂展示。对解说提及但未找到的样片，或明显快速切图的视频，用更短间隔或补充场景切换联系表：将选择表达式改为 `isnan(prev_selected_t)+gt(scene,0.30)`，输出另用 `scenes-%03d.jpg` 和 `scenes.log`。`0.30` 只是起点，场景检测也可能漏掉缓慢叠化和画中画；根据实际画面调整。长视频分段浏览并保留片段起点偏移，避免一次加载大量候选图。

核对清楚后用实际时间提取未调色的原始截图：

```bash
ffmpeg -hide_banner -nostdin -n -ss "$FRAME_TIME" -i "$VIDEO_PATH" \
  -map 0:v:0 -frames:v 1 -update 1 "$FRAME_PATH"
```

选稳定、清晰、作品完整的画面；遇到字幕遮挡或淡入淡出，在附近重取。为便于看图可以另存作品区域裁剪，但保留全帧与裁剪坐标。HDR/SDR 转换、视频压缩、翻拍书页会影响颜色；不把截图数值当原作的精确调色参数，不自动美化作为证据的图片。

对每张选图在图文稿记下：来源 ID、时间、相邻文字段、图片相对路径、作品归属或“不确定”、支持的观察。解说者仿拍、失败对照和摄影师作品不能混为一类。保留完整文字段，新增说明标成“画面观察”或“编者批注”。检查头、中、尾及跨段位置，确保音画未漂移；ASR 缺失则明确哪些段落只有图片。

## 工具依据

工具选项首次执行前以本地帮助核验；这些是实现依据，不是性能承诺。

- [yt-dlp 官方仓库与使用说明](https://github.com/yt-dlp/yt-dlp)
- [whisper.cpp 官方仓库](https://github.com/ggml-org/whisper.cpp) 与 [CLI 参数](https://github.com/ggml-org/whisper.cpp/tree/master/examples/cli)
- [faster-whisper 官方仓库](https://github.com/SYSTRAN/faster-whisper)：复用该后端时读取；迭代完 `segments` 才实际完成转写，并保存 `start`、`end`、`text`。
- [MLX Whisper 官方实现](https://github.com/ml-explore/mlx-examples/tree/main/whisper)：复用本地 MLX 权重时读取。
- [FFmpeg filters](https://ffmpeg.org/ffmpeg-filters.html)：`select`、`showinfo`、`scale`、`tile`。
