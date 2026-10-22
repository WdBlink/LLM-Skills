# 视频输入适配

当前路径为公开下载 → ffprobe 核验 → 明确间隔的图片抽样 → Codex CLI。分析输入为图片及可选的已取得转写，原视频保留在工作目录。

## 获取与核验

先核对 HTTPS YouTube 或 B 站 BV 链接，保留分 P、去除跟踪参数；短链接先解析到完整地址。只下载用户指定范围，禁止自动读取浏览器 Cookie。

```bash
mkdir -p "$RUN/media"
"$R2S/.venv/bin/python" -m yt_dlp \
  --ignore-config --no-playlist --retries 2 --fragment-retries 2 \
  --no-overwrites --write-info-json --write-subs --write-auto-subs \
  --sub-langs 'zh.*,en.*' --sub-format 'srt/vtt/best' \
  -f 'bv*[height<=720]+ba/b[height<=720]' --merge-output-format mp4 \
  -P "$RUN/media" -o '%(id)s.%(ext)s' \
  --print after_move:filepath -- "$VIDEO_URL"
```

变量由外层 Codex 根据真实输入设置并安全引用。取实际返回路径，不猜测后缀；逐个核对分 P 和时长。遇到登录、地区或会员限制时记录缺少的输入，不规避限制。

```bash
ffprobe -v error -show_format -show_streams -of json "$VIDEO_PATH"
```

## 抽帧分析

以下示例每 5 秒取一帧，只适用于任务判断允许该覆盖密度的片段；不是通用视频质量标准：

```bash
python3.11 "$SKILL_DIR/scripts/analyze_local_video.py" \
  --video "$VIDEO_PATH" --source-url "$VIDEO_URL" \
  --source-offset 0 --frame-interval 5 \
  --goal '提炼提供片段中可证实的操作方法' \
  --output-dir "$RUN/extraction"
```

有合法取得的字幕或转写时，加 `--transcript /absolute/path/transcript.txt`；核对它与片段时间、来源一致。可选 `--domain` 或 `--prompt-file`，模型参数沿用文本入口。

默认最多 48 帧，超限直接报错；需按实际任务分段或显式调整间隔，不静默漏掉素材。抽帧保留至多 1280 像素宽度，每张图的片段与原视频时间保存在 `source.json`。片段偏移用 `--source-offset`；切片时验证真正起点，流拷贝可能因关键帧而偏移。

AV1 等原始编码只需本地 ffmpeg 能解码，不再为服务端兼容强制全片 H.264 转码。当前没有 MiniMax 文件上传、1800 秒限制或远端文件清理步骤；限制由本地采样数量与 Codex 实际可接受的输入决定。

## 证据边界

抽样不能证明覆盖全部动态过程。短暂操作、文字变化或动效需要缩短片段和加密抽样，关键结论回原视频核对。画面字幕不足以承载谈话内容时补充可靠转写；本脚本不进行语音识别，不能把结果称作完整音视频理解。

原视频和完整元数据留在运行目录；候选包保留必要来源和证据，不附带临时媒体地址、账号信息或整段视频。
