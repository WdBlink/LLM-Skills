# 将完成的摄影提示词资产归档到 GitHub

## 目标与触发

从当前用户指定的远端与本地目录，或已有明确的项目配置确定归档目标。核对已有克隆的 Git remote，不能从技能示例、摄影师姓名或其他用户的历史配置推定个人资产库。目标缺失时先整理本地资产，再询问缺少的仓库位置；已有有效配置无需重复确认。

当用户要求上传、归档、版本化或维护该仓库，或当前任务已有相应授权时执行。已有授权继续沿用；仅要求研究、生成提示词或修改技能时，不附带远端写入。上传现成资产直接从本阶段开始，不重新下载、转写或改写提示词。

先核对实际账户、仓库远端和私有状态；目标非私有或与指定账户不符时先解决具体差异，不直接上传。仓库已存在则复用。只有确实不存在且创建已在授权范围内时才新建私有仓库。

## 统一目录

```text
<archive-repo>/
├── README.md                       # 所有风格的入口
└── <style-slug>/
    ├── README.md                   # 本风格的介绍
    └── prompt-assets/
        ├── 主动模式/README.md       # 独立、可复制的提示词
        ├── 微调模式/README.md       # 独立、可复制的提示词
        └── reference/              # 共享图片和已知来源记录
```

每个风格沿用稳定目录名；更新已有风格不创建日期副本。确需并列保存同一摄影师的不同系列时，使用可辨认的系列后缀。风格介绍按 [style-to-prompt.md](style-to-prompt.md) 编写；仓库首页链接各风格、模式目录和参考图入口。

图片集中到单数 `reference/`，不散放在 `prompt-assets/` 根目录。旧资产中的 `prompt.md`、带说明的提示词文件名或 `references/` 在归档副本中归一到上述结构；同步更新相对链接，中文和空格路径使用有效的 Markdown 链接目标或百分号编码。模式 README 中的图片路径通常为 `../reference/<filename>`。技能自身的 `references/` 是流程文档目录，不随资产命名规则更改。

## 整理与提交

1. 清点输入，区分两种模式、参考图、来源及已有版本。保留原目录；归档整理只改变目录、文件名、介绍和必要链接，图片字节与可复制提示词正文保持一致。用户要求内容修订时，提交说明应写明实际修改。
2. 只纳入本次获授权的最终资产、风格介绍和来源记录。原视频、完整 ASR、全帧证据、模型、环境与缓存留在原工作区；`.DS_Store` 等系统文件和重复 ZIP 不进入提交。未获授权的用户原片不混入参考图。
3. 风格介绍与来源记录保留已有版本标签（如 `V0.1.2`）、出处和真实验证状态。归档或改名不自动提升提示词内容版本；来源不明则写“用户提供，原始发表链接未提供”，不猜作者或宣称是已核验原作。
4. 查看工作区与暂存区，获取远端更新并整合后再提交；保留其他任务的改动，只暂存本次具体路径。已有暂存内容不得混入本次提交，不使用强推或清空工作区来处理冲突。
5. 检查模式文件能独立使用、风格介绍已到位、图片与目录链接可解析。对于纯整理，核对图片内容指纹与提示词正文未变；执行 `git diff --cached --check` 并审阅暂存差异后提交。

优先复用 `git` 与 `gh`。账户检查可从以下命令开始，变量仅用于本次调用；本机目录不存在时先克隆已确认的远端，不能在不存在的目录里继续操作。

```bash
# 执行前替换为本次已确定的真实目标。
archive_repo='<owner>/<repo>'
archive_dir='/absolute/path/to/archive'
gh auth status
gh repo view "$archive_repo" --json nameWithOwner,isPrivate,url,defaultBranchRef
git -C "$archive_dir" remote -v
git -C "$archive_dir" status --short
git -C "$archive_dir" fetch origin
```

认证失败时先区分网络访问问题与令牌问题。若 `GITHUB_TOKEN` 或 `GH_TOKEN` 覆盖了本机已保存的有效登录，可仅对本次命令使用 `env -u GITHUB_TOKEN -u GH_TOKEN gh auth status` 核验已保存账户，并把同样的前缀用于必要的 `gh` 或 `git` 调用；不打印令牌或改写全局凭据。没有可用登录时保留本地成果，说明尚未上传及需要的登录动作。

## 推送与远端核验

执行前核对实际默认分支、当前工作分支和本次授权的推送目标，不假定分支叫 `main`。已有仓库在完成本地提交后推送对应分支；远端有新提交时先整合，不能用强推覆盖。处于 detached HEAD 时先确定本次使用的分支，不向猜测的分支推送。

```bash
archive_branch="$(git -C "$archive_dir" branch --show-current)"
if test -z "$archive_branch"; then
  echo "Select an authorized branch before pushing." >&2
  exit 1
fi
git -C "$archive_dir" push origin "HEAD:refs/heads/$archive_branch"
git -C "$archive_dir" rev-parse HEAD
git -C "$archive_dir" ls-remote origin "refs/heads/$archive_branch"
```

仅在仓库尚不存在、本地已有首个提交且目标远端未配置时，可按已有创建授权使用：

```bash
gh repo create "$archive_repo" --private --source "$archive_dir" --remote origin --push
```

推送后确认远端分支 SHA 与本地提交一致。需要核对文件时，将 `git ls-tree -r HEAD` 与 GitHub `repos/<owner>/<repo>/git/trees/<branch>?recursive=1` 返回的路径和 blob SHA 对比；不能只核对文件数量。确认旧路径已移除、新路径和图片链接有效。

检查风格与两种模式目录能直接展示 README；可通过 GitHub 的 `repos/<owner>/<repo>/readme/<directory>?ref=<branch>` 接口核对返回路径与文件 SHA，目录中的中文需正确编码，或在已登录浏览器中查看。`reference/` 可作为纯图片目录，不要求额外介绍文件。

最后交付仓库或风格入口、提交版本与实际检查结果。推送失败时区分本地已提交、远端未更新；不通过重复建库绕过失败，不把上传成功当成图像编辑效果已验证。
