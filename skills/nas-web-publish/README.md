# 群晖公网发布

![Version](https://img.shields.io/badge/version-0.1.0-CC785C)
![Reserved](https://img.shields.io/badge/Rights_reserved-blueviolet)

通过群晖 DSM 反向代理为内网 Web 服务提供 HTTPS 公网入口。支持 IPv6 DDNS、域名加端口和按需配置的 IPv4 转发。

## 安装到 Codex

在本仓库根目录执行。

```bash
mkdir -p ~/.codex/skills/nas-web-publish
rsync -a skills/nas-web-publish/ ~/.codex/skills/nas-web-publish/
```

## 使用

```text
用 $nas-web-publish 把指定的内网 Web 服务发布到 https://service.example.com:9443。
使用 IPv6 DDNS，从独立外网核验 TLS、未登录访问限制和登录后的一项核心操作。
```

提供后端地址与端口、NAS、目标域名以及客户端所需的 IP 版本。将示例域名和端口替换为你的入口。已有的网络记录可用于定位设备，配置以现场核验为准。

部署流程见 [SKILL.md](SKILL.md)。分层验收、`curl` 示例和排障方法见 [外网验收指南](references/verification.md)。
