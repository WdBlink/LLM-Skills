# 发布验收与定位

以下命令是只读检查示例，替换为现场已核验的值。不要把示例地址直接用于实际变更。优先现有工具和 curl；不安装额外探针，不假定第三方测速服务支持 IPv6。

## 最小证据矩阵

| 检查 | 来源与含义 |
| --- | --- |
| 后端 health/首页 | NAS 或同网段到后端，证明后端可达 |
| 指定 NAS 地址 + 正确 Host/SNI | 证明该反代入口与证书可用，绕过 DNS |
| 公共 DNS A/AAAA | 证明外部解析返回什么；记录 TTL，不等于连接可达 |
| 外部 IPv6 → 域名:端口 | IPv6 发布的必需验收；记录外部来源 |
| 外部 IPv4 → 域名:端口 | 仅双栈/IPv4 目标要求；单独记录 |
| 未登录敏感 API | 应用约定的 401/403 或登录跳转，不能只检查首页 |
| 登录与核心操作 | HTTPS 域名完整路径，检查 Cookie、回调、流式连接等实际功能 |
| DDNS 与证书维护 | 找到更新者及近期成功证据；只配置当前 IP 不等于长期维护 |

## 定向检查示例

```bash
# 当前终端在哪里运行，就代表哪里的网络；--noproxy 绕过 curl 代理，不能证明外网来源。
curl --noproxy '*' --connect-timeout 5 --max-time 15 \
  --resolve 'service.example.com:9443:[2001:db8::54]' \
  'https://service.example.com:9443/api/health'

# 相同 Host/SNI 定向到 NAS 内网，用于区分 DNS 与反代。
curl --noproxy '*' --connect-timeout 5 --max-time 15 \
  --resolve 'service.example.com:9443:192.0.2.54' \
  'https://service.example.com:9443/api/health'

# 在受控外部主机执行，直接测试对应 IP 版本的入口。
curl --noproxy '*' -6 --connect-timeout 5 --max-time 15 'https://service.example.com:9443/'
curl --noproxy '*' -4 --connect-timeout 5 --max-time 15 'https://service.example.com:9443/'

# fake-IP 环境的 DNS 交叉检查；这是公共递归解析结果。
curl --max-time 15 'https://dns.alidns.com/resolve?name=service.example.com&type=AAAA'
```

省略 health 路径时按应用实际接口替换。禁止用 `-k` 的结果证明 TLS 正常。响应头可用于诊断，但要避免打印登录响应里的 Set-Cookie。

## 按结果定位

- 后端失败：查进程、绑定、后端防火墙和日志；不先改 DNS。
- 后端成功、NAS 定向失败：查规则来源 Host/端口、证书绑定、目标地址，以及 NAS 到后端的网络。
- 内网定向成功、域名失败：查 A/AAAA、缓存、代理和 IP 版本。带错误 A 的域名可能影响客户端，不能假定所有客户端都会可靠回落 IPv6。
- 内网成功、外网超时：核验对应 IP 版本的路由器/NAS 入站策略；IPv4 再核对公网 WAN/NAT。对照相同外部网络访问已知正常 NAS 端口，必要时捕获实际入站是否到达。只有证据才能区分运营商限制与设备策略。
- 换高端口后能通：保留此现象和验证范围，避免凭此推广所有运营商均封 443。若改变了既定 URL，更新交付和应用回调配置。
- 首页成功、生成/聊天失败：查反代超时、SSE 缓冲、WebSocket Upgrade、Cookie、跨源/公开 URL、应用模型配置；不能用 health 成功结束验收。
- 无独立外部 IPv6 测试能力：请用户关闭 Wi-Fi 测试**完整 HTTPS URL（含端口）**，反馈页面或具体错误；同时对照既有 NAS 入口。未收到结果时保留“外网待验收”。

第三方探针只提交公开 URL 或公开地址/端口，不提交凭据或私有内容。被拒、超时或出现验证码时不要反复重试或绕过访问控制，换已授权的验证来源。
