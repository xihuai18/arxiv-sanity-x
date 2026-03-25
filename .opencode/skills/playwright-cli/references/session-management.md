# Session Management with playwright-cli

## 什么时候要用命名 session

这些情况不要继续依赖默认 session：

- 同时操作两个网站或两个账号
- 一个任务要长期保留登录态
- 你需要让用户稍后继续接管同一个浏览器上下文

## 常见模式

### 模式 1：一个任务一个 session

```bash
playwright-cli -s=checkout open https://example.com/checkout --persistent
playwright-cli -s=checkout snapshot
playwright-cli -s=checkout close
```

适合：流程明确、需要隔离状态。

### 模式 2：一个项目一个 session

```bash
playwright-cli -s=admin-panel open https://admin.example.com --persistent
playwright-cli -s=admin-panel state-save admin.json
```

适合：同一产品反复调试。

## 状态保存建议

- 需要跨浏览器重启复用状态：`--persistent`
- 需要显式导出登录态：`state-save auth.json`
- 需要恢复状态：`state-load auth.json`

## 排障建议

- session 异常时先 `playwright-cli list`
- 卡死时用 `playwright-cli close-all`
- 浏览器僵死或残留进程时再用 `playwright-cli kill-all`
- 用户想观察 agent 正在做什么时用 `playwright-cli show`
