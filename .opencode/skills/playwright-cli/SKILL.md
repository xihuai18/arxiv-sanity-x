---
name: playwright-cli
description: 使用 playwright-cli 做网页导航、表单交互、截图、录屏、网络调试、多标签和会话管理。
license: Apache-2.0
compatibility: opencode
metadata:
  category: browser-automation
  requires_cli: '@playwright/cli'
  workflow: cli-first
---

# Playwright CLI Skill

## What I do

- 用 `playwright-cli` 驱动真实浏览器完成网页任务
- 适合登录、表单填写、截图、抓取页面信息、调试前端问题
- 比 MCP 更偏 CLI 工作流：命令短、上下文更轻、适合 coding agent 持续调用

## Before you start

需要先具备这些前提：

1. Node.js 18+
2. 已安装 CLI

```bash
npm install -g @playwright/cli@latest
playwright-cli --help
```

3. 如果全局命令不可用，可改用：

```bash
npx playwright-cli --help
```

4. 如果用户想直接使用上游仓库自带 skill，可以执行：

```bash
playwright-cli install --skills
```

## When to use me

优先使用我：

- 用户要操作真实网页
- 用户要填写表单、登录、切标签、上传文件
- 用户要生成截图、PDF、trace、video 或排查控制台 / 网络请求
- 用户明确要求使用 `playwright-cli`

不要优先使用我：

- 用户只需要静态抓取 HTML 文本
- 本地已有 Playwright 测试且用户只要修代码，不需要真实浏览器交互
- 环境里没有安装 `@playwright/cli` 且用户不希望安装任何额外工具

## Core mental model

`playwright-cli` 的基本流程是：

1. `open` 或 `goto`
2. `snapshot` 获取元素引用
3. 使用形如 `e15` 的 ref 做 `click` / `fill` / `check` / `select`
4. 在关键节点保存 artifact
5. `close` 或保留 session 继续工作

重要习惯：

- 导航后先 `snapshot`
- 页面结构变化后再 `snapshot`
- 不要猜 ref；要根据最新 snapshot 操作
- `screenshot` 适合交付物，`snapshot` 才是主工作流

## Quick start

```bash
playwright-cli open https://playwright.dev
playwright-cli snapshot
playwright-cli click e15
playwright-cli type "locator"
playwright-cli press Enter
playwright-cli screenshot
playwright-cli close
```

## Commands to reach for first

### Open and navigate

```bash
playwright-cli open
playwright-cli open https://example.com
playwright-cli open https://example.com --headed
playwright-cli goto https://example.com/login
playwright-cli reload
playwright-cli go-back
playwright-cli go-forward
```

### Inspect and interact

```bash
playwright-cli snapshot
playwright-cli click e3
playwright-cli dblclick e7
playwright-cli fill e5 "user@example.com"
playwright-cli type "hello"
playwright-cli press Enter
playwright-cli hover e4
playwright-cli select e9 "option-value"
playwright-cli check e12
playwright-cli uncheck e12
playwright-cli drag e2 e8
playwright-cli upload ./document.pdf
playwright-cli eval "document.title"
playwright-cli eval "el => el.textContent" e5
```

### Save artifacts

```bash
playwright-cli screenshot
playwright-cli screenshot e5
playwright-cli screenshot --filename=page.png
playwright-cli pdf --filename=page.pdf
playwright-cli console
playwright-cli console warning
playwright-cli network
```

### Debug deeper

```bash
playwright-cli run-code "async page => await page.title()"
playwright-cli tracing-start
playwright-cli tracing-stop
playwright-cli video-start
playwright-cli video-stop demo.webm
playwright-cli route "**/*.jpg" --status=404
playwright-cli route-list
playwright-cli unroute
```

## Sessions

命名 session 很重要，尤其是并行任务或需要长期保留登录态时。

```bash
playwright-cli -s=todo-app open https://demo.playwright.dev/todomvc/ --persistent
playwright-cli -s=todo-app snapshot
playwright-cli -s=todo-app click e21
playwright-cli -s=todo-app close
```

常用命令：

```bash
playwright-cli list
playwright-cli close-all
playwright-cli kill-all
playwright-cli show
```

如果用户希望 agent 默认使用同一个 session，也可以配合环境变量：

```bash
PLAYWRIGHT_CLI_SESSION=todo-app claude .
```

## Persistence and state

- 默认 profile 在内存里，浏览器关掉就丢失
- `--persistent` 会把 profile 落盘，适合需要跨重启保留状态
- `state-save` / `state-load` 适合显式保存与恢复登录态

常用命令：

```bash
playwright-cli state-save auth.json
playwright-cli state-load auth.json
playwright-cli cookie-list
playwright-cli localstorage-list
playwright-cli sessionstorage-list
```

更多细节见：`references/session-management.md`

## Recommended workflow patterns

### A. 真实页面验证

1. `open` 到目标页面
2. `snapshot`
3. 按 ref 执行交互
4. 在关键节点用 `console` / `network` / `screenshot` 留证据
5. 输出用户可复现的步骤

### B. 表单提交或登录

1. 打开页面后先 `snapshot`
2. 用 `fill` / `type` 输入
3. 提交前后各做一次 `snapshot`
4. 若需要保留状态，用 `state-save` 或 `--persistent`

### C. 前端报错排查

1. 重现问题
2. 立即跑 `console`
3. 必要时跑 `network`
4. 如需更深信息，用 `tracing-start` / `tracing-stop`

### D. 生成测试思路

1. 先用 CLI 真实走完用户流程
2. 固化稳定步骤和断言点
3. 再把它们翻译成 Playwright test

更多细节见：`references/test-generation.md`

## Open parameters

```bash
playwright-cli open --browser=chrome
playwright-cli open --browser=firefox
playwright-cli open --browser=webkit
playwright-cli open --browser=msedge
playwright-cli open --extension
playwright-cli open --persistent
playwright-cli open --profile=/path/to/profile
playwright-cli open --config=my-config.json
```

## Success criteria

一个任务完成得好，通常至少满足：

- 页面交互步骤可复现
- 关键证据被记录下来，例如 snapshot、screenshot、console、network、trace 或 video
- 如果要交付测试，交付的是稳定步骤，不依赖过时 ref

## Common mistakes

- 没有先 `snapshot` 就直接猜元素 ref
- 页面更新后继续使用旧 ref
- 应该用命名 session 却一直复用默认 session，导致状态混乱
- 明明要交付证据，却只做了交互没保存 artifact
- 环境里没装 `@playwright/cli`，却直接假设 `playwright-cli` 可执行

## Local fallback

如果用户的环境里没有全局二进制，可改用：

```bash
npx playwright-cli open https://example.com
npx playwright-cli snapshot
```

## Tests

这个 skill 在仓库里带了一个最小 smoke test：

```bash
python -m unittest "skills/playwright-cli/tests/test_playwright_cli_smoke.py"
```

它会验证两件事：

- `playwright-cli --help` 可用
- 最小浏览器流程 `open -> snapshot -> eval -> close` 可跑通

## How to help the user well

- 对“调试网页”任务，优先给可复现命令链，而不是只给口头描述
- 对“写测试”任务，先用 CLI 把真实用户路径走通，再总结测试步骤
- 对“截图/录屏取证”任务，明确输出文件名与保存时机
- 如果用户明确提到多页面、多账号或长期会话，第一时间启用命名 session
