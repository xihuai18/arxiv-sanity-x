# Test Generation from playwright-cli Workflows

## 推荐流程

先用 `playwright-cli` 跑真人路径，再写 Playwright test。这样更稳定。

1. 用 `open` / `goto` 进入目标页面
2. 用 `snapshot` 找到稳定元素
3. 完成一次真实业务流程
4. 记录关键断言点
5. 把步骤翻译成 `@playwright/test`

## 要记录什么

- 页面 URL 变化
- 关键按钮 / 输入框的稳定定位方式
- 成功提示、错误提示、页面状态变化
- 是否涉及 cookie、localStorage、上传文件、多标签

## 生成测试时的建议

- 不要把 `e15` 这种 snapshot ref 直接写进正式测试代码
- 正式测试应该回到真实 locator，比如 `getByRole`、`getByLabel`、`getByTestId`
- 先让 CLI 帮你确认交互路径，再让测试代码用更稳定的 selector 重建它

## 一个最小翻译思路

如果 CLI 流程是：

```bash
playwright-cli open https://example.com/login
playwright-cli snapshot
playwright-cli fill e3 "user@example.com"
playwright-cli fill e5 "password"
playwright-cli click e8
playwright-cli snapshot
```

那么正式测试里应当产出类似结构：

```ts
import { test, expect } from '@playwright/test'

test('login flow', async ({ page }) => {
  await page.goto('https://example.com/login')
  await page.getByLabel('Email').fill('user@example.com')
  await page.getByLabel('Password').fill('password')
  await page.getByRole('button', { name: 'Sign in' }).click()
  await expect(page).toHaveURL(/dashboard/)
})
```

重点是“保留流程与断言”，不要照搬临时 ref。
