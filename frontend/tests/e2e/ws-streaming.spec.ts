import { test, expect } from '@playwright/test'

// WebSocket streaming behaviour: verify that after sending a message, the
// assistant response is rendered via the streaming bubble (tied to
// message.chunk events) and eventually promoted to a normal assistant
// message (after message.complete).

test('assistant response streams then finalizes in chat', async ({ page }) => {
  test.setTimeout(90_000)

  await page.goto('/')

  // Ensure sessions sidebar is present.
  await expect(page.getByRole('heading', { name: 'Sessions' })).toBeVisible()

  const projectName = 'WS Streaming E2E'
  page.once('dialog', async (dialog) => {
    await dialog.accept(projectName)
  })

  await page.getByRole('button', { name: 'New' }).click()
  await page.waitForURL('**/sessions/*')

  const chatInput = page.getByPlaceholder('Describe a feature or requirement...').last()
  await expect(chatInput).toBeVisible()

  await chatInput.fill('Users should be able to reset their password via email.')
  await chatInput.press('Enter')

  // 1. Streaming bubble should appear while the assistant is responding.
  const streamingBubble = page.locator('[data-testid="streaming-message"]')
  await expect(streamingBubble).toBeVisible({ timeout: 45_000 })

  // 2. Eventually, streaming bubble should disappear once the message is
  // completed and promoted into the main chat history.
  await expect(streamingBubble).toBeHidden({ timeout: 45_000 })

  // 3. At least one assistant chat message should now be present.
  const assistantMessages = page.locator('[data-testid="chat-message"][data-role="assistant"]')
  await expect(assistantMessages).toHaveCount(1, { timeout: 45_000 })

  const firstAssistantText = (await assistantMessages.first().textContent()) ?? ''
  expect(firstAssistantText.trim().length).toBeGreaterThan(0)
})
