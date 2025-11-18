import { test, expect } from '@playwright/test'

// Minimal smoke test for STORY-005: verify that the frontend boots
// and renders the main header text.

test('home page loads and shows header', async ({ page }) => {
  await page.goto('/')

  // Header from Header.tsx
  await expect(
    page.getByRole('heading', { name: /Requirements Engineering Platform/ }),
  ).toBeVisible()
})
