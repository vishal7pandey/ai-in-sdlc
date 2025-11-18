import { test, expect } from '@playwright/test'

// End-to-end flow: create a new session from the sidebar
// and land in the chat view for that session.
// NOTE: This test expects the backend API to be running on http://localhost:8000
// so that /api/v1/sessions can be created successfully via the Vite proxy.

test('user can create a session and see the chat UI', async ({ page }) => {
  await page.goto('/')

  // Ensure the sessions sidebar is visible (desktop viewport by default).
  await expect(page.getByRole('heading', { name: 'Sessions' })).toBeVisible()

  // When clicking "New", the app uses window.prompt to ask for a project name.
  // Handle the prompt by providing a deterministic project name.
  page.once('dialog', async (dialog) => {
    await dialog.accept('Playwright E2E Project')
  })

  await page.getByRole('button', { name: 'New' }).click()

  // After creating the session, the app navigates to /sessions/:id.
  await page.waitForURL('**/sessions/*')

  // The chat input should now be visible on the session page.
  // There are two ChatPanel instances in the layout (mobile + desktop). On desktop viewport
  // the second one is the visible chat box, so assert visibility on the last match.
  await expect(
    page.getByPlaceholder('Describe a feature or requirement...').last(),
  ).toBeVisible()
})
