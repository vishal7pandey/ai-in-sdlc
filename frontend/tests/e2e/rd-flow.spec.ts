import { test, expect } from '@playwright/test'
import { promises as fs } from 'fs'

// End-to-end RD flow: create a session via the UI, send a requirements-style
// message, wait for requirements to appear in the sidebar, generate an RD
// preview, then export the markdown file and validate its contents.
//
// This test assumes:
// - Backend API is running on http://localhost:8000
// - WebSocket endpoint is available at VITE_WS_URL (e.g. ws://localhost:8000/ws)
// - RD generation and export endpoints are wired and working
//
// It is intentionally focused on the happy path per STORY-006B.

test('user can generate and export an RD via the UI', async ({ page }) => {
  // RD flow can be a bit slower due to orchestrator + synthesis; allow extra time.
  test.setTimeout(90_000)

  // 1. Navigate to home
  await page.goto('/')

  // Ensure the sessions sidebar is visible.
  await expect(page.getByRole('heading', { name: 'Sessions' })).toBeVisible()

  // 2. Create a new session via the existing "New" button + window.prompt.
  const projectName = 'RD E2E Project'
  page.once('dialog', async (dialog) => {
    await dialog.accept(projectName)
  })

  await page.getByRole('button', { name: 'New' }).click()

  // 3. Wait for navigation to /sessions/:id and basic chat UI visibility.
  await page.waitForURL('**/sessions/*')

  const chatInput = page.getByPlaceholder('Describe a feature or requirement...').last()
  await expect(chatInput).toBeVisible()

  // 4. Send a requirements-style message into the chat.
  const requirementText = 'Users must be able to log in with email and password.'
  await chatInput.fill(requirementText)
  await chatInput.press('Enter')

  // 5. Wait for a requirement to appear in the Requirements sidebar.
  const requirementsAside = page.locator('aside', { hasText: 'Requirements' })

  // Wait until some REQ-xxx style identifier appears. This relies on the
  // Extraction agent + orchestrator producing requirements with IDs.
  await expect(
    requirementsAside.getByText(/REQ-\d{3,}/),
  ).toBeVisible({ timeout: 45_000 })

  // 6. Click "Generate RD" to trigger RD generation via the backend endpoint.
  const generateButton = requirementsAside.getByRole('button', { name: /Generate RD/i })
  await expect(generateButton).toBeEnabled()
  await generateButton.click()

  // 7. Wait for the RD preview to appear. The preview is rendered as a <pre>
  // once rdContent is populated in RequirementsSidebar.
  const rdPreview = requirementsAside.locator('pre')
  await expect(rdPreview).toBeVisible({ timeout: 45_000 })

  const previewText = (await rdPreview.textContent()) ?? ''
  expect(previewText.length).toBeGreaterThan(0)
  expect(previewText).toMatch(/Requirements Document/i)
  expect(previewText).toMatch(/REQ-\d{3,}/)

  // 8. Export the RD as markdown and validate the downloaded file.
  const exportButton = requirementsAside.getByRole('button', { name: /Export MD/i })

  const [download] = await Promise.all([
    page.waitForEvent('download'),
    exportButton.click(),
  ])

  const filename = download.suggestedFilename()
  expect(filename).toMatch(/requirements-.*\.md$/)

  const downloadPath = await download.path()
  if (!downloadPath) {
    throw new Error('Download path is undefined')
  }

  const content = await fs.readFile(downloadPath, 'utf-8')

  // Basic structure checks
  expect(content).toMatch(/^# Requirements Document/m)
  expect(content).toMatch(/REQ-\d{3,}/)

  // The original requirement text (or a close variant) should be present.
  expect(content.toLowerCase()).toContain('login')

  // Ensure the file is non-trivial in size.
  expect(content.length).toBeGreaterThan(200)
})

test('Generate RD is not available when there are no requirements', async ({ page }) => {
  // This test covers the "no requirements" scenario from STORY-006B AC7.

  await page.goto('/')

  await expect(page.getByRole('heading', { name: 'Sessions' })).toBeVisible()

  const projectName = 'RD Empty Session'
  page.once('dialog', async (dialog) => {
    await dialog.accept(projectName)
  })

  await page.getByRole('button', { name: 'New' }).click()

  // Wait for navigation to the newly created session.
  await page.waitForURL('**/sessions/*')

  const requirementsAside = page.locator('aside', { hasText: 'Requirements' })
  await expect(requirementsAside).toBeVisible()

  // With no requirements extracted yet, the helper text should be visible
  // and the Generate RD button should not be present.
  await expect(
    requirementsAside.getByText('Extracted requirements will appear here as you converse with the AI.'),
  ).toBeVisible()

  const generateButton = requirementsAside.getByRole('button', { name: /Generate RD/i })
  await expect(generateButton).toHaveCount(0)
})
