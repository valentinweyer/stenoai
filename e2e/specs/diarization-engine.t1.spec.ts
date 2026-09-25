import { test, expect } from '../fixtures/electron';

/**
 * T1 -- the Speaker detection picker's download-then-save interaction. The
 * backend contract (a non-default engine is refused while its models are
 * missing) is diarization-engine.t2; this pins what the user sees around it:
 * picking Nemotron 3 prepares its models and then shows it as saved, and a
 * failed download says so and leaves Standard selected. macOS-only UI.
 */

async function openAiSettings(page: import('@playwright/test').Page) {
  await page.evaluate(() => {
    window.location.hash = '#/settings?tab=ai';
  });
  await expect(page.getByTestId('diarization-engine-select')).toBeVisible();
}

async function chooseNemotron(page: import('@playwright/test').Page) {
  await page.getByTestId('diarization-engine-select').click();
  await page.getByRole('option', { name: /nemotron 3/i }).click();
}

test('picking Nemotron 3 prepares its models and saves it', async ({ launchApp }) => {
  test.skip(process.platform !== 'darwin', 'speaker detection engines are macOS-only');
  const { page } = await launchApp({ mockIpc: true });
  await openAiSettings(page);

  const trigger = page.getByTestId('diarization-engine-select');
  await expect(trigger).toContainText('Standard');
  await chooseNemotron(page);

  await expect(trigger).toContainText('Nemotron 3');
  await expect(trigger).toBeEnabled();
  const saved = await page.evaluate(() =>
    (window as unknown as {
      stenoai: { diarizationEngine: { get: () => Promise<{ engine: string }> } };
    }).stenoai.diarizationEngine.get(),
  );
  expect(saved.engine).toBe('nemotron3');
});

test('a failed Nemotron 3 download keeps Standard and says so', async ({ launchApp }) => {
  test.skip(process.platform !== 'darwin', 'speaker detection engines are macOS-only');
  const { page } = await launchApp({
    mockIpc: true,
    env: { STENOAI_E2E_SPEAKER_MODEL_FAILURE: '1' },
  });
  await openAiSettings(page);
  await chooseNemotron(page);

  const trigger = page.getByTestId('diarization-engine-select');
  await expect(page.locator('#diarization-engine-description')).toContainText(
    'previous model is still active',
  );
  await expect(trigger).toContainText('Standard');
  await expect(trigger).toBeEnabled();
});
