import { test, expect } from '../fixtures/electron';

/**
 * T1 -- the Speaker detection picker's download-then-save interaction. The
 * backend contract (a non-default engine is refused while its models are
 * missing) is diarization-engine.t2; this pins what the user sees around it:
 * picking Nemotron 3 prepares its models and then shows it as saved, a
 * failed download says so and leaves Standard selected, and an upgraded
 * install whose models are missing gets a Download action. macOS-only UI.
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
  const status = await page.evaluate(() =>
    (window as unknown as {
      stenoai: { setup: { speakerModelsStatus: (engine: string) => Promise<unknown> } };
    }).stenoai.setup.speakerModelsStatus('nemotron3'),
  );
  expect(status).toMatchObject({ success: false, ready: false, error: 'synthetic model status failure' });
  expect(status).not.toHaveProperty('missing_models');
  await chooseNemotron(page);

  const trigger = page.getByTestId('diarization-engine-select');
  await expect(page.locator('#diarization-engine-description')).toContainText(
    'previous model is still active',
  );
  await expect(trigger).toContainText('Standard');
  await expect(trigger).toBeEnabled();
});

test('a failed speaker detection setting read offers a retry', async ({ launchApp }) => {
  test.skip(process.platform !== 'darwin', 'speaker detection engines are macOS-only');
  const { app, page } = await launchApp({
    mockIpc: true,
    env: { STENOAI_E2E_DIARIZATION_ENGINE_READ_FAILURE: '1' },
  });
  await openAiSettings(page);

  const trigger = page.getByTestId('diarization-engine-select');
  const description = page.locator('#diarization-engine-description');
  await expect(trigger).toBeDisabled();
  await expect(description).toContainText('Could not load the speaker detection setting');
  const retry = page.getByTestId('diarization-engine-retry');
  await expect(retry).toBeVisible();

  await app.evaluate(() => {
    process.env.STENOAI_E2E_DIARIZATION_ENGINE_READ_FAILURE = '0';
  });
  await retry.click();
  await expect(trigger).toBeEnabled();
  await expect(trigger).toContainText('Standard');
  await expect(description).toContainText('Runs on your device');
  await expect(retry).toHaveCount(0);
});

test('an upgraded install is offered a download for its missing Standard models', async ({
  launchApp,
}) => {
  test.skip(process.platform !== 'darwin', 'speaker detection engines are macOS-only');
  // FluidAudio 0.17 moved the Sortformer cache to sortformer/v3/fp16/, so a
  // pre-upgrade install reads as missing until it re-downloads. Meeting
  // processing never downloads, so this row is the visible way back.
  const { page } = await launchApp({
    mockIpc: true,
    env: { STENOAI_E2E_SPEAKER_MODELS_MISSING: '1' },
  });
  await openAiSettings(page);

  const description = page.locator('#diarization-engine-description');
  const trigger = page.getByTestId('diarization-engine-select');
  await expect(description).toContainText('models are unavailable');
  const download = page.getByTestId('diarization-models-download');
  await download.click();

  // The button also hides the moment the download STARTS, so wait for the
  // settled state: picker enabled again and the default description back.
  await expect(trigger).toBeEnabled();
  await expect(description).toContainText('Runs on your device');
  await expect(download).toHaveCount(0);
  await expect(trigger).toContainText('Standard');
});
