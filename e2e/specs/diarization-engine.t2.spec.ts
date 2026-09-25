import { test, expect } from '../fixtures/electron';
import { realUserDataDir, fileSig } from '../fixtures/real-user-data';
import { readUserConfig, writeUserConfig } from '../fixtures/user-config';

/**
 * T2 — speaker-diarization engine setting (Settings -> Transcribe, "Speaker
 * detection"). Drives the real backend through the preload bridge and asserts
 * config.json. Model-free + deterministic on every OS: the one guarantee that
 * matters -- a non-default engine is never saved while its models are missing,
 * because meeting processing never downloads them -- holds whether the
 * steno-diarize sidecar is bundled (its cache under the isolated temp
 * user-data dir is empty) or absent (Windows, or a dev build without it).
 * Actually downloading Nemotron 3 (~95 MB) is out of scope here.
 */

type Bridge = {
  diarizationEngine: {
    get: () => Promise<{ success: boolean; engine?: string; valid_engines?: string[] }>;
    set: (
      engine: string,
    ) => Promise<{ success: boolean; engine?: string; models_ready?: boolean; error?: string }>;
  };
};
type StenoWindow = Window & { stenoai: Bridge };

test('a fresh install diarizes with Sortformer', async ({ launchApp }) => {
  const { page } = await launchApp();

  const res = await page.evaluate(() => (window as StenoWindow).stenoai.diarizationEngine.get());
  expect(res).toMatchObject({
    success: true,
    engine: 'sortformer',
    valid_engines: ['sortformer', 'nemotron3'],
  });
});

test('Nemotron 3 is refused while its models are missing; config untouched', async ({
  launchApp,
  userDataDir,
}) => {
  const realDirBefore = fileSig(realUserDataDir());
  // A developer's STENOAI_DIARIZE_MODEL_DIR pointing at a populated cache
  // would make Nemotron 3 ready and the refusal below vacuous; the sidecar
  // treats an empty value as unset, so the isolated user-data cache is used.
  const { page } = await launchApp({ env: { STENOAI_DIARIZE_MODEL_DIR: '' } });

  const res = await page.evaluate(() =>
    (window as StenoWindow).stenoai.diarizationEngine.set('nemotron3'),
  );
  expect(res.success).toBe(false);
  expect(readUserConfig(userDataDir).diarization_engine ?? 'sortformer').toBe('sortformer');

  const after = await page.evaluate(() => (window as StenoWindow).stenoai.diarizationEngine.get());
  expect(after).toMatchObject({ success: true, engine: 'sortformer' });

  expect(fileSig(realUserDataDir())).toBe(realDirBefore);
});

test('switching back to Sortformer persists without a model check', async ({
  launchApp,
  userDataDir,
}) => {
  // Seeded before launch: the state a user is in after a successful
  // Nemotron 3 download + save.
  writeUserConfig(userDataDir, { diarization_engine: 'nemotron3' });
  const { page } = await launchApp();

  const before = await page.evaluate(() => (window as StenoWindow).stenoai.diarizationEngine.get());
  expect(before).toMatchObject({ success: true, engine: 'nemotron3' });

  const res = await page.evaluate(() =>
    (window as StenoWindow).stenoai.diarizationEngine.set('sortformer'),
  );
  expect(res).toMatchObject({ success: true, engine: 'sortformer' });
  await expect
    .poll(() => readUserConfig(userDataDir).diarization_engine)
    .toBe('sortformer');
});

test('an unknown engine is rejected at the IPC boundary', async ({ launchApp, userDataDir }) => {
  const { page } = await launchApp();

  const res = await page.evaluate(() =>
    (window as StenoWindow).stenoai.diarizationEngine.set('--help'),
  );
  expect(res).toMatchObject({ success: false, error: 'Unknown speaker detection engine' });
  expect(readUserConfig(userDataDir).diarization_engine ?? 'sortformer').toBe('sortformer');
});
