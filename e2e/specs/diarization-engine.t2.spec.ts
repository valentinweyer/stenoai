import { mkdirSync, writeFileSync } from 'fs';
import path from 'path';
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

type SpeakerModelStatus = {
  success: boolean;
  ready: boolean;
  missing_models?: string[];
  error?: string;
};
type SetupCheck = {
  success: boolean;
  allGood?: boolean;
  checks?: { name: string; ok: boolean; status: string; detail: string }[];
};

type Bridge = {
  setup: {
    speakerModelsStatus: (engine?: string) => Promise<SpeakerModelStatus>;
    check: () => Promise<SetupCheck>;
  };
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

/** Seed a complete compiled bundle with the artifacts the sidecar's
 *  readiness check requires (ModelReadiness.requiredArtifactRelativePaths). */
function seedBundle(bundle: string, artifacts: string[]) {
  for (const artifact of artifacts) {
    const file = path.join(bundle, artifact);
    mkdirSync(path.dirname(file), { recursive: true });
    writeFileSync(file, Buffer.from([1]));
  }
}

test('a Sortformer cache from before the FluidAudio 0.17 upgrade reads as missing', async ({
  launchApp,
  userDataDir,
}) => {
  test.skip(process.platform !== 'darwin', 'the steno-diarize sidecar is macOS-only');
  // Exactly what a user who finished onboarding on a FluidAudio 0.15 build
  // has: Sortformer bundles directly under sortformer/ (0.17 reads only the
  // rebuilt sortformer/v3/fp16/ set) plus complete embedding models.
  const cache = path.join(userDataDir, 'models', 'speaker-diarization');
  const sortformerArtifacts = [
    'coremldata.bin',
    'metadata.json',
    'model0/model.mil',
    'model0/weights/0-weight.bin',
    'model1/model.mil',
    'model1/weights/1-weight.bin',
  ];
  for (const bundle of ['Sortformer_v2.1.mlmodelc', 'SortformerNvidiaHigh_v2.mlmodelc']) {
    seedBundle(path.join(cache, 'sortformer', bundle), sortformerArtifacts);
  }
  const embeddingArtifacts = ['coremldata.bin', 'metadata.json', 'model.mil', 'weights/weight.bin'];
  for (const bundle of ['pyannote_segmentation.mlmodelc', 'wespeaker_v2.mlmodelc']) {
    seedBundle(path.join(cache, 'speaker-diarization', bundle), embeddingArtifacts);
  }
  const { page } = await launchApp({ env: { STENOAI_DIARIZE_MODEL_DIR: '' } });

  const status = await page.evaluate(() =>
    (window as StenoWindow).stenoai.setup.speakerModelsStatus(),
  );
  test.skip(
    !status.success,
    `steno-diarize is not bundled in this backend build (${status.error ?? 'unavailable'})`,
  );
  // Only the two re-laid-out Sortformer bundles are missing -- this is what
  // Settings -> Speaker detection turns into a Download action.
  expect(status).toMatchObject({ success: true, ready: false });
  expect(status.missing_models).toEqual([
    'sortformer/v3/fp16/Sortformer_v2.1.mlmodelc',
    'sortformer/v3/fp16/SortformerNvidiaHigh_v2.mlmodelc',
  ]);

  // Speaker models stay optional for setup: the stale cache is a non-failing
  // warning, so allGood is unaffected and onboarding never re-runs for it --
  // which is why the Settings row is the visible repair path.
  const setup = await page.evaluate(() => (window as StenoWindow).stenoai.setup.check());
  const speaker = setup.checks?.find((c) => c.name.includes('speaker'));
  expect(speaker).toMatchObject({ ok: true, status: 'warn' });
});
