import { describe, test, expect, beforeEach, vi } from 'vitest';
import * as React from 'react';
import { renderHook, act, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

/**
 * Meeting processing never downloads speaker models, so switching engines
 * must prepare any missing models BEFORE the choice is saved -- otherwise
 * every later meeting silently falls back to channel-only "You"/"Others"
 * labels. That includes Sortformer: FluidAudio 0.17 moved its cache to
 * `sortformer/v3/fp16/`, so an upgraded install has to re-download it. These
 * pin the ordering, the upgrade case and the failure path.
 */

const h = vi.hoisted(() => ({
  getEngine: vi.fn(),
  setEngine: vi.fn(),
  speakerModels: vi.fn(),
  speakerModelsStatus: vi.fn(),
}));

vi.mock('@/lib/ipc', () => ({
  ipc: () => ({
    diarizationEngine: { get: h.getEngine, set: h.setEngine },
    setup: { speakerModels: h.speakerModels, speakerModelsStatus: h.speakerModelsStatus },
  }),
}));

import {
  useDiarizationEngine,
  useDiarizationModelsReady,
  useSetDiarizationEngine,
} from './useModels';

function status(ready: boolean) {
  return {
    success: true,
    ready,
    cache_directory: '/tmp/models',
    required_models: [],
    missing_models: ready ? [] : ['sortformer/v3/fp16/Sortformer_v2.1.mlmodelc'],
  };
}

const UNAVAILABLE = {
  success: false,
  ready: false,
  error: 'Speaker diarization is unavailable on this system',
};

function wrapper() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
  };
}

function renderEngine() {
  return renderHook(
    () => {
      const engine = useDiarizationEngine();
      return {
        engine,
        ready: useDiarizationModelsReady(engine.data),
        setEngine: useSetDiarizationEngine(),
      };
    },
    { wrapper: wrapper() },
  );
}

describe('useSetDiarizationEngine', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    h.getEngine.mockResolvedValue({
      success: true,
      engine: 'sortformer',
      valid_engines: ['sortformer', 'nemotron3'],
    });
    h.speakerModelsStatus.mockResolvedValue(status(true));
  });

  test('prepares missing Nemotron 3 models before saving the choice', async () => {
    h.speakerModelsStatus.mockImplementation(async (engine?: string) =>
      status(engine !== 'nemotron3'),
    );
    h.speakerModels.mockResolvedValue(status(true));
    h.setEngine.mockResolvedValue({ success: true, engine: 'nemotron3' });
    const { result } = renderEngine();
    await waitFor(() => expect(result.current.engine.data).toBe('sortformer'));

    await act(async () => {
      await result.current.setEngine.mutateAsync('nemotron3');
    });

    expect(h.speakerModelsStatus).toHaveBeenCalledWith('nemotron3');
    expect(h.speakerModels).toHaveBeenCalledWith('nemotron3');
    expect(h.speakerModels.mock.invocationCallOrder[0]).toBeLessThan(
      h.setEngine.mock.invocationCallOrder[0],
    );
    expect(h.setEngine).toHaveBeenCalledWith('nemotron3');
    await waitFor(() => expect(result.current.engine.data).toBe('nemotron3'));
  });

  test('a failed download never saves the engine and keeps the previous one', async () => {
    h.speakerModelsStatus.mockResolvedValue(status(false));
    h.speakerModels.mockResolvedValue({ success: false, ready: false, error: 'offline' });
    const { result } = renderEngine();
    await waitFor(() => expect(result.current.engine.data).toBe('sortformer'));

    await act(async () => {
      await result.current.setEngine.mutateAsync('nemotron3').catch(() => undefined);
    });

    expect(h.setEngine).not.toHaveBeenCalled();
    await waitFor(() => expect(result.current.setEngine.isError).toBe(true));
    expect(result.current.engine.data).toBe('sortformer');
  });

  test('an engine whose models are ready switches without a download', async () => {
    h.setEngine.mockResolvedValue({ success: true, engine: 'sortformer' });
    const { result } = renderEngine();
    await waitFor(() => expect(result.current.engine.data).toBe('sortformer'));

    await act(async () => {
      await result.current.setEngine.mutateAsync('sortformer');
    });

    expect(h.speakerModels).not.toHaveBeenCalled();
    expect(h.setEngine).toHaveBeenCalledWith('sortformer');
  });

  test('an upgraded install re-downloads missing Sortformer models', async () => {
    let prepared = false;
    h.speakerModelsStatus.mockImplementation(async () => status(prepared));
    h.speakerModels.mockImplementation(async () => {
      prepared = true;
      return status(true);
    });
    h.setEngine.mockResolvedValue({ success: true, engine: 'sortformer' });
    const { result } = renderEngine();
    await waitFor(() => expect(result.current.ready.data).toBe(false));

    await act(async () => {
      await result.current.setEngine.mutateAsync('sortformer');
    });

    expect(h.speakerModels).toHaveBeenCalledWith('sortformer');
    expect(h.setEngine).toHaveBeenCalledWith('sortformer');
    await waitFor(() => expect(result.current.ready.data).toBe(true));
  });

  test('an unavailable sidecar never blocks switching back to Sortformer', async () => {
    h.speakerModelsStatus.mockResolvedValue(UNAVAILABLE);
    h.setEngine.mockResolvedValue({ success: true, engine: 'sortformer' });
    const { result } = renderEngine();
    await waitFor(() => expect(result.current.engine.data).toBe('sortformer'));

    await act(async () => {
      await result.current.setEngine.mutateAsync('sortformer');
    });

    expect(h.speakerModels).not.toHaveBeenCalled();
    expect(h.setEngine).toHaveBeenCalledWith('sortformer');
    // Nothing confirmed the models, so readiness must stay unknown, not true.
    await waitFor(() => expect(result.current.ready.isFetching).toBe(false));
    expect(result.current.ready.data).toBeNull();
  });
});

describe('useDiarizationModelsReady', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    h.getEngine.mockResolvedValue({
      success: true,
      engine: 'sortformer',
      valid_engines: ['sortformer', 'nemotron3'],
    });
  });

  test("reports the saved engine's readiness", async () => {
    h.speakerModelsStatus.mockResolvedValue(status(false));
    const { result } = renderEngine();

    await waitFor(() => expect(result.current.ready.data).toBe(false));
    expect(h.speakerModelsStatus).toHaveBeenCalledWith('sortformer');
  });

  test('stays unknown, not missing, when the sidecar cannot answer', async () => {
    h.speakerModelsStatus.mockResolvedValue(UNAVAILABLE);
    const { result } = renderEngine();

    await waitFor(() => expect(result.current.ready.isSuccess).toBe(true));
    expect(result.current.ready.data).toBeNull();
  });
});
