import { describe, test, expect, beforeEach, vi } from 'vitest';
import * as React from 'react';
import { renderHook, act, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

/**
 * Meeting processing never downloads speaker models, so switching to a
 * non-default diarization engine must prepare its models BEFORE the choice is
 * saved -- otherwise every later meeting silently falls back to channel-only
 * "You"/"Others" labels. These pin that ordering and the failure path.
 */

const h = vi.hoisted(() => ({
  getEngine: vi.fn(),
  setEngine: vi.fn(),
  speakerModels: vi.fn(),
}));

vi.mock('@/lib/ipc', () => ({
  ipc: () => ({
    diarizationEngine: { get: h.getEngine, set: h.setEngine },
    setup: { speakerModels: h.speakerModels },
  }),
}));

import { useDiarizationEngine, useSetDiarizationEngine } from './useModels';

const READY = {
  success: true,
  ready: true,
  cache_directory: '/tmp/models',
  required_models: [],
  missing_models: [],
};

function renderEngine() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={qc}>{children}</QueryClientProvider>
  );
  return renderHook(
    () => ({ engine: useDiarizationEngine(), setEngine: useSetDiarizationEngine() }),
    { wrapper },
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
  });

  test('prepares Nemotron 3 models before saving the choice', async () => {
    h.speakerModels.mockResolvedValue(READY);
    h.setEngine.mockResolvedValue({ success: true, engine: 'nemotron3' });
    const { result } = renderEngine();
    await waitFor(() => expect(result.current.engine.data).toBe('sortformer'));

    await act(async () => {
      await result.current.setEngine.mutateAsync('nemotron3');
    });

    expect(h.speakerModels).toHaveBeenCalledWith('nemotron3');
    expect(h.speakerModels.mock.invocationCallOrder[0]).toBeLessThan(
      h.setEngine.mock.invocationCallOrder[0],
    );
    expect(h.setEngine).toHaveBeenCalledWith('nemotron3');
    await waitFor(() => expect(result.current.engine.data).toBe('nemotron3'));
  });

  test('a failed download never saves the engine and keeps the previous one', async () => {
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

  test('switching back to Sortformer needs no download', async () => {
    h.setEngine.mockResolvedValue({ success: true, engine: 'sortformer' });
    const { result } = renderEngine();
    await waitFor(() => expect(result.current.engine.data).toBe('sortformer'));

    await act(async () => {
      await result.current.setEngine.mutateAsync('sortformer');
    });

    expect(h.speakerModels).not.toHaveBeenCalled();
    expect(h.setEngine).toHaveBeenCalledWith('sortformer');
  });
});
