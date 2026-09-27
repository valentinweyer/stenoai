import { beforeEach, describe, expect, test, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';

const diarization = vi.hoisted(() => ({
  engine: { data: 'sortformer' as string | undefined, isError: false, refetch: vi.fn() },
  modelsReady: { data: true as boolean | null | undefined },
  mutation: {
    mutate: vi.fn(),
    isPending: false,
    isError: false,
    variables: undefined as string | undefined,
  },
}));

vi.mock('@/hooks/useSettings', () => ({
  useIdentityMatchingEnabledSetting: () => ({ data: false }),
  useSetIdentityMatchingEnabled: () => ({ mutate: vi.fn() }),
}));

vi.mock('@/hooks/useModels', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/hooks/useModels')>()),
  useDiarizationEngine: () => diarization.engine,
  useDiarizationModelsReady: () => diarization.modelsReady,
  useSetDiarizationEngine: () => diarization.mutation,
}));

import { DiarizationEngineSetting, SpeakerIdentificationSetting } from './AiTab';

describe('Speaker identification setting', () => {
  test('describes itself to assistive technology', () => {
    render(<SpeakerIdentificationSetting />);

    // The wiring is what this guards: the switch must point at a description
    // element that actually exists and carries text. The wording itself is
    // asserted by the copy inventory, not here, so a copy edit doesn't have to
    // touch this test to stay honest.
    const toggle = screen.getByRole('switch', { name: 'Speaker identification' });
    const descriptionId = toggle.getAttribute('aria-describedby');
    expect(descriptionId).toBe('speaker-identification-description');
    const description = document.getElementById(descriptionId!);
    expect(description).not.toBeNull();
    expect(description!.textContent?.trim()).toBe('Optional and off by default.');
  });
});

describe('Speaker detection setting', () => {
  beforeEach(() => {
    diarization.engine = { data: 'sortformer', isError: false, refetch: vi.fn() };
    diarization.modelsReady = { data: true };
    diarization.mutation = {
      mutate: vi.fn(),
      isPending: false,
      isError: false,
      variables: undefined,
    };
  });

  function describedText() {
    const trigger = screen.getByRole('combobox', { name: 'Speaker detection' });
    const descriptionId = trigger.getAttribute('aria-describedby');
    expect(descriptionId).toBe('diarization-engine-description');
    return { trigger, text: document.getElementById(descriptionId!)?.textContent ?? '' };
  }

  test('shows the saved engine and describes itself', () => {
    render(<DiarizationEngineSetting />);

    const { trigger, text } = describedText();
    expect(trigger.textContent).toContain('Standard');
    expect((trigger as HTMLButtonElement).disabled).toBe(false);
    expect(text.trim().length).toBeGreaterThan(0);
  });

  test('explains a failed setting read and retries it', () => {
    diarization.engine = { data: undefined, isError: true, refetch: vi.fn() };
    render(<DiarizationEngineSetting />);

    const { trigger, text } = describedText();
    expect((trigger as HTMLButtonElement).disabled).toBe(true);
    expect(text).toContain('Could not load the speaker detection setting');
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
    expect(diarization.engine.refetch).toHaveBeenCalledOnce();
  });

  test('locks the picker and says so while Nemotron 3 downloads', () => {
    diarization.mutation = { ...diarization.mutation, isPending: true, variables: 'nemotron3' };
    render(<DiarizationEngineSetting />);

    const { trigger, text } = describedText();
    expect((trigger as HTMLButtonElement).disabled).toBe(true);
    expect(trigger.textContent).toContain('Nemotron 3');
    expect(text).toContain('Downloading');
  });

  test('reports a failed switch against the still-active engine', () => {
    diarization.mutation = { ...diarization.mutation, isError: true, variables: 'nemotron3' };
    render(<DiarizationEngineSetting />);

    const { trigger, text } = describedText();
    expect(trigger.textContent).toContain('Standard');
    expect(text).toContain('previous model is still active');
  });

  test('offers a download when the saved engine\'s models are missing', () => {
    diarization.modelsReady = { data: false };
    render(<DiarizationEngineSetting />);

    const { text } = describedText();
    expect(text).toContain('models are unavailable');
    const download = screen.getByRole('button', { name: 'Download' });
    expect(download.getAttribute('aria-describedby')).toBe('diarization-engine-description');
    fireEvent.click(download);
    expect(diarization.mutation.mutate).toHaveBeenCalledWith('sortformer');
  });

  test('offers no download when readiness is unknown or the models are present', () => {
    diarization.modelsReady = { data: null };
    const { unmount } = render(<DiarizationEngineSetting />);
    expect(screen.queryByRole('button', { name: 'Download' })).toBeNull();
    unmount();

    diarization.modelsReady = { data: true };
    render(<DiarizationEngineSetting />);
    expect(screen.queryByRole('button', { name: 'Download' })).toBeNull();
  });

  test('describes a Standard download while it runs', () => {
    diarization.modelsReady = { data: false };
    diarization.mutation = { ...diarization.mutation, isPending: true, variables: 'sortformer' };
    render(<DiarizationEngineSetting />);

    const { text } = describedText();
    expect(text).toContain('Downloading the speaker detection models');
    expect(screen.queryByRole('button', { name: 'Download' })).toBeNull();
  });

  test('a failed Download says the download failed and keeps the button', () => {
    diarization.modelsReady = { data: false };
    diarization.mutation = { ...diarization.mutation, isError: true, variables: 'sortformer' };
    render(<DiarizationEngineSetting />);

    const { text } = describedText();
    expect(text).toContain('Could not download the models');
    expect(text).not.toContain('previous model');
    expect(screen.getByRole('button', { name: 'Download' })).toBeTruthy();
  });
});
