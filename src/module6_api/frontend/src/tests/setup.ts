import '@testing-library/jest-dom';
import { afterEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';

// Cleanup DOM after each test
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

// Polyfill global fetch if needed
if (!globalThis.fetch) {
  globalThis.fetch = vi.fn();
}
