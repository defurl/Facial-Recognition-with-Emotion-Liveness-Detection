import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import App from './App';

const mockResponse = (body: unknown) =>
  Promise.resolve({
    ok: true,
    json: async () => body,
  } as Response);

beforeAll(() => {
  const stream = {
    getTracks: () => [],
  } as unknown as MediaStream;
  Object.defineProperty(navigator, 'mediaDevices', {
    value: {
      getUserMedia: () => Promise.resolve(stream),
    },
    configurable: true,
  });
});

beforeEach(() => {
  jest.spyOn(global, 'fetch').mockImplementation((input) => {
    const url =
      typeof input === 'string'
        ? input
        : input instanceof Request
        ? input.url
        : "";
    if (url.endsWith('/health')) return mockResponse({ status: 'ok' });
    if (url.endsWith('/threshold')) return mockResponse({ threshold: 0.45 });
    if (url.endsWith('/employees')) return mockResponse([{ name: 'Test User' }]);
    return mockResponse({});
  });
});

afterEach(() => {
  jest.restoreAllMocks();
});

test('renders verification dashboard heading', async () => {
  render(<App />);
  await waitFor(() => screen.getByText(/verification dashboard/i));
  expect(screen.getByText(/verification dashboard/i)).toBeInTheDocument();
});
