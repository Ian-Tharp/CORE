import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  timeout: 30_000,
  expect: {
    timeout: 7_500
  },
  use: {
    baseURL: 'http://127.0.0.1:4201',
    trace: 'on-first-retry'
  },
  webServer: {
    command: 'npm run build:ng -- --configuration development && node e2e/static-server.mjs',
    url: 'http://127.0.0.1:4201',
    reuseExistingServer: false,
    timeout: 240_000
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] }
    }
  ]
});
