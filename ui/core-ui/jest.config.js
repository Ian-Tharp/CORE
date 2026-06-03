/** @type {import('jest').Config} */
module.exports = {
  preset: 'jest-preset-angular',
  testEnvironment: 'jsdom',
  setupFilesAfterEnv: ['<rootDir>/setup-jest.ts'],
  globalSetup: 'jest-preset-angular/global-setup',
  testMatch: ['**/?(*.)+(spec).ts'],
  moduleFileExtensions: ['ts', 'html', 'js', 'json'],
  transform: {
    '^.+\\.(ts|mjs|html)$': [
      'jest-preset-angular',
      {
        tsconfig: '<rootDir>/tsconfig.spec.json',
        stringifyContentPathRegex: '\\.(html)$'
      }
    ]
  },
  transformIgnorePatterns: ['node_modules/(?!.*)'],
  testPathIgnorePatterns: ['/node_modules/', '/dist/', '/e2e/'],
  // A pre-existing service leaves an open handle (timer/socket) after specs
  // finish, so Jest hangs at exit without this — which would stall CI
  // (`npm test` runs no --forceExit). forceExit lets the run terminate once
  // all tests complete. TODO: track down the leak via --detectOpenHandles.
  forceExit: true
};


