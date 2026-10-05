// Keep tests quiet. The logger writes through to the console otherwise, and a
// failing assertion is hard to find in a page of debug output.
//
// This file used to mock chrome.runtime, chrome.tabs, WebLLM and
// Transformers.js. All four left with the Chrome extension build: the app has
// no extension APIs to stand in for, and inference moved into Rust in #47.
jest.mock('./utils/logger', () => ({
  logger: {
    debug: jest.fn(),
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    search: jest.fn(),
    chat: jest.fn(),
    model: jest.fn(),
    database: jest.fn(),
    api: jest.fn(),
    intent: jest.fn(),
  },
}));
