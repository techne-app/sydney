// Mock dependencies first, before imports
jest.mock('./webLLMClient');
jest.mock('./configStore');
jest.mock('@mlc-ai/web-llm', () => ({
  CreateExtensionServiceWorkerMLCEngine: jest.fn(),
  prebuiltAppConfig: {}
}));

import { IntentDetector } from './intentDetector';
import { webLLMClient } from './webLLMClient';
import { configStore } from './configStore';

const mockWebLLMClient = webLLMClient as jest.Mocked<typeof webLLMClient>;
const mockConfigStore = configStore as jest.Mocked<typeof configStore>;

describe('IntentDetector', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockConfigStore.getConfig.mockResolvedValue({
      model: 'test-model',
      temperature: 0.7,
      topP: 0.95,
      maxTokens: 4096
    });
  });

  describe('parseIntentResponse', () => {
    // Access private method for testing
    const parseIntentResponse = (IntentDetector as any).parseIntentResponse;

    test('parses valid JSON response correctly', () => {
      const response = '{"isSearch": true, "searchQuery": "AI", "confidence": 0.9, "reasoning": "Clear search intent"}';
      const result = parseIntentResponse(response);

      expect(result).toEqual({
        isSearch: true,
        searchQuery: 'AI',
        confidence: 0.9,
        reasoning: 'Clear search intent'
      });
    });

    test('handles JSON embedded in text', () => {
      const response = 'Here is the result: {"isSearch": false, "confidence": 0.8} and some trailing text';
      const result = parseIntentResponse(response);

      expect(result).toEqual({
        isSearch: false,
        searchQuery: undefined,
        confidence: 0.8,
        reasoning: undefined
      });
    });

    test('validates isSearch field type', () => {
      const response = '{"isSearch": "true", "confidence": 0.8}';
      const result = parseIntentResponse(response);

      expect(result).toEqual({
        isSearch: false,
        confidence: 0.0,
        reasoning: 'Failed to parse LLM response'
      });
    });

    test('validates confidence field range', () => {
      const response = '{"isSearch": true, "confidence": 1.5}';
      const result = parseIntentResponse(response);

      expect(result).toEqual({
        isSearch: false,
        confidence: 0.0,
        reasoning: 'Failed to parse LLM response'
      });
    });

    test('validates confidence field type', () => {
      const response = '{"isSearch": true, "confidence": "high"}';
      const result = parseIntentResponse(response);

      expect(result).toEqual({
        isSearch: false,
        confidence: 0.0,
        reasoning: 'Failed to parse LLM response'
      });
    });

    test('handles missing JSON in response', () => {
      const response = 'This is just plain text with no JSON';
      const result = parseIntentResponse(response);

      expect(result).toEqual({
        isSearch: false,
        confidence: 0.0,
        reasoning: 'Failed to parse LLM response'
      });
    });

    test('handles invalid JSON syntax', () => {
      const response = '{"isSearch": true, "confidence": 0.8'; // Missing closing brace
      const result = parseIntentResponse(response);

      expect(result).toEqual({
        isSearch: false,
        confidence: 0.0,
        reasoning: 'Failed to parse LLM response'
      });
    });

    test('handles null searchQuery correctly', () => {
      const response = '{"isSearch": false, "searchQuery": null, "confidence": 0.9}';
      const result = parseIntentResponse(response);

      expect(result).toEqual({
        isSearch: false,
        searchQuery: undefined,
        confidence: 0.9,
        reasoning: undefined
      });
    });

    test('handles empty string searchQuery', () => {
      const response = '{"isSearch": true, "searchQuery": "", "confidence": 0.7}';
      const result = parseIntentResponse(response);

      expect(result).toEqual({
        isSearch: true,
        searchQuery: undefined,
        confidence: 0.7,
        reasoning: undefined
      });
    });

    test('preserves valid searchQuery', () => {
      const response = '{"isSearch": true, "searchQuery": "machine learning", "confidence": 0.85}';
      const result = parseIntentResponse(response);

      expect(result.searchQuery).toBe('machine learning');
    });

    test('handles edge case confidence values', () => {
      const response1 = '{"isSearch": true, "confidence": 0}';
      const result1 = parseIntentResponse(response1);
      expect(result1.confidence).toBe(0);

      const response2 = '{"isSearch": true, "confidence": 1}';
      const result2 = parseIntentResponse(response2);
      expect(result2.confidence).toBe(1);
    });
  });

  describe('buildIntentDetectionPrompt', () => {
    // Access private method for testing
    const buildIntentDetectionPrompt = (IntentDetector as any).buildIntentDetectionPrompt;

    test('includes user message in prompt', () => {
      const message = 'find AI discussions';
      const prompt = buildIntentDetectionPrompt(message);

      expect(prompt).toContain(`User message: "${message}"`);
    });

    test('includes JSON format specification', () => {
      const prompt = buildIntentDetectionPrompt('test message');

      expect(prompt).toContain('"isSearch": boolean');
      expect(prompt).toContain('"searchQuery": "extracted search terms or null"');
      expect(prompt).toContain('"confidence": number between 0 and 1');
      expect(prompt).toContain('"reasoning": "brief explanation"');
    });

    test('includes examples', () => {
      const prompt = buildIntentDetectionPrompt('test message');

      expect(prompt).toContain('find discussions about AI');
      expect(prompt).toContain('hello');
      expect(prompt).toContain('what about startups?');
    });
  });

  describe('detectSearchIntent', () => {
    test('successfully detects search intent', async () => {
      const mockResponse = '{"isSearch": true, "searchQuery": "React", "confidence": 0.9, "reasoning": "Clear search intent"}';
      
      mockWebLLMClient.chat.mockImplementation(({ onFinish }) => {
        setTimeout(() => onFinish?.(mockResponse), 0);
        return Promise.resolve();
      });

      const result = await IntentDetector.detectSearchIntent('find React discussions');

      expect(result).toEqual({
        isSearch: true,
        searchQuery: 'React',
        confidence: 0.9,
        reasoning: 'Clear search intent'
      });
    });

    test('calls webLLMClient with correct parameters', async () => {
      const mockResponse = '{"isSearch": false, "confidence": 0.8}';
      
      mockWebLLMClient.chat.mockImplementation(({ onFinish }) => {
        setTimeout(() => onFinish?.(mockResponse), 0);
        return Promise.resolve();
      });

      await IntentDetector.detectSearchIntent('hello there');

      expect(mockWebLLMClient.chat).toHaveBeenCalledWith({
        messages: [
          {
            role: 'user',
            content: expect.stringContaining('User message: "hello there"')
          }
        ],
        config: {
          model: 'test-model',
          temperature: 0.1,
          topP: 0.9,
          maxTokens: 200,
          stream: true
        },
        onUpdate: expect.any(Function),
        onFinish: expect.any(Function),
        onError: expect.any(Function)
      });
    });

    test('handles LLM errors gracefully', async () => {
      mockWebLLMClient.chat.mockImplementation(({ onError }) => {
        setTimeout(() => onError?.('Model failed to load'), 0);
        return Promise.resolve();
      });

      await expect(IntentDetector.detectSearchIntent('test message')).rejects.toThrow('Model failed to load');
    });

    test('calls callbacks for model loading', async () => {
      const mockResponse = '{"isSearch": true, "confidence": 0.9}';
      const onModelLoading = jest.fn();
      const onModelProgress = jest.fn();
      
      mockWebLLMClient.chat.mockImplementation(({ onUpdate, onFinish }) => {
        setTimeout(() => {
          onUpdate?.('', 'Loading... 50%');
          onFinish?.(mockResponse);
        }, 0);
        return Promise.resolve();
      });

      await IntentDetector.detectSearchIntent('test message', {
        onModelLoading,
        onModelProgress
      });

      expect(onModelLoading).toHaveBeenCalledWith(true);
      expect(onModelProgress).toHaveBeenCalledWith(0.5, 'Loading... 50%');
      expect(onModelLoading).toHaveBeenCalledWith(false);
    });
  });

  describe('isAvailable', () => {
    test('returns true when config is available', async () => {
      mockConfigStore.getConfig.mockResolvedValue({
        model: 'test-model',
        temperature: 0.7,
        topP: 0.95,
        maxTokens: 4096
      });

      const result = await IntentDetector.isAvailable();
      expect(result).toBe(true);
    });

    test('returns false when config throws error', async () => {
      mockConfigStore.getConfig.mockRejectedValue(new Error('Config unavailable'));

      const result = await IntentDetector.isAvailable();
      expect(result).toBe(false);
    });
  });
});