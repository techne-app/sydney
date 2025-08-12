// Mock dependencies first, before imports
jest.mock('./webLLMClient');
jest.mock('./configStore');
jest.mock('@mlc-ai/web-llm', () => ({
  CreateExtensionServiceWorkerMLCEngine: jest.fn(),
  prebuiltAppConfig: {}
}));

import { IntentDetector, FunctionCallingResult } from './intentDetectorSingleStep';
import { webLLMClient } from './webLLMClient';
import { configStore } from './configStore';
import { ThreadCardData } from '../types/chat';

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

  describe('detectIntent - Single-Step Approach', () => {
    test('correctly handles chat intent (single step)', async () => {
      const intentResponse = '{"intent": "chat", "confidence": 0.9, "reasoning": "Social greeting"}';
      
      mockWebLLMClient.chat.mockImplementation(({ onFinish }) => {
        setTimeout(() => onFinish?.(intentResponse), 0);
        return Promise.resolve();
      });

      const result = await IntentDetector.detectIntent('hello there');

      expect(result).toEqual({
        intent: 'chat',
        functionCall: {
          name: 'no_action',
          parameters: { response_type: 'explanation' }
        },
        confidence: 0.9,
        reasoning: 'Social greeting'
      });
    });

    test('correctly handles action intent with search function (single step)', async () => {
      const singleStepResponse = '{"intent": "action", "function": "get_thread_cards", "confidence": 0.8, "reasoning": "Search for discussions", "parameters": {"keyword_filter": "AI"}}';
      
      mockWebLLMClient.chat.mockImplementation(({ onFinish }) => {
        setTimeout(() => onFinish?.(singleStepResponse), 0);
        return Promise.resolve();
      });

      const result = await IntentDetector.detectIntent('find AI discussions');

      expect(result).toEqual({
        intent: 'action',
        functionCall: {
          name: 'get_thread_cards',
          parameters: { keyword_filter: 'AI' }
        },
        confidence: 0.8,
        reasoning: 'Search for discussions'
      });
    });

    test('correctly handles pinned thread summary action', async () => {
      const pinnedThread: ThreadCardData = {
        id: 1,
        cumulative_karma: 100,
        story_id: 12345,
        story_title: 'Test Story',
        theme: 'Test Theme',
        category: 'TEST',
        comment_count: 50,
        summary: 'Test summary',
        story_url: 'https://test.com',
        anchor: 'https://test.com#anchor',
        updated_at: '2023-01-01T00:00:00Z'
      };

      const singleStepResponse = '{"intent": "action", "function": "summarize_pinned_thread", "confidence": 0.9, "reasoning": "Process pinned thread", "parameters": {"format": "paragraph"}}';
      
      mockWebLLMClient.chat.mockImplementation(({ onFinish }) => {
        setTimeout(() => onFinish?.(singleStepResponse), 0);
        return Promise.resolve();
      });

      const result = await IntentDetector.detectIntent('summarize this thread', undefined, pinnedThread);

      expect(result).toEqual({
        intent: 'action',
        functionCall: {
          name: 'summarize_pinned_thread',
          parameters: { format: 'paragraph' }
        },
        confidence: 0.9,
        reasoning: 'Process pinned thread'
      });
    });

    test('calls webLLMClient with correct parameters for single step', async () => {
      const singleStepResponse = '{"intent": "action", "function": "get_thread_cards", "confidence": 0.8}';
      
      mockWebLLMClient.chat.mockImplementation(({ onFinish }) => {
        setTimeout(() => onFinish?.(singleStepResponse), 0);
        return Promise.resolve();
      });

      await IntentDetector.detectIntent('test message');

      // Should be called once for single-step approach
      expect(mockWebLLMClient.chat).toHaveBeenCalledTimes(1);
      
      // Check the call parameters
      expect(mockWebLLMClient.chat).toHaveBeenCalledWith({
        messages: [
          {
            role: 'user',
            content: expect.stringContaining('test message')
          }
        ],
        config: {
          model: 'test-model',
          temperature: 0.1,
          topP: 0.9,
          maxTokens: 300, // Single-step uses 300 tokens
          stream: true
        },
        onUpdate: expect.any(Function),
        onFinish: expect.any(Function),
        onError: expect.any(Function)
      });
    });

    test('handles parsing errors gracefully', async () => {
      const invalidResponse = 'invalid json response';
      
      mockWebLLMClient.chat.mockImplementation(({ onFinish }) => {
        setTimeout(() => onFinish?.(invalidResponse), 0);
        return Promise.resolve();
      });

      const result = await IntentDetector.detectIntent('test message');

      expect(result).toEqual({
        intent: 'chat',
        functionCall: {
          name: 'no_action',
          parameters: { response_type: 'explanation' }
        },
        confidence: 0.0,
        reasoning: 'Failed to parse LLM response, defaulting to chat'
      });
    });


    test('calls callbacks for model loading progress', async () => {
      const intentResponse = '{"intent": "chat", "confidence": 0.9}';
      const onModelLoading = jest.fn();
      const onModelProgress = jest.fn();
      
      mockWebLLMClient.chat.mockImplementation(({ onUpdate, onFinish }) => {
        setTimeout(() => {
          onUpdate?.('', 'Loading... 50%');
          onFinish?.(intentResponse);
        }, 0);
        return Promise.resolve();
      });

      await IntentDetector.detectIntent('test message', {
        onModelLoading,
        onModelProgress
      });

      expect(onModelLoading).toHaveBeenCalledWith(true);
      expect(onModelProgress).toHaveBeenCalledWith(0, 'Loading... 50%');
      expect(onModelLoading).toHaveBeenCalledWith(false);
    });

    test('includes pinned thread context in prompts', async () => {
      const pinnedThread: ThreadCardData = {
        id: 1,
        cumulative_karma: 100,
        story_id: 12345,
        story_title: 'Test Story',
        theme: 'Test Theme',
        category: 'TEST',
        comment_count: 50,
        summary: 'Test summary',
        story_url: 'https://test.com',
        anchor: 'https://test.com#anchor',
        updated_at: '2023-01-01T00:00:00Z'
      };

      const intentResponse = '{"intent": "chat", "confidence": 0.9}';
      
      mockWebLLMClient.chat.mockImplementation(({ onFinish }) => {
        setTimeout(() => onFinish?.(intentResponse), 0);
        return Promise.resolve();
      });

      await IntentDetector.detectIntent('test message', undefined, pinnedThread);

      // Check that the prompt includes pinned thread context
      expect(mockWebLLMClient.chat).toHaveBeenCalledWith(
        expect.objectContaining({
          messages: [
            {
              role: 'user',
              content: expect.stringContaining('Test Story')
            }
          ]
        })
      );
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