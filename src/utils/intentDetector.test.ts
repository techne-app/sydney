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

  describe('detectIntent - Two-Step Approach', () => {
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

    test('correctly handles action intent with search function (two steps)', async () => {
      const intentResponse = '{"intent": "action", "confidence": 0.9, "reasoning": "Search request"}';
      const actionResponse = '{"function": "get_thread_cards", "confidence": 0.8, "reasoning": "Search for discussions", "parameters": {"keyword_filter": "AI"}}';
      
      let callCount = 0;
      mockWebLLMClient.chat.mockImplementation(({ onFinish }) => {
        setTimeout(() => {
          callCount++;
          onFinish?.(callCount === 1 ? intentResponse : actionResponse);
        }, 0);
        return Promise.resolve();
      });

      const result = await IntentDetector.detectIntent('find AI discussions');

      expect(result).toEqual({
        intent: 'action',
        functionCall: {
          name: 'get_thread_cards',
          parameters: { keyword_filter: 'AI' }
        },
        confidence: 0.8, // Minimum of both steps
        reasoning: 'Intent: Search request; Function: Search for discussions'
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

      const intentResponse = '{"intent": "action", "confidence": 0.95, "reasoning": "Summarize pinned thread"}';
      const actionResponse = '{"function": "summarize_pinned_thread", "confidence": 0.9, "reasoning": "Process pinned thread", "parameters": {"format": "paragraph"}}';
      
      let callCount = 0;
      mockWebLLMClient.chat.mockImplementation(({ onFinish }) => {
        setTimeout(() => {
          callCount++;
          onFinish?.(callCount === 1 ? intentResponse : actionResponse);
        }, 0);
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
        reasoning: 'Intent: Summarize pinned thread; Function: Process pinned thread'
      });
    });

    test('calls webLLMClient with correct parameters for both steps', async () => {
      const intentResponse = '{"intent": "action", "confidence": 0.9}';
      const actionResponse = '{"function": "get_thread_cards", "confidence": 0.8}';
      
      let callCount = 0;
      mockWebLLMClient.chat.mockImplementation(({ onFinish }) => {
        setTimeout(() => {
          callCount++;
          onFinish?.(callCount === 1 ? intentResponse : actionResponse);
        }, 0);
        return Promise.resolve();
      });

      await IntentDetector.detectIntent('test message');

      // Should be called twice - once for intent, once for action
      expect(mockWebLLMClient.chat).toHaveBeenCalledTimes(2);
      
      // Check first call (intent detection)
      expect(mockWebLLMClient.chat).toHaveBeenNthCalledWith(1, {
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
          maxTokens: 200,
          stream: true
        },
        onUpdate: expect.any(Function),
        onFinish: expect.any(Function),
        onError: expect.any(Function)
      });
    });

    test('handles step 1 parsing errors gracefully', async () => {
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
        reasoning: 'Failed to parse LLM response'
      });
    });

    test('handles step 2 parsing errors gracefully', async () => {
      const intentResponse = '{"intent": "action", "confidence": 0.9}';
      const invalidActionResponse = 'invalid json response';
      
      let callCount = 0;
      mockWebLLMClient.chat.mockImplementation(({ onFinish }) => {
        setTimeout(() => {
          callCount++;
          onFinish?.(callCount === 1 ? intentResponse : invalidActionResponse);
        }, 0);
        return Promise.resolve();
      });

      const result = await IntentDetector.detectIntent('test message');

      expect(result).toEqual({
        intent: 'action',
        functionCall: {
          name: 'get_thread_cards',
          parameters: { keyword_filter: 'general discussion' }
        },
        confidence: 0.0, // Fallback confidence from step 2 error
        reasoning: 'Intent: action; Function: Failed to parse LLM response, fallback to search'
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