// Mock dependencies first, before imports
jest.mock('./webLLMClient');
jest.mock('./configStore');
jest.mock('@mlc-ai/web-llm', () => ({
  CreateExtensionServiceWorkerMLCEngine: jest.fn(),
  prebuiltAppConfig: {}
}));

import { IntentDetector, FunctionCallingResult } from './intentDetector';
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
      const singleStepResponse = '{"intent": "action", "function": "search_threads", "confidence": 0.8, "reasoning": "Search for discussions", "parameters": {"keyword_filter": "AI"}}';
      
      mockWebLLMClient.chat.mockImplementation(({ onFinish }) => {
        setTimeout(() => onFinish?.(singleStepResponse), 0);
        return Promise.resolve();
      });

      const result = await IntentDetector.detectIntent('find AI discussions');

      expect(result).toEqual({
        intent: 'action',
        functionCall: {
          name: 'search_threads',
          parameters: { keyword_filter: 'AI' }
        },
        confidence: 0.8,
        reasoning: 'Intent: Search for discussions; Function: Search for discussions'
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

      const result = await IntentDetector.detectIntent('summarize this thread', pinnedThread);

      expect(result).toEqual({
        intent: 'action',
        functionCall: {
          name: 'summarize_pinned_thread',
          parameters: { format: 'paragraph' }
        },
        confidence: 0.9,
        reasoning: 'Intent: Process pinned thread; Function: Process pinned thread'
      });
    });

    test('calls webLLMClient properly for two-step approach', async () => {
      // Return 'action' for first step to trigger second step
      const intentResponse = '{"intent": "action", "confidence": 0.8}';
      const actionResponse = '{"function": "search_threads", "parameters": {"keyword_filter": "test"}, "confidence": 0.8}';
      
      let callCount = 0;
      mockWebLLMClient.chat.mockImplementation(({ onFinish }) => {
        setTimeout(() => {
          callCount++;
          if (callCount === 1) {
            onFinish?.(intentResponse); // First call: intent detection
          } else {
            onFinish?.(actionResponse); // Second call: action selection  
          }
        }, 0);
        return Promise.resolve();
      });

      await IntentDetector.detectIntent('test message');

      // Should be called twice for two-step approach (intent detection + action selection)
      expect(mockWebLLMClient.chat).toHaveBeenCalledTimes(2);
      
      // Verify basic structure of calls
      expect(mockWebLLMClient.chat).toHaveBeenCalledWith(
        expect.objectContaining({
          config: expect.objectContaining({
            model: 'test-model',
            temperature: 0.1,
            stream: true
          }),
          onUpdate: expect.any(Function),
          onFinish: expect.any(Function),
          onError: expect.any(Function)
        })
      );
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
        reasoning: 'Failed to parse LLM response'
      });
    });


    test('works without UI callbacks (pure business logic)', async () => {
      const intentResponse = '{"intent": "chat", "confidence": 0.9}';
      
      mockWebLLMClient.chat.mockImplementation(({ onUpdate, onFinish }) => {
        setTimeout(() => {
          onUpdate?.('', '');  // Empty parameters for compatibility
          onFinish?.(intentResponse);
        }, 0);
        return Promise.resolve();
      });

      const result = await IntentDetector.detectIntent('test message');

      expect(result).toEqual({
        intent: 'chat',
        functionCall: {
          name: 'no_action',
          parameters: { response_type: 'explanation' }
        },
        confidence: 0.9,
        reasoning: 'Classified as conversational'
      });
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

      await IntentDetector.detectIntent('test message', pinnedThread);

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