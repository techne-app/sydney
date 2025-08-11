import { webLLMClient } from './webLLMClient';
import { configStore } from './configStore';
import { logger } from './logger';
import { SEARCH_INTENT_PROMPT } from '../prompts/searchIntent';
import { ThreadCardData } from '../types/chat';

export interface IntentDetectionResult {
  isSearch: boolean;
  searchQuery?: string;
  confidence: number;
  reasoning?: string;
}

export interface IntentDetectionCallbacks {
  onModelLoading?: (isLoading: boolean) => void;
  onModelProgress?: (progress: number, text: string) => void;
}

export class IntentDetector {
  /**
   * Detect if a message contains search intent using LLM
   * @param message - User message to analyze
   * @param callbacks - Optional callbacks for loading state
   * @param pinnedThread - Optional pinned thread context
   * @returns Promise<IntentDetectionResult>
   */
  static async detectSearchIntent(
    message: string, 
    callbacks?: IntentDetectionCallbacks,
    pinnedThread?: ThreadCardData | null
  ): Promise<IntentDetectionResult> {
    logger.model('Starting intent detection for message:', message);
    const prompt = this.buildIntentDetectionPrompt(message, pinnedThread);
    logger.debug('Built intent detection prompt');
    const response = await this.queryLLM(prompt, callbacks);
    logger.debug('Raw LLM response:', response);
    const result = this.parseIntentResponse(response);
    logger.intent('Parsed intent result:', result);
    return result;
  }

  /**
   * Build prompt for intent detection with optional pinned thread context
   * @param message - User message
   * @param pinnedThread - Optional pinned thread data
   * @returns formatted prompt
   */
  private static buildIntentDetectionPrompt(message: string, pinnedThread?: ThreadCardData | null): string {
    let contextInfo = '';
    
    if (pinnedThread) {
      contextInfo = `Context: User has pinned this thread:
- Title: "${pinnedThread.story_title}"
- Theme: "${pinnedThread.theme}"
- Category: "${pinnedThread.category}"
- Comments: ${pinnedThread.comment_count}
- Summary: "${pinnedThread.summary}"

`;
    } else {
      contextInfo = `Context: No thread currently pinned.

`;
    }
    
    return contextInfo + SEARCH_INTENT_PROMPT.replace('{message}', message);
  }

  /**
   * Query LLM for intent detection
   * @param prompt - Intent detection prompt
   * @param callbacks - Optional callbacks for loading state
   * @returns LLM response
   */
  private static async queryLLM(prompt: string, callbacks?: IntentDetectionCallbacks): Promise<string> {
    const config = await configStore.getConfig();
    
    return new Promise((resolve, reject) => {
      // Signal model loading start
      callbacks?.onModelLoading?.(true);
      
      webLLMClient.chat({
        messages: [
          { role: 'user', content: prompt }
        ],
        config: {
          model: config.model,
          temperature: 0.1, // Low temperature for consistent structured output
          topP: 0.9,
          maxTokens: 200,
          stream: true
        },
        onUpdate: (_, chunk) => {
          // Handle model loading progress
          if (chunk && (chunk.includes('Loading') || chunk.includes('Initializing') || chunk.includes('%'))) {
            callbacks?.onModelProgress?.(0, chunk);
            // Try to extract progress percentage
            const progressMatch = chunk.match(/(\d+)%/);
            if (progressMatch) {
              callbacks?.onModelProgress?.(parseInt(progressMatch[1]) / 100, chunk);
            }
          } else {
            // Model loading complete when we get actual content
            callbacks?.onModelLoading?.(false);
          }
        },
        onFinish: (finalMessage) => {
          callbacks?.onModelLoading?.(false);
          resolve(finalMessage);
        },
        onError: (error) => {
          callbacks?.onModelLoading?.(false);
          reject(new Error(error));
        }
      });
    });
  }

  /**
   * Parse LLM response for intent detection
   * @param response - Raw LLM response
   * @returns IntentDetectionResult
   */
  private static parseIntentResponse(response: string): IntentDetectionResult {
    try {
      // Extract JSON from response (in case there's extra text)
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        throw new Error('No JSON found in response');
      }

      const parsed = JSON.parse(jsonMatch[0]);
      
      // Validate required fields
      if (typeof parsed.isSearch !== 'boolean') {
        throw new Error('Invalid isSearch field');
      }
      
      if (typeof parsed.confidence !== 'number' || parsed.confidence < 0 || parsed.confidence > 1) {
        throw new Error('Invalid confidence field');
      }

      return {
        isSearch: parsed.isSearch,
        searchQuery: parsed.searchQuery || undefined,
        confidence: parsed.confidence,
        reasoning: parsed.reasoning || undefined
      };
    } catch (error) {
      logger.error('Error parsing intent response:', error);
      
      // Return fallback result
      return {
        isSearch: false,
        confidence: 0.0,
        reasoning: 'Failed to parse LLM response'
      };
    }
  }


  /**
   * Check if intent detection is available
   * @returns boolean
   */
  static async isAvailable(): Promise<boolean> {
    try {
      await configStore.getConfig();
      return true;
    } catch {
      return false;
    }
  }
}