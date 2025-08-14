import { webLLMClient } from './webLLMClient';
import { configStore } from './configStore';
import { logger } from './logger';
import { INTENT_ONLY_PROMPT } from '../prompts/intentOnly';
import { ACTION_ONLY_PROMPT } from '../prompts/actionOnly';
import { ThreadCardData } from '../types/chat';

export interface FunctionCall {
  name: string;
  parameters: Record<string, any>;
}

export interface FunctionCallingResult {
  intent: 'action' | 'chat';
  functionCall?: FunctionCall;
  confidence: number;
  reasoning?: string;
}


/**
 * Clean two-step intent detection using context-aware prompts.
 * 
 * Step 1: Intent classification (action vs chat)
 * Step 2: Function selection (when intent = action)
 * 
 * This implementation matches the evaluated approach that achieved 76.3% accuracy.
 */
export class IntentDetector {
  /**
   * Main intent detection method using two-step inference.
   * This is the only public API - all other approaches have been removed.
   * 
   * @param message - User message to analyze
   * @param pinnedThread - Optional pinned thread context
   * @returns Promise<FunctionCallingResult>
   */
  static async detectIntent(
    message: string,
    pinnedThread?: ThreadCardData | null
  ): Promise<FunctionCallingResult> {
    logger.model('Starting two-step intent detection for message:', message);
    
    // Step 1: Intent detection (action vs chat)
    const intentResult = await this._detectIntentOnly(message, pinnedThread);
    logger.intent('Step 1 - Intent detection result:', intentResult);
    
    if (intentResult.intent === 'chat') {
      // For chat intent, return no_action function call
      return {
        intent: 'chat',
        functionCall: {
          name: 'no_action',
          parameters: { response_type: 'explanation' }
        },
        confidence: intentResult.confidence,
        reasoning: intentResult.reasoning || 'Classified as conversational'
      };
    }
    
    // Step 2: Select function for action intent
    const actionResult = await this._detectActionOnly(message, pinnedThread);
    logger.intent('Step 2 - Action selection result:', actionResult);
    
    return {
      intent: 'action',
      functionCall: actionResult.functionCall,
      confidence: Math.min(intentResult.confidence, actionResult.confidence), // Take minimum confidence
      reasoning: `Intent: ${intentResult.reasoning || 'action'}; Function: ${actionResult.reasoning || 'selected'}`
    };
  }

  /**
   * Step 1: Detect intent only (action vs chat)
   * @private - Internal method for two-step process
   */
  private static async _detectIntentOnly(
    message: string,
    pinnedThread?: ThreadCardData | null
  ): Promise<{ intent: 'action' | 'chat'; confidence: number; reasoning?: string }> {
    const prompt = this._buildIntentOnlyPrompt(message, pinnedThread);
    const response = await this._queryLLM(prompt);
    return this._parseIntentOnlyResponse(response);
  }

  /**
   * Step 2: Select function for action intent (chat is not an option)
   * @private - Internal method for two-step process
   */
  private static async _detectActionOnly(
    message: string,
    pinnedThread?: ThreadCardData | null
  ): Promise<{ functionCall: FunctionCall; confidence: number; reasoning?: string }> {
    const prompt = this._buildActionOnlyPrompt(message, pinnedThread);
    const response = await this._queryLLM(prompt);
    return this._parseActionOnlyResponse(response);
  }

  /**
   * Build consistent context string for all prompts
   * @private - Internal utility method
   */
  private static _buildContextString(pinnedThread?: ThreadCardData | null): string {
    if (pinnedThread) {
      return `Context: User has pinned this thread:
- Title: "${pinnedThread.story_title}"
- Theme: "${pinnedThread.theme}"
- Category: "${pinnedThread.category}"
- Comments: ${pinnedThread.comment_count}
- Summary: "${pinnedThread.summary}"

`;
    } else {
      return `Context: No thread currently pinned.

`;
    }
  }

  /**
   * Build prompt for intent-only detection
   * @private - Internal method
   */
  private static _buildIntentOnlyPrompt(message: string, pinnedThread?: ThreadCardData | null): string {
    const contextInfo = this._buildContextString(pinnedThread);
    return INTENT_ONLY_PROMPT.replace('{context}', contextInfo).replace('{message}', message);
  }

  /**
   * Build prompt for action-only function selection
   * @private - Internal method
   */
  private static _buildActionOnlyPrompt(message: string, pinnedThread?: ThreadCardData | null): string {
    const contextInfo = this._buildContextString(pinnedThread);
    return ACTION_ONLY_PROMPT.replace('{context}', contextInfo).replace('{message}', message);
  }

  /**
   * Query LLM for intent detection
   * @private - Internal method
   */
  private static async _queryLLM(prompt: string): Promise<string> {
    const config = await configStore.getConfig();
    
    return new Promise((resolve, reject) => {
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
        onUpdate: () => {
          // Model loading progress handled by webLLMClient internally
        },
        onFinish: (finalMessage) => {
          resolve(finalMessage);
        },
        onError: (error) => {
          reject(new Error(error));
        }
      });
    });
  }

  /**
   * Parse LLM response for intent-only detection
   * @private - Internal method
   */
  private static _parseIntentOnlyResponse(response: string): { intent: 'action' | 'chat'; confidence: number; reasoning?: string } {
    try {
      // Extract JSON from response (in case there's extra text)
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        throw new Error('No JSON found in response');
      }

      const parsed = JSON.parse(jsonMatch[0]);
      
      // Validate required fields
      if (!parsed.intent || !['action', 'chat'].includes(parsed.intent)) {
        throw new Error('Invalid intent field');
      }
      
      if (typeof parsed.confidence !== 'number' || parsed.confidence < 0 || parsed.confidence > 1) {
        throw new Error('Invalid confidence field');
      }

      return {
        intent: parsed.intent,
        confidence: parsed.confidence,
        reasoning: parsed.reasoning || undefined
      };
    } catch (error) {
      logger.error('Error parsing intent-only response:', error);
      
      // Return fallback result
      return {
        intent: 'chat',
        confidence: 0.0,
        reasoning: 'Failed to parse LLM response'
      };
    }
  }

  /**
   * Parse LLM response for action-only function selection
   * @private - Internal method
   */
  private static _parseActionOnlyResponse(response: string): { functionCall: FunctionCall; confidence: number; reasoning?: string } {
    try {
      // Extract JSON from response (in case there's extra text)
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        throw new Error('No JSON found in response');
      }

      const parsed = JSON.parse(jsonMatch[0]);
      
      // Validate required fields
      if (!parsed.function || typeof parsed.function !== 'string') {
        throw new Error('Invalid function field');
      }
      
      if (typeof parsed.confidence !== 'number' || parsed.confidence < 0 || parsed.confidence > 1) {
        throw new Error('Invalid confidence field');
      }

      return {
        functionCall: {
          name: parsed.function,
          parameters: parsed.parameters || {}
        },
        confidence: parsed.confidence,
        reasoning: parsed.reasoning || undefined
      };
    } catch (error) {
      logger.error('Error parsing action-only response:', error);
      
      // Return fallback result
      return {
        functionCall: {
          name: 'search_threads',
          parameters: { keyword_filter: 'general discussion' }
        },
        confidence: 0.0,
        reasoning: 'Failed to parse LLM response, fallback to search'
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