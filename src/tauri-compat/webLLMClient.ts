/**
 * Tauri-specific WebLLM client.
 * Replaces the Chrome extension WebLLM client (webLLMClient.ts).
 * Instead of talking to the Chrome service worker, this calls the Python sidecar.
 *
 * Vite aliases ../../utils/webLLMClient → this file for the Tauri build.
 * The original file is untouched and still used by the Chrome extension build.
 */

import { type ChatCompletionMessageParam } from "@mlc-ai/web-llm";
import { logger } from '../utils/logger';
import { modelState } from '../utils/modelState';

export interface ChatOptions {
  messages: ChatCompletionMessageParam[];
  config: {
    model: string;
    temperature: number;
    topP: number;
    maxTokens: number;
    stream: boolean;
  };
  onUpdate?: (message: string, chunk: string) => void;
  onFinish?: (message: string) => void;
  onError?: (error: string) => void;
}

const SIDECAR_URL = 'http://localhost:8000';

class TauriWebLLMClient {
  private ready = false;

  async chat(options: ChatOptions): Promise<void> {
    // Signal to UI that we're working
    modelState.setLoading(true, 'Calling Python sidecar...');

    // Send full conversation history
    const messages = options.messages.map(m => ({
      role: m.role,
      content: typeof m.content === 'string' ? m.content : '',
    }));

    try {
      const response = await fetch(`${SIDECAR_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages }),
      });

      if (!response.ok) {
        throw new Error(`Sidecar returned ${response.status}`);
      }

      const data = await response.json();
      const reply = data.reply as string;

      this.ready = true;
      modelState.setLoaded(true);

      // Call the same callbacks ChatInterface expects
      options.onUpdate?.(reply, reply);
      options.onFinish?.(reply);

    } catch (err: any) {
      const msg = err.message || 'Could not reach Python sidecar';
      logger.error('[TauriWebLLMClient] Error:', msg);
      modelState.setLoading(false, `Error: ${msg}`);
      options.onError?.(msg);
    }
  }

  isModelLoaded(): boolean {
    return this.ready;
  }

  async reset(): Promise<void> {
    this.ready = false;
    modelState.setLoaded(false);
  }

  getModelName(): string {
    return 'Python Sidecar';
  }
}

export const webLLMClient = new TauriWebLLMClient();
