/**
 * Tauri intent router — talks to the Python sidecar's /route endpoint.
 *
 * Replaces the manual IntentDetector for the Tauri build: the sidecar LLM
 * (Qwen, native tool-calling) decides whether to call a tool or reply in chat.
 * See docs/specs/intent-shift.md and sidecar/main.py (/route).
 */
import { logger } from '../utils/logger';
import { ThreadCardData } from '../types/chat';

const SIDECAR_URL = 'http://localhost:8000';

export interface RouteMessage {
  role: string;
  content: string;
}

export interface RouteResult {
  type: 'tool_call' | 'chat';
  name?: string;
  arguments?: Record<string, any>;
  reply?: string;
}

class RouteClient {
  /**
   * Ask the sidecar to route a message: returns either a tool call (action) or
   * a conversational reply (chat). Throws on transport error so the caller can
   * fall back to the plain chat path.
   */
  async route(
    messages: RouteMessage[],
    pinnedThread?: ThreadCardData | null
  ): Promise<RouteResult> {
    const body: Record<string, any> = { messages };

    // Pinned-thread context, keys mirror sidecar _build_route_system_prompt().
    if (pinnedThread) {
      body.pinned_thread = {
        story_title: pinnedThread.story_title,
        theme: pinnedThread.theme,
        category: pinnedThread.category,
        comment_count: pinnedThread.comment_count,
        summary: pinnedThread.summary,
      };
    }

    const response = await fetch(`${SIDECAR_URL}/route`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      throw new Error(`Sidecar /route returned ${response.status}`);
    }

    const data = (await response.json()) as RouteResult;
    logger.intent('[routeClient] decision:', data);
    return data;
  }
}

export const routeClient = new RouteClient();
