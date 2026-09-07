/**
 * Talks to the sidecar's agent.
 *
 * One endpoint serves every conversation; the session id in the path selects
 * which history to load. The Dexie conversation id IS the session id, so a
 * conversation and an agent session are the same thing seen from two sides —
 * Dexie holds the rendered prose the UI displays, the agent's SQLite store
 * holds the structure it reasons over.
 *
 * Replaces routeClient + ToolOrchestrator: the agent runs its own tool loop,
 * so the frontend no longer decides what to execute or replays tool output
 * back as conversation history.
 */
import { logger } from '../utils/logger';
import { ThreadCardData } from '../types/chat';

const SIDECAR_URL = 'http://localhost:8000';

export interface SearchHit {
  thread_id: number;
  story_id: number;
  story_title: string;
  theme: string;
  category: string;
  anchor: string;
  score: number;
}

export interface AgentResponse {
  reply: string;
  tool_calls?: Array<{ name: string; arguments: Record<string, any> }>;
  /** Raw tool output. Links are rendered from this, never from `reply`. */
  results?: SearchHit[];
  error?: string;
}

class SessionClient {
  /**
   * Send one message to a conversation's agent session.
   *
   * `pinnedThread` is what the user currently has open, or null. It goes into
   * the agent's session state rather than the prompt, so it persists with the
   * conversation and the model always sees the current context.
   */
  async send(
    sessionId: string,
    message: string,
    pinnedThread?: ThreadCardData | null
  ): Promise<AgentResponse> {
    const response = await fetch(
      `${SIDECAR_URL}/sessions/${encodeURIComponent(sessionId)}/messages`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          pinned_thread: pinnedThread
            ? {
                story_title: pinnedThread.story_title,
                theme: pinnedThread.theme,
                category: pinnedThread.category,
                comment_count: pinnedThread.comment_count,
                summary: pinnedThread.summary,
              }
            : null,
        }),
      }
    );

    if (!response.ok) {
      throw new Error(`Sidecar returned ${response.status}`);
    }

    const data = (await response.json()) as AgentResponse;
    logger.chat('[sessionClient] tools:', data.tool_calls?.length ?? 0,
                '| results:', data.results?.length ?? 0);
    return data;
  }
}

export const sessionClient = new SessionClient();
