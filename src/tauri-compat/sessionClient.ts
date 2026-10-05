/**
 * Talks to the agent in Tauri core.
 *
 * The conversation id IS the agent's session id, so a conversation and an agent
 * session are the same thing seen from two sides — Dexie holds the rendered
 * prose the UI displays, the agent's SQLite store holds the structure it
 * reasons over.
 *
 * The agent used to live in the Python sidecar and was reached over HTTP. It
 * now runs in Rust inside the app process, so this is a Tauri command rather
 * than a fetch — iOS forbids the subprocesses that design depended on.
 */
import { invoke } from '@tauri-apps/api/core';
import { logger } from '../utils/logger';
import { ThreadCardData } from '../types/chat';

/**
 * A thread as the agent's tools return it — see corpus.public_view. These are
 * deliberately not our database column names: the model echoes the keys it is
 * handed, so `theme` came back as "Theme:" in replies meant for a user who has
 * never heard the word. `link` is only present on get_thread results; a search
 * hit has no link, which is what stops the UI appending a second copy of the
 * list the model just wrote.
 */
export interface SearchHit {
  thread_id: number;
  title: string;
  /** Search hits only — the one-line description of what the thread is about. */
  about?: string;
  /** get_thread results only. The UI renders a link for exactly these. */
  link?: string;
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
    const data = await invoke<AgentResponse>('send_message', {
      sessionId,
      message,
      // The id is the important part: it is what lets the agent go and read the
      // discussion. This used to send `summary` instead — the card's stored one,
      // written once by the pipeline and never revised — which left the model
      // nothing to do but repeat it back, and no way to answer anything it did
      // not already cover. Title and theme stay because they orient the model
      // cheaply, before it decides whether reading is needed at all.
      pinnedThread: pinnedThread
        ? {
            id: pinnedThread.id,
            story_title: pinnedThread.story_title,
            theme: pinnedThread.theme,
          }
        : null,
    });

    logger.chat('[sessionClient] tools:', data.tool_calls?.length ?? 0,
                '| results:', data.results?.length ?? 0);
    return data;
  }
}

export const sessionClient = new SessionClient();
