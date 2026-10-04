/**
 * Asks the app to summarise one HN thread, right now.
 *
 * Every thread card already carries a summary from the backend, but it was
 * written once when the pipeline first analysed the thread and never revised —
 * one measured card had a summary describing 2 comments of a 151-comment
 * discussion. So a summary the user asks for is generated fresh, from the
 * comments as they stand, by the local model.
 *
 * Slow by nature: fetching the thread takes a couple of seconds and the model
 * takes another twenty or so. The caller must show that it is working.
 */
import { invoke } from '@tauri-apps/api/core';
import { logger } from '../utils/logger';

export interface ThreadSummary {
  summary: string;
  /** Comments actually read, and how many the thread has. These differ on a
   *  busy thread: the model's context window cannot hold all of it, so the
   *  shallow, main conversation is kept and deep tangents are dropped. The UI
   *  says so rather than implying the summary covers everything. */
  used: number;
  total: number;
}

/**
 * Summaries already produced this session, by thread id.
 *
 * Reopening a card should not spend twenty seconds regenerating what the user
 * just read. The staleness this feature exists to fix is measured in hours —
 * nothing moves in the minutes an app session lasts — so holding them until
 * restart costs nothing in freshness.
 */
const cache = new Map<number, ThreadSummary>();

export async function summarizeThread(threadId: number): Promise<ThreadSummary> {
  const cached = cache.get(threadId);
  if (cached) {
    logger.chat(`[summary] thread ${threadId}: from this session's cache`);
    return cached;
  }

  const started = Date.now();
  const result = await invoke<ThreadSummary>('summarize_thread_command', { threadId });
  logger.chat(
    `[summary] thread ${threadId}: ${result.used}/${result.total} comments in ` +
    `${Math.round((Date.now() - started) / 1000)}s`
  );
  cache.set(threadId, result);
  return result;
}
