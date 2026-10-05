/**
 * Asks the app to summarise one HN thread, right now.
 *
 * Every thread card already carries a summary from the backend, but it was
 * written once when the pipeline first analysed the thread and never revised —
 * one measured card had a summary describing 2 comments of a 151-comment
 * discussion. So a summary the user asks for is generated fresh, from the
 * comments as they stand, by the local model.
 *
 * Slow the first time: fetching the thread takes a second and the model takes
 * another twenty. The caller must show that it is working.
 *
 * There is no cache here any more. There used to be a `Map` in this file, which
 * worked for the modal and was invisible to everything else — the agent runs in
 * Rust and cannot see a JavaScript object, so asking the chat about a thread
 * just read here would summarise it all over again. The cache now lives in
 * `ThreadStore` on the Rust side, where both can reach it: whichever asks first
 * pays, and the other is instant, in either direction.
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

export async function summarizeThread(threadId: number): Promise<ThreadSummary> {
  const started = Date.now();
  const result = await invoke<ThreadSummary>('summarize_thread_command', { threadId });
  logger.chat(
    `[summary] thread ${threadId}: ${result.used}/${result.total} comments in ` +
    `${Math.round((Date.now() - started) / 1000)}s`
  );
  return result;
}
