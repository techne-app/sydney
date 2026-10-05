/**
 * What the user has done — threads opened, searches run — and who wants to know.
 *
 * This is deliberately small, and it replaces something much larger. The app was
 * once a Chrome extension, where a popup could not write to the database
 * directly: content scripts run in the page's origin, so the background service
 * worker owned IndexedDB and everything reached it by `chrome.runtime
 * .sendMessage`. The Tauri app has no service worker and no origin problem — the
 * window *is* the app — so a shim was standing in for a postman who was no
 * longer delivering anything.
 *
 * Four messages went through it: two writes, two "something changed" broadcasts.
 * They are the four functions below.
 */
import { contextDb } from './contextDb';
import { logger } from './logger';

/** Which list changed, so a view can ignore the one it does not show. */
export type ActivityKind = 'threads' | 'searches';

const listeners = new Set<(kind: ActivityKind) => void>();

/**
 * Called whenever the recorded activity changes. Returns the unsubscribe, so a
 * React effect can hand it straight back:
 *
 *     useEffect(() => onActivity(kind => { ... }), []);
 */
export function onActivity(listener: (kind: ActivityKind) => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function announce(kind: ActivityKind): void {
  // forEach rather than for..of: the build targets ES5, where iterating a Set
  // needs downlevelIteration.
  listeners.forEach(listener => {
    try {
      listener(kind);
    } catch (error) {
      // One view throwing must not stop the others being told. The old message
      // bus gave us this for free; doing it by hand means doing it here.
      logger.debug('[activity] a listener threw', error);
    }
  });
}

/**
 * A thread the user opened on HN.
 *
 * Stored under the thread's *theme* rather than the link's text, which is why
 * every caller has to go through here: a thread link that records itself by
 * what it says fills Memory with rows called "Join the thread - 10 comments".
 */
export async function recordVisitedThread(theme: string, anchor: string): Promise<void> {
  try {
    await contextDb.storeTag(theme, 'visited_thread', anchor);
    announce('threads');
  } catch (error) {
    logger.debug('[activity] could not store the visited thread', error);
  }
}

/** A search the agent ran, for the Memory view's archive. */
export async function recordSearch(query: string): Promise<void> {
  try {
    await contextDb.storeSearch(query);
    announce('searches');
  } catch (error) {
    logger.debug('[activity] could not store the search', error);
  }
}
