/**
 * Chrome API shim for Tauri.
 *
 * In Chrome, the popup communicates with the background script via chrome.runtime.
 * In Tauri there is no background script — this shim replaces chrome.runtime so
 * existing popup components don't crash.
 *
 * - onMessage.addListener / removeListener → tracks listeners so TAG_MATCH_RESPONSE can be dispatched
 * - sendMessage(NEW_TAG)          → calls contextDb.storeTag() directly
 * - sendMessage(NEW_SEARCH)       → calls contextDb.storeSearch() directly
 * - sendMessage(TAG_MATCH_REQUEST)→ runs embed_tags + cosine similarity, dispatches TAG_MATCH_RESPONSE
 *
 * Installed in entry.tsx before React renders.
 */

import { contextDb } from '../background/contextDb';
import { embed_tags } from './embed';
import { computeTensorSimilarity } from '../background/personalize';
import { MessageType } from '../types/messages';
import * as ort from 'onnxruntime-web';

// Track registered message listeners so responses can be dispatched back to callers
const messageListeners: ((message: any) => void)[] = [];

function dispatchMessage(message: any) {
  for (const listener of messageListeners) {
    try { listener(message); } catch {}
  }
}

// Intercept all target="_blank" link clicks and open them in the system browser via Rust
document.addEventListener('click', (e) => {
  const anchor = (e.target as HTMLElement).closest('a');
  if (anchor && anchor.target === '_blank' && anchor.href) {
    e.preventDefault();
    (window as any).__TAURI_INTERNALS__?.invoke('open_external_url', { url: anchor.href });
  }
}, true);

(window as any).chrome = {
  runtime: {
    onMessage: {
      addListener: (fn: (message: any) => void) => {
        messageListeners.push(fn);
      },
      removeListener: (fn: (message: any) => void) => {
        const idx = messageListeners.indexOf(fn);
        if (idx >= 0) messageListeners.splice(idx, 1);
      },
    },
    sendMessage: async (msg: any): Promise<void> => {
      if (!msg?.type) return;

      if (msg.type === MessageType.NEW_TAG && msg.data) {
        contextDb
          .storeTag(msg.data.tag, msg.data.type, msg.data.anchor)
          .catch((err: any) => console.debug('[chrome-shim] storeTag failed', err));
      }

      if (msg.type === MessageType.NEW_SEARCH && msg.data) {
        contextDb
          .storeSearch(msg.data.query)
          .catch((err: any) => console.debug('[chrome-shim] storeSearch failed', err));
      }

      if (msg.type === MessageType.TAG_MATCH_REQUEST && msg.data) {
        // Run same logic as background/index.ts registerTagMatchingListener
        // but directly in the Tauri webview (embed.js has no Chrome APIs)
        (async () => {
          try {
            const { inputText, tags } = msg.data;

            if (!inputText || !tags || tags.length === 0) {
              dispatchMessage({
                type: MessageType.TAG_MATCH_RESPONSE,
                data: { matches: [], error: 'Invalid input or no tags available' },
              });
              return;
            }

            // Compute embedding for the query
            const inputEmbedding = await embed_tags([inputText]);

            // Compute embeddings for all tags
            const tagTexts = tags.map((t: { tag: string }) => t.tag);
            const tagEmbeddings = await embed_tags(tagTexts);

            // Compute cosine similarity for each tag
            const matches = [];
            for (let i = 0; i < tagTexts.length; i++) {
              const inputTensor = new ort.Tensor(
                'float32',
                new Float32Array(inputEmbedding[0].data),
                inputEmbedding[0].dims
              );
              const tagTensor = new ort.Tensor(
                'float32',
                new Float32Array(tagEmbeddings[i].data),
                tagEmbeddings[i].dims
              );
              const similarity = await computeTensorSimilarity(inputTensor, tagTensor);
              matches.push({
                tag: tags[i].tag,
                type: tags[i].type,
                anchor: tags[i].anchor,
                score: similarity,
              });
            }

            // Sort by score descending, return top 3 (same as background script)
            const topMatches = matches
              .sort((a, b) => b.score - a.score)
              .slice(0, 3);

            dispatchMessage({
              type: MessageType.TAG_MATCH_RESPONSE,
              data: { matches: topMatches },
            });
          } catch (err) {
            console.debug('[chrome-shim] TAG_MATCH_REQUEST failed:', err);
            dispatchMessage({
              type: MessageType.TAG_MATCH_RESPONSE,
              data: { matches: [], error: String(err) },
            });
          }
        })();
      }
    },
  },
};
