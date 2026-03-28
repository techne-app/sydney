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

// WKWebView: dragstart fires but macOS swallows everything after (drop/dragend/mouseup).
// Fix: backup dataTransfer data, cancel native drag, use pointer events + ghost clone instead.

// Backup clipboard — browser's dataTransfer returns empty on synthetic drops.
const dataTransferCache = new Map<string, string>();
const _origSetData = DataTransfer.prototype.setData;
DataTransfer.prototype.setData = function (type: string, data: string) {
  dataTransferCache.set(type, data);
  _origSetData.call(this, type, data);
};
const _origGetData = DataTransfer.prototype.getData;
DataTransfer.prototype.getData = function (type: string) {
  return _origGetData.call(this, type) || dataTransferCache.get(type) || '';
};

// Suppress text selection on draggable cards during drag.
(function injectDragStyles() {
  const s = document.createElement('style');
  s.textContent = '.wkwebview-dragging,[draggable="true"]{-webkit-user-select:none;user-select:none;}';
  document.head.appendChild(s);
})();

let _ghost: HTMLElement | null = null;
let _offX = 0, _offY = 0;

function _removeGhost() {
  if (_ghost) { _ghost.remove(); _ghost = null; }
  document.body.classList.remove('wkwebview-dragging');
}

// Clone card on press (invisible) — ready to show the moment drag is confirmed.
document.addEventListener('pointerdown', (e: PointerEvent) => {
  const el = (e.target as Element).closest('[draggable="true"]') as HTMLElement | null;
  if (!el) return;
  const r = el.getBoundingClientRect();
  _offX = e.clientX - r.left; // cursor offset inside the card
  _offY = e.clientY - r.top;
  _ghost = el.cloneNode(true) as HTMLElement;
  Object.assign(_ghost.style, {
    position: 'fixed', left: `${r.left}px`, top: `${r.top}px`,
    width: `${r.width}px`, height: `${r.height}px`,
    opacity: '0', pointerEvents: 'none', zIndex: '9999',
    boxShadow: '0 4px 8px rgba(0,0,0,0.15)', transition: 'none',
  });
  document.body.appendChild(_ghost);
}, true);

// Cancel native drag (keeps pointer events alive), reveal ghost.
document.addEventListener('dragstart', (e: DragEvent) => {
  e.preventDefault();
  document.body.classList.add('wkwebview-dragging');
  if (_ghost) _ghost.style.opacity = '0.75';
}, true);

// Move ghost with cursor.
document.addEventListener('pointermove', (e: PointerEvent) => {
  if (_ghost) { _ghost.style.left = `${e.clientX - _offX}px`; _ghost.style.top = `${e.clientY - _offY}px`; }
}, true);

// Release: remove ghost, fire synthetic drop — bubbles up to React's onDrop handler.
document.addEventListener('pointerup', (e: PointerEvent) => {
  _removeGhost();
  const cached = dataTransferCache.get('application/json');
  if (!cached) return;
  const target = document.elementFromPoint(e.clientX, e.clientY);
  if (!target) return;
  try {
    let dt: DataTransfer | undefined;
    try { dt = new DataTransfer(); } catch {}
    target.dispatchEvent(new DragEvent('drop', {
      bubbles: true, cancelable: true,
      clientX: e.clientX, clientY: e.clientY,
      dataTransfer: dt,
    } as any));
  } catch (err) {
    console.debug('[chrome-shim] synthetic drop failed:', err);
  } finally {
    dataTransferCache.clear();
  }
}, true);

document.addEventListener('pointercancel', () => { _removeGhost(); dataTransferCache.clear(); }, true);

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
          .then(() => dispatchMessage({ type: MessageType.TAGS_UPDATED, data: {} }))
          .catch((err: any) => console.debug('[chrome-shim] storeTag failed', err));
      }

      if (msg.type === MessageType.NEW_SEARCH && msg.data) {
        contextDb
          .storeSearch(msg.data.query)
          .then(() => dispatchMessage({ type: MessageType.SEARCHES_UPDATED, data: {} }))
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
