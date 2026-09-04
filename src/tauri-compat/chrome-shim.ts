/**
 * Chrome API shim for Tauri.
 *
 * In Chrome, the popup communicates with the background script via chrome.runtime.
 * In Tauri there is no background script — this shim replaces chrome.runtime so
 * existing popup components don't crash.
 *
 * - onMessage.addListener / removeListener → tracks listeners
 * - sendMessage(NEW_TAG)          → calls contextDb.storeTag() directly
 * - sendMessage(NEW_SEARCH)       → calls contextDb.storeSearch() directly
 *
 * Installed in entry.tsx before React renders.
 */

import { contextDb } from '../background/contextDb';
import { MessageType } from '../types/messages';
// OLD — only the removed TAG_MATCH_REQUEST handler used these. Leaving them
// imported would keep transformers.js and onnxruntime-web in the webview bundle
// for no reason, which is most of the point of moving search to the sidecar.
// import { embed_tags } from './embed';
// import { computeTensorSimilarity } from '../background/personalize';
// import * as ort from 'onnxruntime-web';

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

// A thread the user actually opened, wherever the link came from.
const HN_ITEM_URL = /^https?:\/\/news\.ycombinator\.com\/item\?id=/;

function recordVisit(anchor: HTMLAnchorElement) {
  // Cards record their own visit (ThreadCard has the theme to hand, which makes
  // a better label than link text) and mark themselves so we don't double-store.
  if (anchor.dataset.visitRecorded === 'true') return;
  if (!HN_ITEM_URL.test(anchor.href)) return;

  const label = (anchor.textContent || '').trim();
  if (!label) return;

  contextDb
    .storeTag(label, 'visited_thread', anchor.href)
    .then(() => dispatchMessage({ type: MessageType.TAGS_UPDATED, data: {} }))
    .catch((err: any) => console.debug('[chrome-shim] storeTag failed', err));
}

// Opening external links is tauri-plugin-shell's job — its injected script
// already catches target="_blank" clicks and calls `plugin:shell|open`, so the
// custom open_external_url command was doing the same work a second time. That
// duplicate is what produced "shell.open not allowed" on every click: the
// plugin fired too and was denied, since capabilities granted only core:default.
//
// We now grant shell:allow-open and let the plugin open the link — which also
// means the URL goes through its scope validation rather than straight into
// `open` with no checks. This listener only records the visit.
//
// Deliberately no preventDefault/stopPropagation: the plugin's listener sits on
// body in the bubble phase, and React binds at the root container, so stopping
// propagation here would also kill ThreadCard's own click handler.
document.addEventListener('click', (e) => {
  const anchor = (e.target as HTMLElement).closest('a');
  if (anchor && anchor.target === '_blank' && anchor.href) {
    // Every external link passes through here, so this is the one place that
    // sees thread opens from chat search results as well as from the sidebar.
    recordVisit(anchor as HTMLAnchorElement);
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

      /* --- OLD in-webview semantic matching (MiniLM + cosine over ~30 themes).
       * Search now runs in the Python sidecar over 30 days of pre-embedded
       * threads — see src/tauri-compat/searchClient.ts and sidecar/corpus.py.
       * This was the sole consumer of embed_tags/computeTensorSimilarity in the
       * Tauri build, so with it gone the webview loads no ML model at all.
       * (The Chrome extension build still uses them via background/index.ts for
       * RANK_TAGS — do not delete embed.js or personalize.ts.)
       * Kept for migration reference; delete once the sidecar path is proven.
       *
       * if (msg.type === MessageType.TAG_MATCH_REQUEST && msg.data) {
       *   (async () => {
       *     try {
       *       const { inputText, tags } = msg.data;
       *       if (!inputText || !tags || tags.length === 0) {
       *         dispatchMessage({
       *           type: MessageType.TAG_MATCH_RESPONSE,
       *           data: { matches: [], error: 'Invalid input or no tags available' },
       *         });
       *         return;
       *       }
       *       const inputEmbedding = await embed_tags([inputText]);
       *       const tagTexts = tags.map((t: { tag: string }) => t.tag);
       *       const tagEmbeddings = await embed_tags(tagTexts);
       *       const matches = [];
       *       for (let i = 0; i < tagTexts.length; i++) {
       *         const inputTensor = new ort.Tensor('float32',
       *           new Float32Array(inputEmbedding[0].data), inputEmbedding[0].dims);
       *         const tagTensor = new ort.Tensor('float32',
       *           new Float32Array(tagEmbeddings[i].data), tagEmbeddings[i].dims);
       *         const similarity = await computeTensorSimilarity(inputTensor, tagTensor);
       *         matches.push({ tag: tags[i].tag, type: tags[i].type,
       *                        anchor: tags[i].anchor, score: similarity });
       *       }
       *       const topMatches = matches.sort((a, b) => b.score - a.score).slice(0, 3);
       *       dispatchMessage({
       *         type: MessageType.TAG_MATCH_RESPONSE, data: { matches: topMatches },
       *       });
       *     } catch (err) {
       *       console.debug('[chrome-shim] TAG_MATCH_REQUEST failed:', err);
       *       dispatchMessage({
       *         type: MessageType.TAG_MATCH_RESPONSE,
       *         data: { matches: [], error: String(err) },
       *       });
       *     }
       *   })();
       * }
       */
    },
  },
};
