/**
 * The things a WKWebView does not do the way a browser does.
 *
 * Two of them, both the kind of problem that only appears once the app is
 * running in a real window:
 *
 * - **Drag and drop.** `dragstart` fires and macOS then swallows everything
 *   after it, and `dataTransfer` comes back empty on a synthetic drop. So drags
 *   are tracked by hand, with their data cached and their ghost image drawn and
 *   torn down explicitly.
 * - **External links.** Opening one is `tauri-plugin-shell`'s job, through a
 *   listener it injects on `<body>`. We watch the same clicks only to record the
 *   visit, and must not interfere with them.
 *
 * This file used to also impersonate `chrome.runtime`, for components ported
 * from the Chrome extension — a message bus whose whole job was writing two rows
 * to IndexedDB. The extension lives in its own repo now, so those components
 * call `utils/activity` directly and the impersonation is gone.
 *
 * Installed from entry.tsx before React renders.
 */

import { recordVisitedThread } from '../utils/activity';

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
    console.debug('[webview] synthetic drop failed:', err);
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

  // Falling back to the link's own text, which is why a thread link that can do
  // better should record itself — see JoinThreadLink.
  const label = (anchor.textContent || '').trim();
  if (!label) return;

  recordVisitedThread(label, anchor.href);
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
