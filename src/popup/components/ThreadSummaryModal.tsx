import React, { useEffect, useState } from 'react';
import { ExternalLink, X } from 'lucide-react';
import { ThreadCardData } from '../../types/chat';
import { summarizeThread } from '../../tauri-compat/summaryClient';
import { logger } from '../../utils/logger';
import { Tag, formatTimeAgo, JoinThreadLink } from './ThreadCard';

interface ThreadSummaryModalProps {
  thread: ThreadCardData | null;
  onClose: () => void;
}

/**
 * The card, opened out, with a summary written for it on the spot.
 *
 * Laid out to mirror ThreadCard exactly — same beige, same tag, same centred
 * theme, same two links — so it reads as the card expanding rather than as a
 * separate screen. That is the pattern the landing page uses for its own "read
 * more", and following it keeps the two surfaces recognisably the same product.
 *
 * The summary itself is generated here rather than taken from the card. The
 * card's stored summary was written when the pipeline first analysed the thread
 * and never revised: on a measured example it described 2 comments of a
 * 151-comment discussion.
 */
/**
 * The summary, laid out as the three things it actually is.
 *
 * The model returns an opening, two or three quotations from the discussion,
 * and a closing. Rendering that as one centred block wasted it: centred text
 * past two lines makes the eye hunt for each line start, and quotations set like
 * prose stop reading as somebody's actual words.
 *
 * Split by LINE, not by blank line. The model puts each quotation on its own
 * line but does not leave a blank one between them, so splitting on paragraphs
 * merged all three into a single italic block with stray quote marks still in
 * it — which looked exactly like the wall of text this replaced.
 *
 * Parsed rather than returned as structured fields because the model is more
 * reliable writing prose than JSON, and the shape is unambiguous: a line that
 * opens and closes with a quotation mark is a quotation. Anything else falls
 * through as a paragraph, so an unexpected answer still renders.
 */
const QUOTED = /^["\u201c]([\s\S]+)["\u201d]$/;

const SummaryBody: React.FC<{ text: string }> = ({ text }) => {
  const blocks: { quote: boolean; text: string }[] = [];

  for (const line of text.split('\n').map(l => l.trim()).filter(Boolean)) {
    const quoted = line.match(QUOTED);
    const previous = blocks[blocks.length - 1];
    if (quoted) {
      blocks.push({ quote: true, text: quoted[1] });
    } else if (previous && !previous.quote) {
      // A paragraph the model wrapped across lines, rather than a new one.
      previous.text += ' ' + line;
    } else {
      blocks.push({ quote: false, text: line });
    }
  }

  // Said once, above the first quotation. A rule and an italic say "set apart",
  // which the reader can just as easily take for the summary's own emphasis —
  // the one thing that makes these worth reading is that they are what people
  // actually wrote, and nothing on the page said so.
  const firstQuote = blocks.findIndex(b => b.quote);

  return (
    <div className="text-[#333] text-sm leading-6">
      {blocks.map((block, i) => (
        <React.Fragment key={i}>
          {i === firstQuote && (
            <div className="text-[#999] text-[11px] uppercase tracking-wider mt-4 mb-2">
              From the discussion
            </div>
          )}
          {block.quote ? (
            // The quotation marks stay. They are the one mark every reader
            // already knows means somebody said this, and stripping them left
            // the attribution resting entirely on a coloured rule.
            <blockquote className="my-2.5 pl-3 border-l-2 border-[#ff6600]/50 text-[#444]">
              &ldquo;{block.text}&rdquo;
            </blockquote>
          ) : (
            <p className="mb-3">{block.text}</p>
          )}
        </React.Fragment>
      ))}
    </div>
  );
};

export const ThreadSummaryModal: React.FC<ThreadSummaryModalProps> = ({ thread, onClose }) => {
  const [summary, setSummary] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!thread) return;

    // This takes long enough to close and open another card meanwhile, and
    // without the guard the first answer would land in the second modal.
    let active = true;
    setSummary(null);
    setError(null);

    summarizeThread(thread.id)
      .then(result => {
        if (active) setSummary(result.summary);
      })
      .catch(err => {
        logger.error('Thread summary failed:', err);
        if (active) setError(String(err));
      });

    return () => {
      active = false;
    };
  }, [thread?.id]);

  if (!thread) return null;

  return (
    <div
      className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      // Close only on the backdrop itself. The obvious alternative — stopping
      // propagation on the panel — silently breaks every link inside it:
      // external links are opened by tauri-plugin-shell's listener on <body>,
      // so a click that never bubbles there never opens anything.
      onClick={e => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="bg-[#f6f6ef] border border-[#e0e0e0] rounded-lg shadow-xl max-w-2xl w-full max-h-[80vh] flex flex-col">
        <div className="p-4">
          <div className="flex items-center justify-between mb-3 text-sm">
            <Tag label={thread.category} />
            <div className="flex items-center gap-2">
              <span className="text-[#999] text-xs">{formatTimeAgo(thread.updated_at)}</span>
              <button
                onClick={onClose}
                className="text-[#999] hover:text-[#333] p-1 transition-colors duration-200"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          <div className="border-b border-[#e0e0e0] mb-3" />

          <div className="mb-3 text-center">
            <h2 className="text-base font-semibold leading-tight" style={{ color: '#0066cc' }}>
              {thread.theme}
            </h2>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-4 min-h-[6rem]">
          {!summary && !error && (
            // Only reached the first time a thread is opened — after that the
            // client serves it from this session's cache — so the spinner marks
            // real work rather than flashing on an instant result.
            <div className="flex flex-col items-center gap-3 py-10">
              <div
                className="animate-spin rounded-full h-5 w-5 border-2"
                style={{ borderColor: '#e0e0e0', borderTopColor: '#0066cc' }}
              />
              <span className="text-[#999] text-sm">Reading the discussion…</span>
            </div>
          )}

          {error && (
            <div className="text-[#999] text-sm text-center py-6">
              Couldn't summarise this discussion: {error}
            </div>
          )}

          {summary && <SummaryBody text={summary} />}
        </div>

        <div className="p-4">
          <div className="mb-3 flex justify-center text-xs font-mono">
            <JoinThreadLink
              anchor={thread.anchor}
              theme={thread.theme}
              comment_count={thread.comment_count}
            />
          </div>

          <div className="border-b border-[#e0e0e0] mb-3" />

          <div className="text-center">
            <a
              href={thread.story_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[#0066cc] text-xs hover:underline inline-flex items-start gap-1 leading-tight"
            >
              🔗 <span>{thread.story_title}</span>
              <ExternalLink className="w-3 h-3 text-[#999] flex-shrink-0 mt-0.5" />
            </a>
          </div>
        </div>
      </div>
    </div>
  );
};
