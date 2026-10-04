import React, { useState, useEffect, useRef } from "react";
import { MessageCircle, ExternalLink } from "lucide-react";
import { MessageType, NewTagRequest } from "../../types/messages";
import { logger } from "../../utils/logger";

// ThreadCard data interface (matching the shared component)
interface ThreadCardData {
  id: number;
  cumulative_karma: number;
  comment_count: number;
  theme: string;
  category: string;
  story_id: number;
  story_title: string;
  story_url: string;
  anchor: string;
  summary: string;
  updated_at: string;
}

// Enhanced interface with height options
interface ThreadCardProps extends ThreadCardData {
  gradient?: string;
  height?: string | number;
  minHeight?: string | number;
  maxHeight?: string | number;
  className?: string;
  style?: React.CSSProperties;
  onClick?: () => void;
  /** Shown in place of the summary when the card has none, as an invitation to
   *  generate one. The sidebar passes `summary=""` deliberately — a full summary
   *  does not fit a 260px card — so without this the slot is simply empty. */
  onSummarize?: () => void;
  draggable?: boolean;
  onDragStart?: (e: React.DragEvent<HTMLDivElement>, data: ThreadCardData) => void;
}

// Simple local Card component
const Card: React.FC<{ 
  className?: string; 
  style?: React.CSSProperties; 
  children: React.ReactNode;
  onClick?: () => void;
  draggable?: boolean;
  onDragStart?: (e: React.DragEvent<HTMLDivElement>) => void;
}> = ({ className = "", style = {}, children, onClick, draggable = false, onDragStart }) => (
  <div 
    className={className} 
    style={style} 
    onClick={onClick}
    draggable={draggable}
    onDragStart={onDragStart}
  >
    {children}
  </div>
);

// Simple local Tag component matching landing page styling exactly
/** Shared with ThreadSummaryModal so the two read identically. */
export const formatTimeAgo = (timestamp: string) => {
  const now = new Date();
  // Ensure timestamp is treated as UTC by appending 'Z' if not present
  const utcTimestamp = timestamp.endsWith('Z') ? timestamp : timestamp + 'Z';
  const past = new Date(utcTimestamp);
  const diffMs = now.getTime() - past.getTime();
  const diffMins = Math.floor(diffMs / (1000 * 60));
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffMins < 1) return 'now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  return `${diffDays}d ago`;
  };

export const Tag: React.FC<{ label: string }> = ({ label }) => (
  <span 
    className="text-[11px] tracking-wide uppercase font-semibold rounded px-2 py-1 text-white transition-all duration-500 bg-[#ff6600]"
    style={{
      lineHeight: '1',
      fontFamily: 'Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    }}
  >
    {label}
  </span>
);

/**
 * The link out to the HN discussion, and the one place that records the visit.
 *
 * It must be shared rather than copied. The global click interceptor in
 * chrome-shim records any HN thread link the user opens, and for a link that
 * does not record itself it has nothing to label the visit with but the link's
 * own text — so Memory fills with rows called "Join the thread - 10 comments".
 * `data-visit-recorded` is what tells the interceptor to stand back and let the
 * theme be stored instead. A second copy of this markup elsewhere silently
 * loses that, which is exactly what happened to the summary modal.
 */
export const JoinThreadLink: React.FC<{
  anchor: string;
  theme: string;
  comment_count: number;
  /** The card flashes this link orange when a thread is dropped onto the chat. */
  highlight?: boolean;
}> = ({ anchor, theme, comment_count, highlight = false }) => (
  <a
    href={anchor}
    target="_blank"
    rel="noopener noreferrer"
    className="flex items-center gap-1 text-[#0066cc] hover:underline"
    data-visit-recorded="true"
    onClick={() => {
      // No preventDefault: the link still opens, this only records the visit.
      const msg: NewTagRequest = {
        type: MessageType.NEW_TAG,
        data: { tag: theme, type: 'visited_thread', anchor },
      };
      chrome.runtime.sendMessage(msg).catch(() => {
        logger.debug('No listeners for NEW_TAG message, this is expected');
      });
    }}
  >
    <MessageCircle
      className={`w-3 h-3 transition-all duration-500 ${highlight ? 'text-[#ff6600] animate-pulse' : ''}`}
    />
    <span
      className={`transition-all duration-500 ${
        highlight ? 'text-[#ff6600] font-medium bg-orange-50/50 px-1 rounded-sm' : ''
      }`}
    >
      Join the thread - {comment_count} comments
    </span>
  </a>
);

export const ThreadCard: React.FC<ThreadCardProps> = ({ 
  comment_count,
  theme,
  category,
  story_title,
  story_url,
  anchor,
  summary,
  updated_at,
  gradient,
  height,
  minHeight,
  maxHeight,
  className = "",
  style = {},
  onClick,
  onSummarize,
  draggable = false,
  onDragStart,
  ...threadData
}) => {
  // Track comment count changes for animation
  const [previousCount, setPreviousCount] = useState(comment_count);
  const [isAnimating, setIsAnimating] = useState(false);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Detect comment count changes and trigger animation
  useEffect(() => {
    if (comment_count !== previousCount && previousCount !== 0) {
      const countIncreased = comment_count > previousCount;
      
      if (countIncreased) {
        // Clear any existing timeout
        if (timeoutRef.current) {
          clearTimeout(timeoutRef.current);
        }
        
        // Trigger animations
        setIsAnimating(true);
        
        // Reset animations after duration
        timeoutRef.current = setTimeout(() => {
          setIsAnimating(false);
        }, 2000);
      }
    }
    setPreviousCount(comment_count);
  }, [comment_count, previousCount]);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);
  
  // Handle drag start
  const handleDragStart = (e: React.DragEvent<HTMLDivElement>) => {
    if (onDragStart) {
      const fullThreadData = {
        ...threadData,
        comment_count,
        theme,
        category,
        story_title,
        story_url,
        anchor,
        summary,
        updated_at
      };
      onDragStart(e, fullThreadData);
    }
  };

  // Combine height-related styles - Match landing page exactly  
  const cardStyle: React.CSSProperties = {
    backgroundColor: '#f6f6ef', // HN beige background like landing page
    height,
    minHeight,
    maxHeight,
    cursor: draggable ? 'grab' : (onClick ? 'pointer' : 'default'),
    ...style
  };

  return (
    <Card 
      className={`w-full h-full border border-[#e0e0e0] rounded-lg shadow-sm hover:shadow-md transition-shadow duration-200 overflow-hidden ${
        isAnimating ? 'animate-subtle-glow' : ''
      } ${onClick ? 'cursor-pointer' : ''} ${draggable ? 'draggable-card' : ''} ${className}`}
      style={cardStyle}
      onClick={onClick}
      draggable={draggable}
      onDragStart={handleDragStart}
    >
      <div className={`p-4 ${height ? 'h-full' : ''} flex flex-col justify-between`}>
        <div className="flex-1 flex flex-col">
          {/* Header: Category and Time */}
          <div className="flex items-center justify-between mb-3 text-sm">
            <Tag label={category} />
            <span className="text-[#999] text-xs">{formatTimeAgo(updated_at)}</span>
          </div>
          
          {/* Divider */}
          <div className="border-b border-[#e0e0e0] mb-3"></div>
          
          {/* Theme Title */}
          <div className="mb-3 text-center">
            <h2 className="text-base font-semibold leading-tight" style={{ color: 'var(--primary)' }}>
              {theme}
            </h2>
          </div>

          {/* Offered where the summary would be, so it reads as the thing that
              fills this space. Stops propagation because the card itself may be
              draggable or clickable. */}
          {!summary && onSummarize && (
            <div className="flex-1 mb-4 text-center">
              <button
                // The card is draggable, and a mousedown anywhere inside it starts a
                // drag, which swallows the click. Both lines are needed: the attribute
                // opts this element out of dragging, and stopping mousedown keeps the
                // parent from claiming the gesture first.
                draggable={false}
                onMouseDown={(e) => e.stopPropagation()}
                onClick={(e) => {
                  e.stopPropagation();
                  onSummarize();
                }}
                className="text-sm underline hover:no-underline"
                style={{ color: 'var(--hn-link, #0066cc)' }}
              >
                What are people saying?
              </button>
            </div>
          )}
          {/* Summary - only show if provided */}
          {summary && (
            <div className="flex-1 mb-4">
              <p className="text-[#333] text-sm leading-6 line-clamp-7 text-center font-['Inter','-apple-system','BlinkMacSystemFont','Segoe_UI','Roboto','sans-serif']">
                {summary}
              </p>
            </div>
          )}

          {/* Metrics Row */}
          <div className="mb-3 flex justify-center text-xs font-mono">
            <JoinThreadLink
              anchor={anchor}
              theme={theme}
              comment_count={comment_count}
              highlight={isAnimating}
            />
          </div>

          {/* Divider */}
          <div className="border-b border-[#e0e0e0] mb-3"></div>
        </div>

        {/* Source Story - Fixed to bottom with consistent spacing */}
        <div className="flex-none text-center">
          <a
            href={story_url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-[#0066cc] text-xs hover:underline inline-flex items-start gap-1 leading-tight line-clamp-2"
          >
            🔗 <span className="line-clamp-2">{story_title}</span>
            <ExternalLink className="w-3 h-3 text-[#999] flex-shrink-0 mt-0.5" />
          </a>
        </div>
      </div>
    </Card>
  );
};