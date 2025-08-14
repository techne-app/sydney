import { ThreadCardData, ChatMessage } from '../../types/chat';

export interface ToolResult {
  success: boolean;
  data?: any;
  error?: string;
}

// Pure tool context without UI callbacks
export interface ToolContext {
  conversationId: string;
  messageId: string;
  pinnedThread?: ThreadCardData | null;
}

// Legacy tool interface for backwards compatibility
export interface Tool {
  name: string;
  execute(input: any, context: ToolContext): Promise<ToolResult>;
}

export interface SearchToolInput {
  keyword_filter: string;
}

export interface ThreadSummaryToolInput {
  // No additional input needed - uses pinnedThread from context
}