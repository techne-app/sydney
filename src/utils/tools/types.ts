import { ThreadCardData, ChatMessage } from '../../types/chat';

export interface ElicitationRequest {
  schema: object;
  prompt: string;
  continuation?: any;
}

export interface ToolResult {
  success: boolean;
  data?: any;
  error?: string;
  elicitationRequest?: ElicitationRequest;
}

export interface ToolContext {
  conversationId: string;
  messageId: string;
  pinnedThread?: ThreadCardData | null;
  onProgress?: (content: string) => Promise<void>;
  onStatusUpdate?: (status: string, duration?: number) => void;
}

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