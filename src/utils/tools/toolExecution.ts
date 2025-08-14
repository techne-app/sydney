import { ThreadCardData } from '../../types/chat';

// Pure tool execution context without UI callbacks
export interface ToolExecutionContext {
  conversationId: string;
  messageId: string;
  pinnedThread?: ThreadCardData | null;
}

// Progress events that tools can yield
export interface ToolProgressEvent {
  type: 'status' | 'content' | 'complete' | 'error';
  data?: any;
  message?: string;
  error?: string;
}

// Tool execution result
export interface ToolExecutionResult {
  success: boolean;
  data?: any;
  error?: string;
}

// New pure tool interface using AsyncGenerator
export interface AsyncTool {
  name: string;
  execute(
    input: any, 
    context: ToolExecutionContext
  ): AsyncGenerator<ToolProgressEvent, ToolExecutionResult, unknown>;
}

