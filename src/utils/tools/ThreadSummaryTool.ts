import { Tool, ToolResult, ToolContext, ThreadSummaryToolInput } from './types';
import { ConversationManager } from '../conversationUtils';
import { logger } from '../logger';

export class ThreadSummaryTool implements Tool {
  name = 'thread_summary';

  async execute(input: ThreadSummaryToolInput, context: any): Promise<ToolResult> {
    try {
      logger.chat('ThreadSummaryTool executing for pinned thread');
      
      // Pure tool context (new approach)
      const pureContext: ToolContext = {
        conversationId: context.conversationId,
        messageId: context.messageId,
        pinnedThread: context.pinnedThread
      };
      
      const summaryResponse = this.generatePinnedThreadSummary(pureContext.pinnedThread);

      // Always update database regardless of UI callbacks. The toolCall marker
      // lets /route replay this with the proper roles rather than as prose the
      // model appears to have written itself — see ChatMessage.toolCall.
      await ConversationManager.updateMessage(
        pureContext.conversationId,
        pureContext.messageId,
        summaryResponse,
        {
          toolCall: { name: 'summarize_pinned_thread', arguments: {} },
          toolResult: JSON.stringify({ summary: summaryResponse }),
        }
      );
      
      // Legacy progress callback
      if (context.onProgress) {
        await context.onProgress(summaryResponse);
      }
      
      logger.chat('ThreadSummaryTool execution completed successfully');
      
      return {
        success: true,
        data: { summary: summaryResponse }
      };
      
    } catch (error) {
      logger.error('ThreadSummaryTool execution failed:', error);
      
      const errorContent = "I encountered an error while generating the thread summary. Please try again.";
      
      // Pure tool context (new approach)
      const pureContext: ToolContext = {
        conversationId: context.conversationId,
        messageId: context.messageId,
        pinnedThread: context.pinnedThread
      };
      
      // Legacy progress callback
      if (context.onProgress) {
        await context.onProgress(errorContent);
      }
      
      // Always update database regardless of UI callbacks
      await ConversationManager.updateMessage(
        pureContext.conversationId,
        pureContext.messageId,
        errorContent
      );
      
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Thread summary failed'
      };
    }
  }

  private generatePinnedThreadSummary(pinnedCard: any): string {
    if (!pinnedCard) {
      return "I don't see a pinned thread to summarize. Please drag a thread from the sidebar to pin it first, then ask me to summarize it.";
    }
    
    // Debug logging
    logger.debug('Pinned card data:', pinnedCard);
    logger.debug('Summary field:', pinnedCard.summary);
    logger.debug('Summary exists?', !!pinnedCard.summary);
    logger.debug('Summary length:', pinnedCard.summary?.length || 0);
    
    // Return just the summary text if available, otherwise a helpful message
    if (pinnedCard.summary && pinnedCard.summary.trim() !== '') {
      return pinnedCard.summary;
    }
    
    return "I couldn't find a summary for this thread. The discussion might be too recent or the summary hasn't been generated yet. You can view the full discussion using the 'Join the thread' link in the pinned card above.";
  }
}