import { Tool, ToolResult, ToolContext, ThreadSummaryToolInput } from './types';
import { ConversationManager } from '../conversationUtils';
import { logger } from '../logger';

export class ThreadSummaryTool implements Tool {
  name = 'thread_summary';

  async execute(input: ThreadSummaryToolInput, context: ToolContext): Promise<ToolResult> {
    try {
      logger.chat('ThreadSummaryTool executing for pinned thread');
      
      const summaryResponse = this.generatePinnedThreadSummary(context.pinnedThread);
      
      // Update the assistant message with the summary
      await ConversationManager.updateMessage(
        context.conversationId,
        context.messageId,
        summaryResponse
      );
      
      // Update streaming via progress callback if available
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
      
      if (context.onProgress) {
        await context.onProgress(errorContent);
      }
      
      await ConversationManager.updateMessage(
        context.conversationId,
        context.messageId,
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