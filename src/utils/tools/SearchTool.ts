import { Tool, ToolResult, ToolContext, SearchToolInput } from './types';
import { SearchService } from '../searchService';
import { ConversationManager } from '../conversationUtils';
import { logger } from '../logger';

export class SearchTool implements Tool {
  name = 'search';

  async execute(input: SearchToolInput, context: any): Promise<ToolResult> {
    try {
      logger.search('SearchTool executing with:', input.keyword_filter);
      
      // Pure tool context (new approach)
      const pureContext: ToolContext = {
        conversationId: context.conversationId,
        messageId: context.messageId,
        pinnedThread: context.pinnedThread
      };
      
      // Show search status if legacy callback available
      if (context.onStatusUpdate) {
        context.onStatusUpdate(`🔍 Searching for "${input.keyword_filter}"...`, 1500);
      }
      
      // Update streaming message and database with search status
      const searchStatusContent = `🔍 Searching for "${input.keyword_filter}"...`;
      
      // Legacy progress callback
      if (context.onProgress) {
        await context.onProgress(searchStatusContent);
      }
      
      // Always update database regardless of UI callbacks
      await ConversationManager.updateMessage(
        pureContext.conversationId,
        pureContext.messageId,
        searchStatusContent
      );
      
      // Execute search with streaming, but without UI coupling
      await SearchService.executeSearchStreaming(input.keyword_filter, async (content) => {
        try {
          logger.debug('SearchTool received content update:', content.substring(0, 50) + '...');
          
          // Legacy progress callback
          if (context.onProgress) {
            await context.onProgress(content);
          }
          
          // Always update database regardless of UI callbacks
          await ConversationManager.updateMessage(
            pureContext.conversationId,
            pureContext.messageId,
            content
          );
          
          logger.database('SearchTool database updated with content');
        } catch (error) {
          logger.error('Error in SearchTool streaming callback:', error);
        }
      });
      
      logger.search('SearchTool execution completed successfully');
      
      return {
        success: true,
        data: { keyword_filter: input.keyword_filter }
      };
      
    } catch (error) {
      logger.error('SearchTool execution failed:', error);
      
      // Update the assistant message with error content
      const errorContent = `I encountered an error while searching for "${input.keyword_filter}": ${error instanceof Error ? error.message : 'Unknown error'}`;
      
      // Legacy progress callback
      if (context.onProgress) {
        await context.onProgress(errorContent);
      }
      
      // Always update database regardless of UI callbacks
      const pureContext: ToolContext = {
        conversationId: context.conversationId,
        messageId: context.messageId,
        pinnedThread: context.pinnedThread
      };
      
      await ConversationManager.updateMessage(
        pureContext.conversationId,
        pureContext.messageId,
        errorContent
      );
      
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Search failed'
      };
    }
  }
}