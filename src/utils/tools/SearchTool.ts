import { Tool, ToolResult, ToolContext, SearchToolInput } from './types';
import { SearchService } from '../searchService';
import { ConversationManager } from '../conversationUtils';
import { logger } from '../logger';

export class SearchTool implements Tool {
  name = 'search';

  async execute(input: SearchToolInput, context: ToolContext): Promise<ToolResult> {
    try {
      logger.search('SearchTool executing with:', input.keyword_filter);
      
      // Show search status if status update handler available
      if (context.onStatusUpdate) {
        context.onStatusUpdate(`🔍 Searching for "${input.keyword_filter}"...`, 1500);
      }
      
      // Update streaming message and database with search status
      const searchStatusContent = `🔍 Searching for "${input.keyword_filter}"...`;
      
      if (context.onProgress) {
        await context.onProgress(searchStatusContent);
      }
      
      // Update the database message with search status
      await ConversationManager.updateMessage(
        context.conversationId,
        context.messageId,
        searchStatusContent
      );
      
      // Execute search with streaming
      await SearchService.executeSearchStreaming(input.keyword_filter, async (content) => {
        try {
          logger.debug('SearchTool received content update:', content.substring(0, 50) + '...');
          
          // Update streaming via progress callback
          if (context.onProgress) {
            await context.onProgress(content);
          }
          
          // Update database with current content
          await ConversationManager.updateMessage(
            context.conversationId,
            context.messageId,
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
        error: error instanceof Error ? error.message : 'Search failed'
      };
    }
  }
}