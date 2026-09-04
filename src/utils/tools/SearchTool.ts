import { Tool, ToolResult, ToolContext, SearchToolInput } from './types';
import { searchClient, SearchHit } from '../../tauri-compat/searchClient';
import { MessageType } from '../../types/messages';
import { ConversationManager } from '../conversationUtils';
import { logger } from '../logger';
// OLD in-webview search pipeline — kept for migration reference (delete later):
// import { SearchService } from '../searchService';

/**
 * Render hits the way formatSearchResultsAsMessage did. The UI contract is just
 * a markdown string in the conversation — MessageBubble renders it and
 * chrome-shim intercepts link clicks into open_external_url — so the shape of
 * this string is the whole interface.
 */
function formatResults(query: string, hits: SearchHit[]): string {
  if (hits.length === 0) {
    return `I couldn't find any discussions about "${query}". Try a different search term.`;
  }

  // The theme is the link text, not a trailing "View Discussion". Reads better,
  // and the global click interceptor in chrome-shim uses the link text as the
  // label when it records the visit — so a generic phrase would make every
  // visited thread show up in history as "View Discussion".
  const lines = hits.map(
    (hit, index) => `${index + 1}. [${hit.theme}](${hit.anchor}) — ${hit.story_title}`
  );

  return (
    `I found ${hits.length} discussion${hits.length === 1 ? '' : 's'} about "${query}":\n\n` +
    lines.join('\n\n') +
    `\n\nClick any discussion link to open it.`
  );
}

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
      
      // Record the search so it still shows up in the Activity/Memory view.
      // This used to be emitted inside SearchService.executeSearchStreaming;
      // since we no longer call that, it has to happen here or search history
      // silently goes empty.
      chrome.runtime.sendMessage({
        type: MessageType.NEW_SEARCH,
        data: { query: input.keyword_filter }
      }).catch(() => {
        logger.debug('No listeners for NEW_SEARCH, this is expected');
      });

      // The sidecar does the whole search: embeds the query with nomic, cosines
      // over ~30k in-memory vectors, and has Gemma rerank the top 100 down to 3.
      const response = await searchClient.search(input.keyword_filter);

      const content = response.error && response.results.length === 0
        ? `I couldn't search just now: ${response.error}`
        : formatResults(input.keyword_filter, response.results);

      if (context.onProgress) {
        await context.onProgress(content);
      }

      await ConversationManager.updateMessage(
        pureContext.conversationId,
        pureContext.messageId,
        content
      );

      logger.search('SearchTool execution completed successfully');

      return {
        success: true,
        data: { keyword_filter: input.keyword_filter, results: response.results }
      };

      /* --- OLD in-webview search path (HN Firebase top-30 -> /story-tags/ ->
       * MiniLM embed -> cosine). Superseded by the sidecar's /search above.
       * Kept for migration reference; delete once this is proven.
       *
       * await SearchService.executeSearchStreaming(input.keyword_filter, async (content) => {
       *   try {
       *     logger.debug('SearchTool received content update:', content.substring(0, 50) + '...');
       *     if (context.onProgress) {
       *       await context.onProgress(content);
       *     }
       *     await ConversationManager.updateMessage(
       *       pureContext.conversationId,
       *       pureContext.messageId,
       *       content
       *     );
       *     logger.database('SearchTool database updated with content');
       *   } catch (error) {
       *     logger.error('Error in SearchTool streaming callback:', error);
       *   }
       * });
       */

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