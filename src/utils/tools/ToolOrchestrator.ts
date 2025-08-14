import { Tool, ToolContext } from './types';
import { 
  ToolExecutionContext, 
  ToolProgressEvent
} from './toolExecution';
import { SearchTool } from './SearchTool';
import { ThreadSummaryTool } from './ThreadSummaryTool';
import { IntentDetector } from '../intentDetector';
import { logger } from '../logger';


export class ToolOrchestrator {
  private tools: Map<string, Tool> = new Map();

  constructor() {
    // Register available tools
    this.registerTool('search_threads', new SearchTool());
    this.registerTool('summarize_pinned_thread', new ThreadSummaryTool());
  }

  private registerTool(functionName: string, tool: Tool): void {
    this.tools.set(functionName, tool);
    logger.debug(`Registered tool: ${functionName} -> ${tool.name}`);
  }

  /**
   * Pure tool execution using AsyncGenerator pattern (new approach)
   * @param userMessage - The user's message
   * @param context - Pure tool execution context
   * @returns AsyncGenerator<ToolProgressEvent, void, unknown>
   */
  async* executeTools(
    userMessage: string, 
    context: ToolExecutionContext
  ): AsyncGenerator<ToolProgressEvent, void, unknown> {
    try {
      // Yield status while detecting intent
      yield { type: 'status', message: '💭 Understanding your request...' };
      
      // Use IntentDetector to determine if this is an action
      const functionResult = await IntentDetector.detectIntent(
        userMessage, 
        context.pinnedThread
      );
      
      logger.intent('ToolOrchestrator function call result:', functionResult);
      
      // Check if we detected an action intent with sufficient confidence
      if (functionResult.intent === 'action' && 
          functionResult.functionCall && 
          functionResult.confidence > 0.5) {
        
        const { name: functionName, parameters } = functionResult.functionCall;
        logger.debug(`ToolOrchestrator routing to function: ${functionName}`);
        
        // Find the appropriate tool for this function
        const tool = this.tools.get(functionName);
        if (!tool) {
          logger.error(`No tool registered for function: ${functionName}`);
          yield { 
            type: 'error', 
            error: `No tool registered for function: ${functionName}` 
          };
          return;
        }
        
        // Execute the tool (legacy approach, we'll update tools later)
        const legacyContext: ToolContext = {
          conversationId: context.conversationId,
          messageId: context.messageId,
          pinnedThread: context.pinnedThread
        };
        
        const result = await tool.execute(parameters, legacyContext);
        
        logger.debug(`ToolOrchestrator tool execution result:`, result);
        
        yield { type: 'complete', data: result };
      } else {
        logger.chat('ToolOrchestrator: No action function detected or low confidence, continuing with chat. Confidence:', functionResult.confidence);
        yield { 
          type: 'complete', 
          data: { 
            wasToolCalled: false, 
            functionResult 
          } 
        };
      }
      
    } catch (error) {
      logger.error('ToolOrchestrator intent detection or tool execution failed:', error);
      yield { 
        type: 'error', 
        error: error instanceof Error ? error.message : 'Unknown error' 
      };
    }
  }



  /**
   * Get list of available tools (useful for MCP introspection)
   */
  getAvailableTools(): string[] {
    return Array.from(this.tools.keys());
  }
}