import { Tool, ToolResult, ToolContext } from './types';
import { SearchTool } from './SearchTool';
import { ThreadSummaryTool } from './ThreadSummaryTool';
import { IntentDetector, FunctionCallingResult } from '../intentDetectorSingleStep';
import { logger } from '../logger';

export interface ToolExecutionResult {
  wasToolCalled: boolean;
  result?: ToolResult;
  functionResult?: FunctionCallingResult;
}

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
   * Analyze user message and execute appropriate tool if action intent is detected
   * @param userMessage - The user's message
   * @param context - Tool execution context
   * @returns Promise<ToolExecutionResult>
   */
  async handleMessage(userMessage: string, context: ToolContext): Promise<ToolExecutionResult> {
    try {
      // Show temporary status while detecting intent
      if (context.onStatusUpdate) {
        context.onStatusUpdate('💭 Understanding your request...', 800);
      }

      // Use existing IntentDetector to determine if this is an action
      const functionResult = await IntentDetector.detectIntent(
        userMessage, 
        undefined, 
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
          return {
            wasToolCalled: false,
            functionResult
          };
        }
        
        // Execute the tool
        const result = await tool.execute(parameters, context);
        
        logger.debug(`ToolOrchestrator tool execution result:`, result);
        
        return {
          wasToolCalled: true,
          result,
          functionResult
        };
      } else {
        logger.chat('ToolOrchestrator: No action function detected or low confidence, continuing with chat. Confidence:', functionResult.confidence);
        return {
          wasToolCalled: false,
          functionResult
        };
      }
      
    } catch (error) {
      logger.error('ToolOrchestrator intent detection or tool execution failed:', error);
      return {
        wasToolCalled: false
      };
    }
  }

  /**
   * Handle elicitation responses (future MCP feature)
   * @param elicitationId - ID of the elicitation request
   * @param response - User's response to the elicitation
   * @param context - Tool execution context
   * @returns Promise<ToolResult>
   */
  async handleElicitationResponse(
    elicitationId: string, 
    response: any, 
    context: ToolContext
  ): Promise<ToolResult> {
    // Future implementation for MCP elicitation support
    logger.debug('Elicitation response handling not yet implemented');
    return {
      success: false,
      error: 'Elicitation not yet implemented'
    };
  }

  /**
   * Get list of available tools (useful for MCP introspection)
   */
  getAvailableTools(): string[] {
    return Array.from(this.tools.keys());
  }
}