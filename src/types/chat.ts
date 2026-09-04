export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
  /**
   * Set when this message is the rendered output of a tool rather than
   * something the model wrote. Replayed to /route as a proper tool_calls +
   * tool-role pair, so the model can tell the difference — without it, the
   * model reads its own past tool output as prose it authored and writes an
   * imitation (inventing threads and links) instead of calling the tool again.
   */
  toolCall?: { name: string; arguments: Record<string, any> };
  /** Raw tool output as JSON, replayed in the tool-role message. */
  toolResult?: string;
}

// ThreadCard data interface for pinned threads
export interface ThreadCardData {
  id: number;
  cumulative_karma: number;
  comment_count: number;
  theme: string;
  category: string;
  story_id: number;
  story_title: string;
  story_url: string;
  anchor: string;
  summary: string;
  updated_at: string;
}

export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  model: string;
  modelDisplayName: string;
  createdAt: Date;
  updatedAt: Date;
  isActive?: boolean;
  pinnedThread?: ThreadCardData | null;
}

export interface ChatState {
  conversations: Conversation[];
  activeConversationId: string | null;
  isLoading: boolean;
  error: string | null;
}

export interface ChatConfig {
  model: string;
  temperature: number;
  topP: number;
  maxTokens: number;
}

export interface ModelOption {
  id: string;
  name: string;
  value: string;
}

export const MODEL_OPTIONS: ModelOption[] = [
  {
    id: "llama",
    name: "Llama-3.2-3B (Powerful)",
    value: "Llama-3.2-3B-Instruct-q4f16_1-MLC"
  },
  {
    id: "gemma",
    name:"gemma-2-2B", 
    value: "gemma-2-2b-it-q4f16_1-MLC"  
  },
  {
    id: "phi",
    name: "Phi-3.5-mini",
    value: "Phi-3.5-mini-instruct-q4f16_1-MLC"
  },
  {
    id: "r1-qwen",
    name: "DeepSeek-R1-Distill-Qwen", 
    value: "DeepSeek-R1-Distill-Qwen-7B-q4f16_1-MLC"
  }
];