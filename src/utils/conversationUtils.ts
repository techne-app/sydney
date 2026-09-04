import { contextDb } from '../background/contextDb';
import { Conversation, ChatMessage, MODEL_OPTIONS, ThreadCardData } from '../types/chat';

export class ConversationManager {
  // Create a draft conversation that exists only in memory until first user message
  static createDraftConversation(
    title: string = 'New Conversation',
    modelValue: string = MODEL_OPTIONS[0].value
  ): Conversation {
    const modelOption = MODEL_OPTIONS.find(m => m.value === modelValue) || MODEL_OPTIONS[0];
    return {
      id: `draft_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      title,
      messages: [],
      model: modelValue,
      modelDisplayName: modelOption.name,
      createdAt: new Date(),
      updatedAt: new Date()
    };
  }

  // Save a draft conversation to the database (called when first user message is sent)
  static async saveDraftConversation(conversation: Conversation): Promise<Conversation> {
    // Give it a proper ID for the database
    const savedConversation = {
      ...conversation,
      id: `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    };
    await contextDb.conversations.put(savedConversation);
    return savedConversation;
  }

  static async createNewConversation(
    title: string = 'New Conversation',
    modelValue: string = MODEL_OPTIONS[0].value
  ): Promise<Conversation> {
    const modelOption = MODEL_OPTIONS.find(m => m.value === modelValue) || MODEL_OPTIONS[0];
    return await contextDb.createConversation(title, modelValue, modelOption.name);
  }

  static async getConversations(): Promise<Conversation[]> {
    return await contextDb.getAllConversations();
  }

  static async getConversation(id: string): Promise<Conversation | undefined> {
    return await contextDb.getConversation(id);
  }

  static async addMessage(
    conversationId: string,
    role: 'user' | 'assistant',
    content: string,
    isStreaming: boolean = false
  ): Promise<ChatMessage> {
    const message: ChatMessage = {
      id: `msg_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`,
      role,
      content,
      timestamp: new Date(),
      isStreaming
    };

    await contextDb.addMessageToConversation(conversationId, message);
    return message;
  }

  static async updateMessage(
    conversationId: string,
    messageId: string,
    content: string,
    extra?: Partial<ChatMessage>
  ): Promise<void> {
    await contextDb.updateMessageInConversation(conversationId, messageId, content, extra);
  }

  static async updateConversationTitle(
    conversationId: string,
    title: string
  ): Promise<void> {
    await contextDb.updateConversation(conversationId, { title });
  }

  static async deleteConversation(id: string): Promise<void> {
    await contextDb.deleteConversation(id);
  }

  static generateConversationTitle(firstMessage: string, maxLength: number = 50): string {
    const cleaned = firstMessage.trim().replace(/\s+/g, ' ');
    if (cleaned.length <= maxLength) {
      return cleaned;
    }
    return cleaned.substring(0, maxLength - 3) + '...';
  }

  static formatMessageTime(timestamp: Date): string {
    return new Intl.DateTimeFormat('en-US', {
      month: 'numeric',
      day: 'numeric',
      year: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    }).format(timestamp);
  }

  static formatConversationPreview(conversation: Conversation): {
    title: string;
    messageCount: number;
    lastUpdated: string;
  } {
    return {
      title: conversation.title,
      messageCount: conversation.messages.length,
      lastUpdated: this.formatMessageTime(conversation.updatedAt)
    };
  }

  static async searchConversations(query: string): Promise<Conversation[]> {
    const conversations = await this.getConversations();
    const lowerQuery = query.toLowerCase();
    
    return conversations.filter(conv => 
      conv.title.toLowerCase().includes(lowerQuery) ||
      conv.messages.some(msg => 
        msg.content.toLowerCase().includes(lowerQuery)
      )
    );
  }

  // Utility function to clean up empty conversations (conversations with no messages)
  static async cleanupEmptyConversations(): Promise<number> {
    const conversations = await this.getConversations();
    const emptyConversations = conversations.filter(conv => 
      conv.messages.length === 0 && conv.title === 'New Conversation'
    );
    
    for (const conv of emptyConversations) {
      await this.deleteConversation(conv.id);
    }
    
    return emptyConversations.length;
  }

  // Pinned thread methods
  static async pinThreadToConversation(
    conversationId: string, 
    threadData: ThreadCardData
  ): Promise<void> {
    const conversation = await contextDb.getConversation(conversationId);
    if (!conversation) {
      throw new Error(`Conversation ${conversationId} not found`);
    }

    const updatedConversation = {
      ...conversation,
      pinnedThread: threadData,
      updatedAt: new Date()
    };

    await contextDb.updateConversation(conversationId, updatedConversation);
  }

  static async unpinThreadFromConversation(conversationId: string): Promise<void> {
    const conversation = await contextDb.getConversation(conversationId);
    if (!conversation) {
      throw new Error(`Conversation ${conversationId} not found`);
    }

    const updatedConversation = {
      ...conversation,
      pinnedThread: null,
      updatedAt: new Date()
    };

    await contextDb.updateConversation(conversationId, updatedConversation);
  }

  static async getPinnedThread(conversationId: string): Promise<ThreadCardData | null> {
    const conversation = await contextDb.getConversation(conversationId);
    return conversation?.pinnedThread || null;
  }
}

export default ConversationManager;