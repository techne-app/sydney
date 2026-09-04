import Dexie from 'dexie';
import { Tag } from '../types/tag';
import { Search } from '../types/search';
import { Conversation, ChatMessage } from '../types/chat';

// Define the database schema
class ContextDB extends Dexie {
  tags: Dexie.Table<Tag, number>;
  searches: Dexie.Table<Search, number>;
  settings: Dexie.Table<UserSetting, string>;
  conversations: Dexie.Table<Conversation, string>;

  constructor() {
    super('ContextDB');
    
    // Define tables and their primary keys and indexes
    this.version(1).stores({
      tags: '++id, tag, timestamp, type',
      searches: '++id, query, timestamp',
      settings: 'key, timestamp'
    });
    
    this.version(2).stores({
      tags: '++id, tag, timestamp, type',
      searches: '++id, query, timestamp',
      settings: 'key, timestamp',
      conversations: 'id, title, model, createdAt, updatedAt'
    });
    
    // TypeScript type binding
    this.tags = this.table('tags');
    this.searches = this.table('searches');
    this.settings = this.table('settings');
    this.conversations = this.table('conversations');
  }

  // Tag-related methods
  async getAllTags(): Promise<Tag[]> {
    return await this.tags.toArray();
  }

  async storeTag(tag: string, type: string, anchor: string): Promise<number> {
    return await this.tags.add({
      tag,
      type,
      timestamp: Date.now(),
      anchor
    });
  }

  async clearTags(): Promise<void> {
    await this.tags.clear();
  }

  async deleteTag(id: number): Promise<void> {
    await this.tags.delete(id);
  }

  async getRecentTags(k: number): Promise<Tag[]> {
    return await this.tags
      .orderBy('timestamp')
      .reverse()
      .limit(k)
      .toArray();
  }

  // Search-related methods
  async getAllSearches(): Promise<Search[]> {
    return await this.searches.toArray();
  }

  async storeSearch(query: string): Promise<number> {
    return await this.searches.add({
      query,
      timestamp: Date.now()
    });
  }

  async clearSearches(): Promise<void> {
    await this.searches.clear();
  }

  async deleteSearch(id: number): Promise<void> {
    await this.searches.delete(id);
  }

  async getRecentSearches(k: number): Promise<Search[]> {
    return await this.searches
      .orderBy('timestamp')
      .reverse()
      .limit(k)
      .toArray();
  }

  // Settings-related methods
  async getSetting(key: string): Promise<UserSetting | undefined> {
    return await this.settings.get(key);
  }

  async getSettingValue<T>(key: string, defaultValue: T): Promise<T> {
    const setting = await this.getSetting(key);
    return setting ? setting.value : defaultValue;
  }

  async saveSetting(key: string, value: any): Promise<void> {
    await this.settings.put({
      key,
      value,
      timestamp: Date.now()
    });
  }

  async getAllSettings(): Promise<UserSetting[]> {
    return await this.settings.toArray();
  }

  async getRecentSettings(k: number): Promise<UserSetting[]> {
    return await this.settings
      .orderBy('timestamp')
      .reverse()
      .limit(k)
      .toArray();
  }

  // Conversation-related methods
  async getAllConversations(): Promise<Conversation[]> {
    return await this.conversations
      .orderBy('updatedAt')
      .reverse()
      .toArray();
  }

  async getConversation(id: string): Promise<Conversation | undefined> {
    return await this.conversations.get(id);
  }

  async saveConversation(conversation: Conversation): Promise<void> {
    await this.conversations.put(conversation);
  }

  async createConversation(title: string, model: string, modelDisplayName: string): Promise<Conversation> {
    const conversation: Conversation = {
      id: `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      title,
      messages: [],
      model,
      modelDisplayName,
      createdAt: new Date(),
      updatedAt: new Date()
    };
    
    await this.conversations.put(conversation);
    return conversation;
  }

  async updateConversation(id: string, updates: Partial<Conversation>): Promise<void> {
    const conversation = await this.conversations.get(id);
    if (conversation) {
      const updatedConversation = {
        ...conversation,
        ...updates,
        updatedAt: new Date()
      };
      await this.conversations.put(updatedConversation);
    }
  }

  async deleteConversation(id: string): Promise<void> {
    await this.conversations.delete(id);
  }

  async addMessageToConversation(conversationId: string, message: ChatMessage): Promise<void> {
    const conversation = await this.conversations.get(conversationId);
    if (conversation) {
      conversation.messages.push(message);
      conversation.updatedAt = new Date();
      await this.conversations.put(conversation);
    }
  }

  async updateMessageInConversation(
    conversationId: string,
    messageId: string,
    content: string,
    extra?: Partial<ChatMessage>
  ): Promise<void> {
    const conversation = await this.conversations.get(conversationId);
    if (conversation) {
      const messageIndex = conversation.messages.findIndex(m => m.id === messageId);
      if (messageIndex !== -1) {
        conversation.messages[messageIndex].content = content;
        conversation.messages[messageIndex].isStreaming = false;
        // Tools attach their call + raw result here so /route can replay them
        // with the proper roles. Messages are stored as JSON inside the
        // conversation record, so new optional fields need no schema change.
        if (extra) {
          Object.assign(conversation.messages[messageIndex], extra);
        }
        conversation.updatedAt = new Date();
        await this.conversations.put(conversation);
      }
    }
  }

  async clearConversations(): Promise<void> {
    await this.conversations.clear();
  }
}

// Define the UserSetting interface
export interface UserSetting {
  key: string;
  value: any;
  timestamp: number;
}

// Define setting keys as constants
export const SettingKeys = {
  PERSONALIZATION_ENABLED: 'personalization_enabled',
  CHAT_MODEL: 'chat_model',
  CHAT_TEMPERATURE: 'chat_temperature',
  CHAT_TOP_P: 'chat_top_p',
  CHAT_MAX_TOKENS: 'chat_max_tokens',
  LOG_LEVEL: 'log_level'
};

export const contextDb = new ContextDB();
export type { Tag, Search }; 