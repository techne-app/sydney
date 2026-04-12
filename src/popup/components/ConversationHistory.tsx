import React, { useState, useEffect } from 'react';
import { Conversation } from '../../types/chat';
import { ConversationManager } from '../../utils/conversationUtils';
import { MemoryCard } from './MemoryCard';
import { logger } from '../../utils/logger';

interface ConversationHistoryProps {
  activeConversationId?: string | null;
  onSelectConversation?: (id: string) => void;
  onDeleteConversation?: (id: string) => void;
}

export const ConversationHistory: React.FC<ConversationHistoryProps> = ({
  activeConversationId,
  onSelectConversation,
  onDeleteConversation
}) => {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Load conversations function
  const loadConversations = async () => {
    try {
      const convs = await ConversationManager.getConversations();
      setConversations(convs);
    } catch (err) {
      logger.error('Failed to load conversations:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  // Handle clear all conversations
  const handleClearAll = async () => {
    try {
      // Delete ALL conversations
      const deletePromises = conversations.map(conv => ConversationManager.deleteConversation(conv.id));
      await Promise.all(deletePromises);
      setConversations([]);
      // Fix: notify parent for each deleted conversation so ChatPage state updates too
      if (onDeleteConversation) {
        conversations.forEach(conv => onDeleteConversation(conv.id));
      }
      // Note: The parent component will handle what happens when active conversation is deleted
    } catch (err) {
      logger.error('Failed to clear conversations:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  };


  // Load conversations on component mount
  useEffect(() => {
    loadConversations();
  }, []);

  return (
    <div className="p-4" style={{ fontFamily: 'var(--font-sans)' }}>
      {/* Always show header and clear button */}
      <div className="flex justify-between items-center mb-4">
        <h2 
          className="text-xl font-bold"
          style={{ 
            color: 'var(--text-primary)',
            fontWeight: 'var(--font-weight-bold)',
            letterSpacing: 'var(--letter-spacing-tight)'
          }}
        >
          Past Conversations
        </h2>
        <button
          onClick={handleClearAll}
          className="px-4 py-2 rounded transition-colors disabled:opacity-50"
          style={{ 
            backgroundColor: '#ef4444',
            color: 'white'
          }}
          onMouseEnter={(e) => !e.currentTarget.disabled && (e.currentTarget.style.backgroundColor = '#dc2626')}
          onMouseLeave={(e) => !e.currentTarget.disabled && (e.currentTarget.style.backgroundColor = '#ef4444')}
          disabled={loading || conversations.length === 0}
        >
          Clear
        </button>
      </div>

      {/* Content based on state */}
      {loading ? (
        <div style={{ color: 'var(--text-secondary)' }}>Loading conversations...</div>
      ) : error ? (
        <div style={{ color: '#ef4444' }}>Error: {error}</div>
      ) : conversations.length === 0 ? (
        <div style={{ color: 'var(--text-secondary)' }}>No conversations yet.</div>
      ) : (
        <div className="space-y-2">
          {conversations.map((conversation) => {
            const preview = ConversationManager.formatConversationPreview(conversation);
            return (
              <MemoryCard
                key={conversation.id}
                title={preview.title}
                timestamp={new Date(conversation.createdAt).toLocaleString()}
                isActive={activeConversationId === conversation.id}
                onPrimaryAction={onSelectConversation ? () => onSelectConversation(conversation.id) : undefined}
                onDelete={onDeleteConversation ? () => {
                  onDeleteConversation(conversation.id);
                  // Fix: update local list immediately so UI reflects deletion without reload
                  setConversations(prev => prev.filter(c => c.id !== conversation.id));
                } : undefined}
                primaryActionLabel="Open"
              />
            );
          })}
        </div>
      )}
    </div>
  );
};