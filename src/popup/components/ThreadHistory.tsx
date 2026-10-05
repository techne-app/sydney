import React, { useState, useEffect } from 'react';
import { contextDb, type Tag } from '../../utils/contextDb';
import { onActivity } from '../../utils/activity';
import { MemoryCard } from './MemoryCard';
import { logger } from '../../utils/logger';

export const ThreadHistory: React.FC = () => {
  const [tags, setTags] = useState<Tag[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Load tags function
  const loadTags = async () => {
    try {
      const records = await contextDb.getAllTags();
      // Filter out any records with undefined required fields
      const validRecords = records.filter(
        record => record.tag && record.type && record.anchor
      );
      setTags(validRecords);
    } catch (err) {
      logger.error('Failed to load tags:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  // Handle clear all tags
  const handleClearAll = async () => {
    try {
      await contextDb.clearTags();
      setTags([]);
    } catch (err) {
      logger.error('Failed to clear tags:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  };

  // Handle delete individual tag
  const handleDeleteTag = async (tagId: number) => {
    try {
      await contextDb.deleteTag(tagId);
      setTags(prevTags => prevTags.filter(tag => tag.id !== tagId));
    } catch (err) {
      logger.error('Failed to delete tag:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  };

  // Load tags on component mount and set up message listener
  useEffect(() => {
    loadTags();

    // onActivity returns its own unsubscribe, which is exactly what an effect
    // wants back — no paired add/remove to keep in step.
    return onActivity(kind => {
      if (kind === 'threads') loadTags();
    });
  }, []);

  return (
    <div className="p-4" style={{ fontFamily: 'var(--font-sans)' }}>
      <div className="flex justify-between items-center mb-4">
        <h2 
          className="text-xl font-bold"
          style={{ 
            color: 'var(--text-primary)',
            fontWeight: 'var(--font-weight-bold)',
            letterSpacing: 'var(--letter-spacing-tight)'
          }}
        >
          Visited Threads
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
          disabled={loading || tags.length === 0}
        >
          Clear
        </button>
      </div>
      
      {/* Content area */}
      <div className="space-y-2">
        {loading ? (
          <div style={{ color: 'var(--text-secondary)' }}>Loading tags...</div>
        ) : error ? (
          <div style={{ color: '#ef4444' }}>Error: {error}</div>
        ) : !tags.length ? (
          <div style={{ color: 'var(--text-secondary)' }}>No threads visited yet.</div>
        ) : (
          tags.map((tag) => (
            <MemoryCard
              key={tag.id}
              title={tag.tag || ''}
              timestamp={tag.timestamp ? new Date(tag.timestamp).toLocaleString() : ''}
              href={tag.anchor}
              primaryActionLabel="Go to thread"
              onDelete={() => handleDeleteTag(tag.id!)}
            />
          ))
        )}
      </div>
    </div>
  );
};