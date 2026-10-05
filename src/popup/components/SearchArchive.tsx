import React, { useState, useEffect } from 'react';
import { contextDb, type Search } from '../../utils/contextDb';
import { onActivity } from '../../utils/activity';
import { MemoryCard } from './MemoryCard';
import { logger } from '../../utils/logger';

export const SearchArchive: React.FC = () => {
  const [searches, setSearches] = useState<Search[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Load searches function
  const loadSearches = async () => {
    try {
      const records = await contextDb.getRecentSearches(10); // Get the 10 most recent searches
      setSearches(records);
    } catch (err) {
      logger.error('Failed to load searches:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  // Handle clear all searches
  const handleClearAll = async () => {
    try {
      await contextDb.clearSearches();
      setSearches([]);
    } catch (err) {
      logger.error('Failed to clear searches:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  };

  // Handle delete individual search
  const handleDeleteSearch = async (searchId: number) => {
    try {
      await contextDb.deleteSearch(searchId);
      setSearches(prevSearches => prevSearches.filter(search => search.id !== searchId));
    } catch (err) {
      logger.error('Failed to delete search:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  };

  // Load searches on component mount and set up message listener
  useEffect(() => {
    loadSearches();

    return onActivity(kind => {
      if (kind === 'searches') loadSearches();
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
          Recent Searches
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
          disabled={loading || searches.length === 0}
        >
          Clear
        </button>
      </div>
      
      {/* Content area */}
      <div className="space-y-2">
        {loading ? (
          <div style={{ color: 'var(--text-secondary)' }}>Loading search history...</div>
        ) : error ? (
          <div style={{ color: '#ef4444' }}>Error: {error}</div>
        ) : !searches.length ? (
          <div style={{ color: 'var(--text-secondary)' }}>No search history yet.</div>
        ) : (
          searches.map((search) => (
            <MemoryCard
              key={search.id}
              title={search.query}
              timestamp={new Date(search.timestamp).toLocaleString()}
              onDelete={() => handleDeleteSearch(search.id!)}
            />
          ))
        )}
      </div>
    </div>
  );
};