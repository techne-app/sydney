import { ThreadCardData } from '../types/chat';
import { logger } from './logger';

export class ThreadContextService {
  private static readonly API_BASE_URL = 'https://techne-pipeline-func-prod.azurewebsites.net/api';

  /**
   * Fetch thread cards for sidebar context
   * @param numCards - Number of cards to fetch (default: 3)
   * @returns Promise<ThreadCardData[]>
   */
  static async fetchThreadCards(numCards: number = 3): Promise<ThreadCardData[]> {
    try {
      const response = await fetch(`${this.API_BASE_URL}/thread-cards`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          num_cards: numCards,
          hours_back: 24,
          sort_by: 'karma_density',
          density_min_comment_constant: 100,
          exclude_categories: ['General Discussion', 'Discussion']
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      logger.debug(`ThreadContextService: Fetched ${data.length} thread cards`);
      return data;
    } catch (error) {
      logger.error('ThreadContextService: Error fetching thread cards:', error);
      return [];
    }
  }

  /**
   * Load context threads with loading state management
   * @param numCards - Number of cards to fetch
   * @param onLoadingChange - Callback for loading state changes
   * @param onThreadsUpdate - Callback for threads update
   */
  static async loadContextThreads(
    numCards: number = 3,
    onLoadingChange?: (loading: boolean) => void,
    onThreadsUpdate?: (threads: ThreadCardData[]) => void
  ): Promise<ThreadCardData[]> {
    if (onLoadingChange) {
      onLoadingChange(true);
    }

    try {
      const threads = await this.fetchThreadCards(numCards);
      
      if (onThreadsUpdate) {
        onThreadsUpdate(threads);
      }
      
      return threads;
    } catch (error) {
      logger.error('ThreadContextService: Error loading context threads:', error);
      
      if (onThreadsUpdate) {
        onThreadsUpdate([]);
      }
      
      return [];
    } finally {
      if (onLoadingChange) {
        onLoadingChange(false);
      }
    }
  }

  /**
   * Get API base URL (useful for testing or configuration)
   */
  static getApiBaseUrl(): string {
    return this.API_BASE_URL;
  }
}