/**
 * Tauri search client — talks to the Python sidecar's /search endpoint.
 *
 * Replaces the in-webview search pipeline (HN Firebase top-30 → /story-tags/ →
 * MiniLM embedding → cosine). The sidecar holds 30 days of pre-embedded threads
 * in memory, so it embeds only the query, cosines over ~30k vectors, and has
 * Gemma rerank the shortlist. See docs on issue #11 and sidecar/corpus.py.
 */
import { logger } from '../utils/logger';

const SIDECAR_URL = 'http://localhost:8000';

export interface SearchHit {
  thread_id: number;
  story_id: number;
  story_title: string;
  theme: string;
  category: string;
  anchor: string;
  score: number;
}

export interface SearchResponse {
  results: SearchHit[];
  error?: string;
}

class SearchClient {
  /**
   * Run a search in the sidecar. Throws on transport error so the caller can
   * surface a real message rather than an empty result set.
   */
  async search(query: string, limit = 3): Promise<SearchResponse> {
    const response = await fetch(`${SIDECAR_URL}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, limit }),
    });

    if (!response.ok) {
      throw new Error(`Sidecar /search returned ${response.status}`);
    }

    const data = (await response.json()) as SearchResponse;
    logger.search('[searchClient] hits:', data.results?.length ?? 0, data.error ?? '');
    return data;
  }
}

export const searchClient = new SearchClient();
