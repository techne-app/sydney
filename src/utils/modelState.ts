export interface ModelState {
  isLoading: boolean;
  progress: number;
  status: string;
  isLoaded: boolean;
}

export type ModelStateListener = (state: ModelState) => void;

/**
 * Simple observable model state manager for WebLLM
 * Allows UI components to observe model loading progress without coupling business logic to UI callbacks
 */
export class ModelStateManager {
  private static instance: ModelStateManager;
  private listeners: Set<ModelStateListener> = new Set();
  private state: ModelState = {
    isLoading: false,
    progress: 0,
    status: '',
    isLoaded: false
  };

  static getInstance(): ModelStateManager {
    if (!ModelStateManager.instance) {
      ModelStateManager.instance = new ModelStateManager();
    }
    return ModelStateManager.instance;
  }

  /**
   * Subscribe to model state changes
   * @param listener - Function called when state changes
   * @returns Unsubscribe function
   */
  subscribe(listener: ModelStateListener): () => void {
    this.listeners.add(listener);
    // Immediately call with current state
    listener(this.state);
    
    return () => {
      this.listeners.delete(listener);
    };
  }

  /**
   * Get current model state
   */
  getState(): ModelState {
    return { ...this.state };
  }

  /**
   * Update model state and notify listeners
   * @private - Only webLLMClient should call this
   */
  private updateState(updates: Partial<ModelState>): void {
    this.state = { ...this.state, ...updates };
    this.listeners.forEach(listener => listener(this.state));
  }

  /**
   * Signal model loading started
   * @internal - Called by webLLMClient
   */
  setLoading(isLoading: boolean, status?: string): void {
    this.updateState({
      isLoading,
      status: status || this.state.status,
      progress: isLoading ? 0 : this.state.progress
    });
  }

  /**
   * Update model loading progress
   * @internal - Called by webLLMClient
   */
  setProgress(progress: number, status: string): void {
    this.updateState({
      progress,
      status,
      isLoading: progress < 1
    });
  }

  /**
   * Signal model loading completed
   * @internal - Called by webLLMClient
   */
  setLoaded(isLoaded: boolean): void {
    this.updateState({
      isLoaded,
      isLoading: false,
      progress: isLoaded ? 1 : 0,
      status: isLoaded ? 'Model ready' : 'Model not loaded'
    });
  }
}

// Export singleton instance
export const modelState = ModelStateManager.getInstance();