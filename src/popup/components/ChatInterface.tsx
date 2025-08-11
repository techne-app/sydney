import React, { useState, useRef, useEffect } from 'react';
import { type ChatCompletionMessageParam } from "@mlc-ai/web-llm";
import { Conversation, ChatMessage, MODEL_OPTIONS, ThreadCardData } from '../../types/chat';
import { ConversationManager } from '../../utils/conversationUtils';
import { webLLMClient } from '../../utils/webLLMClient';
import { configStore } from '../../utils/configStore';
import MessageBubble from './MessageBubble';
import { IntentDetector } from '../../utils/intentDetector';
import { SearchService } from '../../utils/searchService';
import { logger } from '../../utils/logger';
import { Modal } from './Modal';
import { ActivityPage } from './ActivityPage';
import { SettingsPage } from './SettingsPage';
import { ThreadCard } from './ThreadCard';
import { MessageCircle, ExternalLink } from 'lucide-react';


// API configuration for fetching thread cards
const API_BASE_URL = 'https://techne-pipeline-func-prod.azurewebsites.net/api';

const fetchThreadCards = async (numCards: number = 3): Promise<ThreadCardData[]> => {
  try {
    const response = await fetch(`${API_BASE_URL}/thread-cards`, {
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
    return data;
  } catch (error) {
    logger.error('Error fetching thread cards:', error);
    return [];
  }
};

// Helper function to convert technical errors to user-friendly messages
const getUserFriendlyErrorMessage = (error: string): string => {
  const lowerError = error.toLowerCase();
  if (lowerError.includes('cache') || lowerError.includes('failed to execute') || lowerError.includes('networkerror')) {
    return 'Unable to download AI model - please check your connection and try again';
  }
  // Default case for other errors
  return 'Something went wrong - please try again';
};

interface ChatInterfaceProps {
  activeConversation: Conversation | null;
  onConversationUpdated: (conversation: Conversation) => void;
  conversations?: Conversation[];
  isTemporary?: boolean; // Flag to indicate if this is a temporary/unsaved conversation
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({ 
  activeConversation, 
  onConversationUpdated,
  conversations = [],
  isTemporary = false
}) => {
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [streamingMessage, setStreamingMessage] = useState<ChatMessage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadedModelName, setLoadedModelName] = useState<string>('');
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [modelLoadingProgress, setModelLoadingProgress] = useState(0);
  const [isMemoryModalOpen, setIsMemoryModalOpen] = useState(false);
  const [isSettingsModalOpen, setIsSettingsModalOpen] = useState(false);
  const [isCreatingConversation, setIsCreatingConversation] = useState(false);
  const [contextThreads, setContextThreads] = useState<ThreadCardData[]>([]);
  const [threadsLoading, setThreadsLoading] = useState(false);
  const [tempStatus, setTempStatus] = useState<string | null>(null);
  const [pinnedCard, setPinnedCard] = useState<ThreadCardData | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Load pinned thread when active conversation changes
  useEffect(() => {
    if (activeConversation?.pinnedThread) {
      setPinnedCard(activeConversation.pinnedThread);
    } else {
      setPinnedCard(null);
    }
  }, [activeConversation]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeConversation?.messages, streamingMessage]);

  // Helper function to show temporary status messages
  const showTempStatus = (message: string, duration: number = 2000) => {
    setTempStatus(message);
    setTimeout(() => setTempStatus(null), duration);
  };

  // Reusable function to create new conversation (as draft)
  const createNewConversation = async () => {
    if (isCreatingConversation) return; // Prevent double-clicks
    
    setIsCreatingConversation(true);
    try {
      const config = await configStore.getConfig();
      const draftConv = ConversationManager.createDraftConversation(
        'New Conversation',
        config.model
      );
      onConversationUpdated(draftConv);
    } catch (error) {
      logger.error('Error creating new conversation:', error);
      setError('Failed to create new conversation. Please try again.');
    } finally {
      setIsCreatingConversation(false);
    }
  };

  // Update model display name when conversation changes
  useEffect(() => {
    if (activeConversation) {
      setLoadedModelName(activeConversation.modelDisplayName);
    }
  }, [activeConversation]);

  // Load context threads on component mount
  useEffect(() => {
    const loadContextThreads = async () => {
      setThreadsLoading(true);
      try {
        const threads = await fetchThreadCards(3);
        setContextThreads(threads);
      } catch (error) {
        logger.error('Error loading context threads:', error);
      } finally {
        setThreadsLoading(false);
      }
    };

    loadContextThreads();
  }, []);

  // Handle drag start from sidebar cards
  const handleDragStart = (e: React.DragEvent<HTMLDivElement>, cardData: ThreadCardData) => {
    e.dataTransfer.setData('application/json', JSON.stringify(cardData));
    e.dataTransfer.effectAllowed = 'copy';
  };

  // Handle drop in chat area
  const handleDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    
    try {
      const cardData = JSON.parse(e.dataTransfer.getData('application/json')) as ThreadCardData;
      setPinnedCard(cardData);
      
      // Save to database if we have an active conversation
      if (activeConversation) {
        await ConversationManager.pinThreadToConversation(activeConversation.id, cardData);
        // Update local conversation state
        onConversationUpdated({
          ...activeConversation,
          pinnedThread: cardData,
          updatedAt: new Date()
        });
      }
    } catch (error) {
      logger.error('Error handling drop:', error);
    }
  };

  // Handle drag over
  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  };

  // Handle drag leave
  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    // No visual feedback needed
  };

  // Remove pinned card
  const removePinnedCard = async () => {
    setPinnedCard(null);
    
    // Remove from database if we have an active conversation
    if (activeConversation) {
      try {
        await ConversationManager.unpinThreadFromConversation(activeConversation.id);
        // Update local conversation state
        onConversationUpdated({
          ...activeConversation,
          pinnedThread: null,
          updatedAt: new Date()
        });
      } catch (error) {
        logger.error('Error removing pinned thread:', error);
      }
    }
  };

  // Handle search request with streaming - reuse existing assistant message
  const handleSearchRequest = async (searchQuery: string, assistantMessageId: string) => {
    try {
      logger.search('Starting handleSearchRequest for:', searchQuery);
      
      // Execute search with streaming
      await SearchService.executeSearchStreaming(searchQuery, async (content) => {
        try {
          logger.debug('Received content update:', content.substring(0, 50) + '...');
          
          // Update streaming message state
          setStreamingMessage(prev => prev ? {
            ...prev,
            content,
            isStreaming: true
          } : null);
          
          // Update database with current content (await the async operation)
          await ConversationManager.updateMessage(
            activeConversation!.id, 
            assistantMessageId, 
            content
          );
          
          logger.database('Database updated with content');
        } catch (error) {
          logger.error('Error in search streaming callback:', error);
        }
      });
      
      logger.search('SearchService.executeSearchStreaming completed');
      
      // Finalize the streaming message
      setStreamingMessage(prev => prev ? {
        ...prev,
        isStreaming: false
      } : null);
      
      logger.debug('Getting updated conversation...');
      
      // Get updated conversation
      const updatedConversation = await ConversationManager.getConversation(activeConversation!.id);
      if (updatedConversation) {
        onConversationUpdated(updatedConversation);
        logger.debug('Conversation updated in UI');
      }
    } catch (error) {
      logger.error('Search error:', error);
      // Update the assistant message with error content
      const errorContent = `I encountered an error while searching for "${searchQuery}": ${error instanceof Error ? error.message : 'Unknown error'}`;
      await ConversationManager.updateMessage(
        activeConversation!.id, 
        assistantMessageId, 
        errorContent
      );
      setError('Search failed. Please try again.');
    } finally {
      logger.debug('handleSearchRequest finally block');
      setIsLoading(false);
      setStreamingMessage(null);
    }
  };


  const handleSubmit = async () => {
    if (!message.trim() || !activeConversation || isModelLoading) return;

    const userMessage = message.trim();
    setMessage('');
    setIsLoading(true);
    setError(null);

    try {
      let workingConversation = activeConversation;
      
      // If this is a draft conversation (first user message), save it to the database
      if (activeConversation.id.startsWith('draft_') && activeConversation.messages.length === 0) {
        // Generate title from first message and save the conversation
        const newTitle = ConversationManager.generateConversationTitle(userMessage);
        workingConversation = await ConversationManager.saveDraftConversation({
          ...activeConversation,
          title: newTitle,
          pinnedThread: pinnedCard  // Preserve the pinned thread when saving draft
        });
        
        // Update the conversation in the UI
        onConversationUpdated(workingConversation);
      }
      
      // Add user message to conversation
      await ConversationManager.addMessage(workingConversation.id, 'user', userMessage);
      
      // Update conversation title if this is the first message of a saved conversation
      if (workingConversation.messages.length === 0 && !workingConversation.id.startsWith('draft_')) {
        const newTitle = ConversationManager.generateConversationTitle(userMessage);
        await ConversationManager.updateConversationTitle(workingConversation.id, newTitle);
      }

      // Get updated conversation
      const updatedConversation = await ConversationManager.getConversation(workingConversation.id);
      if (updatedConversation) {
        onConversationUpdated(updatedConversation);
      }

      // Create streaming assistant message for regular chat
      const assistantMessage = await ConversationManager.addMessage(
        workingConversation.id, 
        'assistant', 
        '', 
        true
      );
      setStreamingMessage(assistantMessage);

      // Prepare chat history for the engine
      const chatHistory: ChatCompletionMessageParam[] = updatedConversation?.messages.map(msg => ({
        role: msg.role as 'user' | 'assistant',
        content: msg.content
      })) || [];

      // Get current config
      const config = await configStore.getConfig();
      setLoadedModelName(MODEL_OPTIONS.find(m => m.value === config.model)?.name || config.model);


      // Check if model is already loaded by checking webLLMClient state
      const isModelLoaded = webLLMClient.isModelLoaded();
      logger.model('Model loaded check:', isModelLoaded);
      
      if (isModelLoaded) {
        logger.model('Model already loaded, checking for search intent...');
        try {
          showTempStatus('💭 Understanding your request...', 800);
          const intentResult = await IntentDetector.detectSearchIntent(userMessage);
          logger.intent('Intent detection result:', intentResult);
          
          if (intentResult.isSearch && intentResult.searchQuery && intentResult.confidence > 0.5) {
            logger.search('Search intent detected with high confidence, executing search for:', intentResult.searchQuery);
            // Show temporary status in bottom bar
            showTempStatus(`🔍 Searching for "${intentResult.searchQuery}"...`, 1500);
            
            // Update both streaming message and database with search status
            const searchStatusContent = `🔍 Searching for "${intentResult.searchQuery}"...`;
            setStreamingMessage(prev => prev ? {
              ...prev,
              content: searchStatusContent,
              isStreaming: false
            } : null);
            
            // Update the database message with search status
            await ConversationManager.updateMessage(
              workingConversation.id, 
              assistantMessage.id, 
              searchStatusContent
            );
            
            await handleSearchRequest(intentResult.searchQuery, assistantMessage.id);
            
            // Get final updated conversation after search completes
            const finalConversation = await ConversationManager.getConversation(workingConversation.id);
            if (finalConversation) {
              onConversationUpdated(finalConversation);
            }
            return;
          } else {
            logger.chat('No search intent detected or low confidence, continuing with chat. Confidence:', intentResult.confidence);
          }
        } catch (error) {
          logger.error('Intent detection failed:', error);
        }
      } else {
        logger.model('Model not loaded, will perform intent detection after loading');
      }

      // Only proceed with chat if we didn't intercept for search
      logger.chat('Starting chat conversation...');

      // Use WebLLM client with web-llm-chat patterns
      try {
        await webLLMClient.chat({
        messages: chatHistory,
        config: {
          model: config.model,
          temperature: config.temperature,
          topP: config.topP,
          maxTokens: config.maxTokens,
          stream: true,
        },
        // Only show loading UI if model actually needs to load
        onModelLoadingStart: !isModelLoaded ? () => {
          setIsModelLoading(true);
          setModelLoadingProgress(0);
        } : undefined,
        onModelLoadingProgress: !isModelLoaded ? (progress) => {
          setModelLoadingProgress(progress);
        } : undefined,
        onModelLoadingComplete: !isModelLoaded ? async () => {
          setIsModelLoading(false);
          setModelLoadingProgress(1);
          
          // Perform intent detection after model loads
          logger.model('Model loading complete, now checking for search intent...');
          try {
            showTempStatus('💭 Understanding your request...', 800);
            const intentResult = await IntentDetector.detectSearchIntent(userMessage);
            logger.intent('Intent detection result:', intentResult);
            
            if (intentResult.isSearch && intentResult.searchQuery && intentResult.confidence > 0.5) {
              logger.search('Search intent detected with high confidence, executing search for:', intentResult.searchQuery);
              // Show temporary status in bottom bar
              showTempStatus(`🔍 Searching for "${intentResult.searchQuery}"...`, 1500);
              
              // Update both streaming message and database with search status
              const searchStatusContent = `🔍 Searching for "${intentResult.searchQuery}"...`;
              setStreamingMessage(prev => prev ? {
                ...prev,
                content: searchStatusContent,
                isStreaming: false
              } : null);
              
              // Update the database message with search status
              await ConversationManager.updateMessage(
                workingConversation.id, 
                assistantMessage.id, 
                searchStatusContent
              );
              
              await handleSearchRequest(intentResult.searchQuery, assistantMessage.id);
              
              // Get final updated conversation after search completes
              const finalConversation = await ConversationManager.getConversation(workingConversation.id);
              if (finalConversation) {
                onConversationUpdated(finalConversation);
              }
              return false; // Abort chat
            } else {
              logger.chat('No search intent detected or low confidence, continuing with chat. Confidence:', intentResult.confidence);
            }
          } catch (error) {
            logger.error('Intent detection failed:', error);
          }
          
          return true; // Continue with chat
        } : undefined,
        onUpdate: (message) => {
          // Update streaming message
          setStreamingMessage(prev => prev ? {
            ...prev,
            content: message
          } : null);
        },
        onFinish: async (message) => {
          // Update the message in the database
          await ConversationManager.updateMessage(workingConversation.id, assistantMessage.id, message);
          
          // Get final updated conversation
          const finalConversation = await ConversationManager.getConversation(workingConversation.id);
          if (finalConversation) {
            onConversationUpdated(finalConversation);
          }

          setStreamingMessage(null);
        },
        onError: (errorMessage) => {
          logger.error('WebLLM chat error:', errorMessage);
          setError(errorMessage);
          setStreamingMessage(null);
          setIsModelLoading(false);
        }
        });
      } catch (error) {
        logger.error('WebLLM chat failed:', error);
        setIsModelLoading(false);
        setError('Chat failed to start');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get response');
      logger.error('Error during chat:', err);
      setStreamingMessage(null);
      setIsModelLoading(false);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  if (!activeConversation) {
    return (
      <div className="flex-1 flex items-center justify-center chat-container" style={{ color: 'var(--text-secondary)' }}>
        <div className="text-center">
          <svg className="w-16 h-16 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" style={{ color: 'var(--hn-border)' }}>
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
          <p className="text-lg font-medium" style={{ color: 'var(--text-primary)' }}>Ready to start chatting</p>
          <p className="text-sm mt-2 mb-4">Create a new conversation or browse your conversation history</p>
          <div className="flex flex-col items-center space-y-3">
            <button
              onClick={createNewConversation}
              disabled={isCreatingConversation}
              className="px-6 py-3 rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              style={{ 
                backgroundColor: 'var(--hn-blue)', 
                color: 'white'
              }}
              onMouseEnter={(e) => {
                if (!isCreatingConversation) {
                  e.currentTarget.style.backgroundColor = '#0052a3';
                }
              }}
              onMouseLeave={(e) => {
                if (!isCreatingConversation) {
                  e.currentTarget.style.backgroundColor = 'var(--hn-blue)';
                }
              }}
            >
              {isCreatingConversation ? 'Creating...' : 'Start New Conversation'}
            </button>
            <button
              onClick={() => setIsMemoryModalOpen(true)}
              className="px-4 py-2 border rounded-lg transition-colors"
              style={{ 
                color: 'var(--text-secondary)',
                borderColor: 'var(--hn-border)'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.color = 'var(--text-primary)';
                e.currentTarget.style.borderColor = 'var(--text-secondary)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.color = 'var(--text-secondary)';
                e.currentTarget.style.borderColor = 'var(--hn-border)';
              }}
            >
              Browse Conversation History
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <>
    <div className="flex-1 flex h-full chat-container min-h-0">
      {/* Left Sidebar - Context Thread Cards */}
      <div className="w-80 flex-shrink-0 overflow-y-auto" style={{ minWidth: '240px' }}>
        <div className="p-4">
          {/* Logo Section - Compact */}
          <div className="flex justify-center mb-4">
            <div className="relative w-12 h-12">
              {/* Background glow layer */}
              <div
                className="absolute inset-0 rounded-full opacity-50"
                style={{
                  background: "radial-gradient(circle, rgba(59, 130, 246, 0.2) 0%, rgba(59, 130, 246, 0.05) 40%, transparent 70%)",
                  mixBlendMode: "screen",
                  animation: "logoGlow 6s ease-in-out infinite"
                }}
              />
              
              {/* Logo */}
              <div
                className="relative z-10"
                style={{
                  filter: "blur(0px) contrast(1.0) saturate(1.0) drop-shadow(0 0 8px rgba(59, 130, 246, 0.3))",
                  mixBlendMode: "normal"
                }}
              >
                <img 
                  src="/logo.png" 
                  alt="Techne Logo" 
                  width={48}
                  height={48}
                  className="w-full h-full rounded-full border border-white/20"
                  style={{
                    clipPath: "circle(44% at 50% 50%)"
                  }}
                />
              </div>
            </div>
          </div>
          {threadsLoading ? (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-6 w-6 border mx-auto mb-2" style={{ borderColor: 'var(--text-secondary)', borderTopColor: 'var(--hn-blue)' }}></div>
              <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>Loading context...</p>
            </div>
          ) : contextThreads.length > 0 ? (
            <div className="space-y-4">
              {contextThreads.map((thread) => (
                <div key={thread.id} className="w-full">
                  <div
                    className="text-xs cursor-pointer transition-colors"
                    style={{
                      backgroundColor: 'transparent', // Let ThreadCard control its own background
                      borderColor: 'transparent'
                    }}
                  >
                    <ThreadCard 
                      {...thread}
                      className="text-xs sidebar-thread-card"
                      style={{
                        fontSize: '11px'
                      }}
                      summary=""
                      draggable={true}
                      onDragStart={handleDragStart}
                    />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>No context available</p>
            </div>
          )}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-h-0">
        {/* Header */}
        <div className="flex-shrink-0 p-4 border-b" style={{ borderColor: 'var(--hn-border)' }}>
          <div className="flex items-center justify-between min-w-0">
            <div className="flex-1 min-w-0 pr-4">
              <h2 className="text-lg font-semibold truncate" style={{ color: 'var(--text-primary)' }}>
                {activeConversation.title}
              </h2>
            </div>
            <div className="flex items-center space-x-2 flex-shrink-0">
              {/* New Conversation Button */}
              <button
                onClick={createNewConversation}
                disabled={isCreatingConversation}
                className="p-2 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                style={{ 
                  color: 'var(--text-secondary)',
                  backgroundColor: 'transparent'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.color = 'var(--text-primary)';
                  e.currentTarget.style.backgroundColor = 'var(--dark-card)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.color = 'var(--text-secondary)';
                  e.currentTarget.style.backgroundColor = 'transparent';
                }}
                title={isCreatingConversation ? "Creating..." : "New Conversation"}
              >
                {isCreatingConversation ? (
                  <div className="animate-spin rounded-full h-5 w-5 border" style={{ borderColor: 'var(--text-secondary)', borderTopColor: 'var(--hn-blue)' }}></div>
                ) : (
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                  </svg>
                )}
              </button>
              
              {/* Memory Icon */}
              <button
                onClick={() => setIsMemoryModalOpen(true)}
                className="p-2 rounded-lg transition-colors"
                style={{ 
                  color: 'var(--text-secondary)',
                  backgroundColor: 'transparent'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.color = 'var(--text-primary)';
                  e.currentTarget.style.backgroundColor = 'var(--dark-card)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.color = 'var(--text-secondary)';
                  e.currentTarget.style.backgroundColor = 'transparent';
                }}
                title="Memory"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
                </svg>
              </button>
              
              {/* Settings Icon */}
              <button
                onClick={() => setIsSettingsModalOpen(true)}
                className="p-2 rounded-lg transition-colors"
                style={{ 
                  color: 'var(--text-secondary)',
                  backgroundColor: 'transparent'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.color = 'var(--text-primary)';
                  e.currentTarget.style.backgroundColor = 'var(--dark-card)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.color = 'var(--text-secondary)';
                  e.currentTarget.style.backgroundColor = 'transparent';
                }}
                title="Settings"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              </button>
            </div>
          </div>
        </div>

        {/* Pinned Card Section */}
        {pinnedCard && (
          <div className="flex-shrink-0 px-4 py-2 border-b" style={{ borderColor: 'var(--hn-border)' }}>
            <div className="relative group">
              <button
                onClick={removePinnedCard}
                className="absolute top-2 right-2 z-10 w-4 h-4 rounded flex items-center justify-center text-white bg-[#ff6600] hover:bg-[#e55a00] transition-colors text-xs leading-none"
                title="Remove context"
              >
                ×
              </button>
              <div 
                className="compact-pinned-card"
                style={{
                  backgroundColor: '#f6f6ef',
                  border: '2px solid var(--hn-blue)',
                  borderRadius: '0.5rem',
                  padding: '0.75rem',
                  height: '70px',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'space-between'
                }}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    <span 
                      className="text-[11px] tracking-wide uppercase font-semibold rounded px-2 py-1 text-white bg-[#ff6600]"
                    >
                      {pinnedCard.category}
                    </span>
                    <div className="flex flex-col flex-1 min-w-0">
                      <h2 className="text-base font-semibold leading-tight" style={{ color: 'var(--primary)' }}>
                        {pinnedCard.theme}
                      </h2>
                      <a
                        href={pinnedCard.story_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-1 text-[#0066cc] hover:underline text-xs truncate"
                        title={pinnedCard.story_title}
                      >
                        <span className="truncate">{pinnedCard.story_title}</span>
                        <ExternalLink className="w-3 h-3 text-[#999] flex-shrink-0" />
                      </a>
                    </div>
                  </div>
                  <div className="flex-shrink-0 ml-2 mr-8">
                    <a
                      href={pinnedCard.anchor}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex flex-col items-start gap-0.5 text-[#0066cc] hover:underline text-xs font-mono leading-tight"
                    >
                      <div className="flex items-center gap-1">
                        <MessageCircle className="w-3 h-3" />
                        <span>Join the thread</span>
                      </div>
                      <span className="text-[#999] ml-4">{pinnedCard.comment_count} comments</span>
                    </a>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Chat Messages */}
        <div 
          className="flex-1 overflow-auto p-4 min-h-0"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <div className="flex flex-col justify-end min-h-full">
            <div className="flex-1"></div> {/* Spacer to push messages to bottom when few messages */}
            <div className="space-y-4">
              {activeConversation.messages.map((msg) => (
                <MessageBubble key={msg.id} message={msg} />
              ))}
              {streamingMessage && (
                <MessageBubble message={streamingMessage} />
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>
        </div>

        {/* Input Area */}
        <div className="flex-shrink-0 p-4 border-t" style={{ borderColor: 'var(--hn-border)' }}>
          <div className="w-full">
            <textarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder={isModelLoading ? "Loading model, please wait..." : "Enter to send"}
              disabled={isLoading || isModelLoading}
              className="chat-input w-full p-3 rounded resize-none focus:outline-none"
              rows={1}
              style={{ minHeight: '44px', maxHeight: '120px' }}
            />
            
            {/* Status indicator - small reserved space, prominent when active */}
            <div className="flex items-center justify-center mt-2 h-6">
              {isModelLoading && (
                <div className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-secondary)' }}>
                  <div className="animate-spin rounded-full h-3 w-3 border" style={{ borderColor: 'var(--text-secondary)', borderTopColor: 'var(--hn-blue)' }}></div>
                  <span>Loading model...</span>
                  <div className="w-24 h-1 rounded-full" style={{ backgroundColor: 'var(--hn-border)' }}>
                    <div 
                      className="h-1 rounded-full transition-all duration-300"
                      style={{ 
                        width: `${Math.round(modelLoadingProgress * 100)}%`,
                        backgroundColor: 'var(--hn-blue)'
                      }}
                    />
                  </div>
                  <span>{Math.round(modelLoadingProgress * 100)}%</span>
                </div>
              )}
              {error && (
                <div className="flex items-center gap-2 text-xs" style={{ color: '#ef4444' }}>
                  <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                  </svg>
                  <span>{getUserFriendlyErrorMessage(error)}</span>
                </div>
              )}
              {tempStatus && !isModelLoading && !error && (
                <div className="text-xs transition-opacity duration-200" style={{ color: 'var(--hn-blue)' }}>
                  {tempStatus}
                </div>
              )}
              {!tempStatus && !isModelLoading && !error && (
                <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                  💬 Search Hacker News or start chatting
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

    </div>

    {/* Modals rendered outside the main layout container */}
    {/* Memory Modal */}
    <Modal
      isOpen={isMemoryModalOpen}
      onClose={() => setIsMemoryModalOpen(false)}
      title="Memory"
    >
      <ActivityPage 
        activeConversationId={activeConversation?.id}
        onSelectConversation={(id) => {
          const conversation = conversations.find(conv => conv.id === id);
          if (conversation) {
            onConversationUpdated(conversation);
            setIsMemoryModalOpen(false);
          }
        }}
        onDeleteConversation={async (id) => {
          try {
            await ConversationManager.deleteConversation(id);
            
            // If deleted conversation was active, we need to handle this at the parent level
            if (activeConversation?.id === id) {
              // Close modal and let parent handle conversation switching
              setIsMemoryModalOpen(false);
              window.location.reload(); // Simple way to reset to initial state
            }
          } catch (error) {
            logger.error('Error deleting conversation:', error);
          }
        }}
      />
    </Modal>

    {/* Settings Modal */}
    <Modal
      isOpen={isSettingsModalOpen}
      onClose={() => setIsSettingsModalOpen(false)}
      title="Settings"
    >
      <SettingsPage />
    </Modal>
    </>
  );
};