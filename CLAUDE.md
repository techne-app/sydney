# Techne Browser Extension

## Overview
Techne is a sophisticated browser extension that enhances the Hacker News experience by providing AI-powered content analysis, personalized tag recommendations, and conversational search capabilities. It helps users discover relevant discussions and understand nuanced tech conversations through a chat-first interface powered by local AI models.

## Key Features
- **AI-Powered Content Tagging**: Automatic tag generation for HN stories and comments
- **Personalized Tag Ranking**: ML-based tag ranking using user browsing history
- **Context-Aware Chat Interface**: Chat-first interface with live context sidebar showing relevant HN threads
- **Conversational Search**: LLM-powered intent detection for natural language search
- **Local AI Chat**: Fully local WebLLM-powered conversational interface with search integration
- **User Data Management**: Local storage with IndexedDB for privacy
- **Streamlined UI**: Pure chat-first interface with contextual sidebar and essential modals

## Tech Stack
- **Frontend**: React 19 + TypeScript
- **Styling**: Tailwind CSS
- **Build**: Webpack 5 + ts-loader
- **AI/ML**: Hugging Face Transformers.js, ONNX Runtime Web, WebLLM
- **Database**: Dexie.js (IndexedDB wrapper)
- **Extension**: Chrome Manifest V3

## Development

### Setup
```bash
npm install
```

### Build Commands
```bash
npm run build       # Production build
npm run dev         # Development build with watch mode
```

### Version Management
```bash
npm run version:patch    # 1.8.0 → 1.8.1
npm run version:minor    # 1.8.0 → 1.9.0
npm run version:major    # 1.8.0 → 2.0.0
```

The version script automatically updates both `package.json` and `public/manifest.json`.

### Testing Extension
1. Run `npm run build` to create the `dist/` folder
2. Open Chrome and navigate to `chrome://extensions/`
3. Enable "Developer mode"
4. Click "Load unpacked" and select the `dist/` folder
5. Navigate to Hacker News to see the extension in action

## Testing Infrastructure

### Testing Framework: Jest
This project uses **Jest** as the primary testing framework, chosen for its comprehensive Chrome extension and React support:

```bash
npm test              # Run all tests
npm run test:watch    # Run tests in watch mode  
npm run test:coverage # Run with coverage report
```

### Jest Configuration
- **Config File**: `jest.config.cjs` (CommonJS for ES module compatibility)
- **Test Environment**: jsdom for DOM manipulation testing
- **TypeScript Support**: ts-jest with proper ES module handling
- **Coverage**: HTML and LCOV reports generated in `coverage/` directory

### Testing Approach

#### 1. Pure Function Testing (High Priority)
**Focus**: Business logic with clear inputs/outputs

**Examples**:
- `src/utils/intentDetector.ts` - JSON parsing, validation, prompt building
- `src/background/personalize.ts` - Mathematical operations, similarity calculations  
- `src/utils/configStore.ts` - Configuration management, default handling

**Why**: These are the most reliable to test and provide the highest value for bug prevention.

#### 2. Chrome Extension API Mocking
**Global Mock Setup** (`src/test-setup.ts`):
```javascript
global.chrome = {
  runtime: {
    sendMessage: jest.fn().mockResolvedValue(undefined),
    onMessage: { addListener: jest.fn(), removeListener: jest.fn() }
  },
  tabs: { create: jest.fn(), update: jest.fn() }
};
```

**Best Practice**: Always mock Chrome APIs to avoid "Receiving end does not exist" errors during testing.

#### 3. External Library Mocking
**WebLLM & Transformers.js**: Heavy dependencies mocked to avoid ES module conflicts
```javascript
jest.mock('@mlc-ai/web-llm');
jest.mock('@huggingface/transformers');
```

### Test Coverage Strategy

#### Current Coverage (Intent Detection: ~98%)
- ✅ **JSON parsing edge cases** - Malformed JSON, invalid types, missing fields
- ✅ **Async operations** - Promise handling, error scenarios, timeouts
- ✅ **Configuration validation** - Type checking, bounds validation, defaults
- ✅ **Chrome API integration** - Message passing, error handling

#### Future Testing Priorities
1. **Search Service** - API integration, result formatting, error handling
2. **React Components** - User interactions, state management, props validation
3. **Background Scripts** - Message routing, ML pipeline integration
4. **Content Scripts** - DOM manipulation, HN page integration

### Testing Best Practices

#### Jest + Chrome Extensions
- Use `jest.config.cjs` for CommonJS compatibility with ES modules
- Mock all Chrome APIs globally to prevent runtime errors
- Test async operations with proper Promise handling
- Validate both success and error paths for all functions

#### Coverage Goals
- **Pure Functions**: Aim for 95%+ coverage
- **Integration Points**: Focus on error handling and edge cases  
- **UI Components**: Test user interactions and state changes
- **Overall**: Maintain >80% line coverage across the project

#### Testing Commands
```bash
npm test                    # Run all tests once
npm run test:watch         # Development mode with file watching
npm run test:coverage      # Generate detailed coverage report
```

Coverage reports are generated in `coverage/` and ignored by git. Open `coverage/lcov-report/index.html` to view detailed coverage analysis.

### GitHub Actions Integration
Tests run automatically on every push and gate deployments:
- **Tests must pass** before version bumping
- **Build must succeed** before Chrome Web Store deployment
- **Coverage reports** generated in CI for tracking trends

## Project Structure
```
src/
├── background/           # Service worker and ML processing
│   ├── index.ts         # Main background script
│   ├── embed.js         # Text embedding functionality
│   ├── personalize.ts   # Tag ranking algorithms
│   └── contextDb.ts     # Database operations
├── content-scripts/     # Page-specific injection scripts
│   ├── main-page/       # HN front page functionality
│   ├── item-page/       # Individual story/comment pages
│   └── profile-page/    # User profile pages
├── popup/               # Extension popup UI
│   ├── index.tsx        # Main popup component
│   └── components/      # React components
├── utils/               # Shared utilities
├── types/               # TypeScript definitions
└── styles/              # CSS/Tailwind styles
```

## Configuration Architecture

This project uses **two separate configuration systems** that serve distinct purposes. Understanding when to use each is critical for maintainability.

### `src/config.ts` - Static Application Configuration
**Purpose**: Immutable application constants determined at build time.

**Contents**:
- **API URLs**: Backend endpoints and base URLs
- **UI Styling**: Default colors and styling constants  
- **Business Rules**: MAX_STORY_TAGS, etc.
- **DEFAULT_MODEL**: Default AI model (`Llama-3.2-3B-Instruct-q4f16_1-MLC`)

**Characteristics**:
- **Synchronous access**: Direct property access (`CONFIG.MAX_STORY_TAGS`)
- **No dependencies**: No database or async operations
- **Build-time values**: Environment-dependent (dev vs prod)
- **High performance**: Values cached in memory

**When to use**: API endpoints, styling constants, business rules, feature defaults that should never change during runtime.

### `src/utils/configStore.ts` - Dynamic User Configuration
**Purpose**: Mutable user preferences that persist across browser sessions.

**Contents**:
- **Chat Configuration**: Model selection, temperature, topP, maxTokens
- **User Preferences**: Log levels and other personalization settings
- **Database Integration**: Async methods for IndexedDB persistence

**Characteristics**:
- **Asynchronous access**: All methods return Promises (`await configStore.getModel()`)
- **Database-backed**: Values persisted in IndexedDB via `contextDb`
- **User-configurable**: Settings that users can modify (even if UI is removed)
- **Error handling**: Fallback to static defaults on database errors

**When to use**: User preferences, settings that need persistence, values that might change during runtime.

### **Architecture Decision: Why Two Systems?**

**DO NOT consolidate these systems**. They serve fundamentally different purposes:

1. **Performance**: Static config provides immediate access; dynamic config requires async database calls
2. **Lifecycle**: Static values determined at build time; dynamic values change throughout user sessions  
3. **Testing**: Static config easily mockable; dynamic config requires database mocking
4. **Maintenance**: Different deployment implications for static vs user-configurable values

**Rule of Thumb**: If a value could theoretically be user-configurable or needs to persist across sessions, use `configStore`. If it's an application constant that should never change during runtime, use `CONFIG`.


## UI Components

**Purpose**: Reusable UI components with consistent styling and behavior.

**Key Components**:
- **`ThreadCard`**: Individual HN discussion cards with HN-style theming, supports compact sidebar variant
- **`Tag`**: Category and theme tags with processing states
- **`Card`**: Base card component for consistent styling
- **`Modal`**: Reusable modal system for secondary interfaces

### CSS Architecture

The extension uses **CSS custom properties** for consistent theming:

```css
:root {
  --hn-bg: #f6f6ef;           /* HN beige background */
  --hn-text: #333;            /* Primary text color */
  --hn-orange: #ff6600;       /* HN accent color */
  --hn-link: #0066cc;         /* Link color */
  --hn-gray: #999;            /* Secondary text */
  --hn-border: #e0e0e0;       /* Borders and dividers */
  --card-bg: var(--hn-bg);    /* Card backgrounds */
  --card-text: var(--hn-text); /* Card text */
}
```

**Component Usage**:
```jsx
// Components use CSS variables for theming
<div style={{ backgroundColor: 'var(--card-bg)', color: 'var(--hn-orange)' }}>
```

**Extension CSS Structure** (from `src/styles/popup.css`):
```css
/* Import Tailwind for utilities */
@import "tailwindcss";

/* Extension-specific utilities */
@layer components {
  .animate-subtle-glow { /* Orange ring animation */ }
  .line-clamp-2 { /* Text truncation */ }
}
```

### `public/manifest.json`
Extension configuration and permissions. Key settings:
- **manifest_version**: 3 (Chrome MV3)
- **permissions**: ["tabs"]
- **content_scripts**: Injected into HN pages
- **web_accessible_resources**: Extension assets
- **minimum_chrome_version**: 92

### `feature-flags.json`
Feature toggle configuration:
```json
{
  "tag_personalization": true,
  "chat_interface": true,
  "tag_search": true
}
```
Note: `tag_search` controls search functionality within the chat interface. The standalone search tab has been removed.

### `webpack.config.js`
Build configuration with multiple entry points:
- Background script
- Popup interface
- Content scripts for different HN pages

## API Integration
- **Techne Backend**: Azure Functions at `techne-pipeline-func-prod.azurewebsites.net`
- **Story Tags**: `/api/story-tags/` endpoint
- **Thread Tags**: `/api/thread-tags/` endpoint
- **Hacker News API**: Firebase API for story metadata

## Key Dependencies
- `@huggingface/transformers`: AI text embeddings
- `@mlc-ai/web-llm`: Local AI chat models
- `dexie`: IndexedDB operations
- `react`: UI framework
- `downshift`: Autocomplete components
- `progressbar.js`: Loading indicators

## Tag System
The extension supports multiple tag types:
- `thread_theme`: Discussion themes (used on main page and chat search)
- `thread_category`: Topic categories (used on item and profile pages)

Tags are visually integrated into the HN UI with a maximum of 3 tags per story.

## Data Storage
All user data is stored locally using IndexedDB:
- **Tag History**: Clicked tags with timestamps
- **Search History**: Previous search queries
- **Chat Conversations**: Conversation history with messages
- **Settings**: User preferences and feature toggles
- **Embeddings**: Cached text embeddings for performance

## User Interface
The popup provides a pure chat-first interface with contextual awareness and essential modals:

### Primary Interface
- **Context Sidebar**: Left sidebar displaying 3 live HN thread cards as conversation context, fetched from the backend API
- **Chat Interface**: Main area providing conversational AI and search functionality with LLM-powered intent detection
- **Integrated Layout**: Context cards and chat work together to provide relevant discussion awareness

### Context Thread Cards
- **Live Content**: Three relevant HN discussions updated from the backend API (24-hour lookback, karma density sorted)
- **Compact Design**: 260px height cards with generous internal spacing, showing category, theme, comment count, and story links
- **No Summary**: Streamlined to show essential information without cluttering the sidebar
- **Direct Links**: "Join the thread" links navigate directly to HN discussions

### Modal Interface (Accessible from Chat Header Icons)
- **Memory Modal**: History of visited threads, clicked tags, and recent searches displayed in a two-column layout (accessible via bookmark icon in chat header)  
- **Settings Modal**: User preferences and feature toggles (accessible via gear icon in chat header)

The interface provides constant contextual awareness through the sidebar while maintaining the clean chat-first design. The context cards serve as reference material for ongoing conversations without requiring modal interactions.

### Modal Architecture
The extension uses a reusable modal system for secondary interfaces:

**Modal Component Features:**
- **Fixed Sizing**: Consistent modal size (max-w-4xl width, 80vh height) regardless of content
- **Backdrop Dismissal**: Click outside modal to close
- **Header with Close Button**: Standard modal header with title and X button
- **Scroll Handling**: Content area scrolls independently when needed
- **Dark Theme Integration**: Styled to match the dark extension theme

**Current Modals:**
- **Memory Modal**: Two-column layout with "Visited Threads" and "Recent Searches"
- **Settings Modal**: Configuration interface for user preferences

**Modal Triggers:**
- Memory: Bookmark icon in chat interface header
- Settings: Gear icon in chat interface header

## Security & Privacy
- **Local-First**: All data stored locally
- **No Cross-Site Tracking**: Only operates on HN domains
- **Content Security Policy**: Strict CSP with WebAssembly support
- **Minimum Chrome Version**: Requires Chrome 92+

## Performance
- **Lazy Loading**: AI models loaded on-demand
- **Efficient Caching**: Embeddings cached to avoid recomputation
- **Memory Management**: Singleton pattern for ML models
- **Batch Processing**: Tags processed in batches

## Intent Detection Architecture

The extension uses a **two-step inference approach** with local AI models for intelligent message routing between search, conversation, and pinned thread summarization.

### Two-Step Architecture
1. **Step 1: Intent Classification** (`src/prompts/intentOnly.ts`)
   - Determines whether user wants **action** (search/summarize) or **chat** (conversation)
   - Context-aware prompts that understand pinned thread context
   - Achieves 90%+ accuracy on pinned thread cases

2. **Step 2: Function Selection** (`src/prompts/actionOnly.ts`) 
   - Executes only when Step 1 identifies "action" intent
   - Selects specific function: `get_thread_cards` (search) or `summarize_pinned_thread`
   - Returns structured function calls with parameters

### Production Implementation
- **Single API**: `IntentDetector.detectIntent()` method in `src/utils/intentDetector.ts`
- **Clean Architecture**: All legacy methods removed in favor of evaluated two-step approach
- **Context Integration**: Pinned thread data automatically included in both steps
- **Error Handling**: Graceful fallbacks for parsing errors and low confidence results

### Model Performance
- **Recommended Model**: Phi-3.5-mini-instruct-q4f16_1-MLC
- **Overall Accuracy**: 76.3% on full dataset (158 test cases)
- **Pinned Thread Accuracy**: 90% with context-aware prompts
- **Evaluation Framework**: `mlc_llm/eval_two_step.py` for comprehensive testing
- **Testing**: `python mlc_llm/quick-intent-test.py "your query"`

### Key Features
- **Context-Aware**: Understands when user has pinned a thread for analysis
- **High Accuracy**: Dramatic improvement from 20-60% to 90% on context cases
- **Production-Tested**: Implementation matches evaluated architecture exactly
- **Dual Confidence**: Combined confidence scoring from both inference steps

## Development Notes
- Extension uses Chrome Extension Manifest V3
- All content scripts inject into specific HN page patterns
- WebAssembly support enabled for ML model execution
- TypeScript strict mode enabled for type safety
- Tailwind CSS for responsive design
- Chat interface runs entirely locally with WebLLM models
- Context-aware interface with live sidebar showing relevant HN threads
- Essential features (Memory, Settings) implemented as modals accessible from chat header
- Modal system provides consistent UI with fixed sizing and clean backdrop dismissal
- Sidebar + chat layout architecture with contextual thread awareness

### Logging
This codebase uses a custom logger system (`src/utils/logger.ts`) instead of direct console statements:
```typescript
import { logger } from '../utils/logger';
logger.debug('message');  // Instead of console.log()
logger.error('error');    // Instead of console.error()
```
Available methods: `debug()`, `info()`, `warn()`, `error()`, `search()`, `chat()`, `model()`, `database()`, `api()`, `intent()`. Development shows all logs, production only shows errors.

## Agentic Search Evolution

The following phases outline the roadmap for evolving from current chat-first search to powerful agentic search capabilities, preparing for MCP (Model Context Protocol) integration.

### Current State (Phase 1 - Completed)
- **Chat-First Interface**: Removed standalone search tab, all search happens through conversational interface
- **SearchService Integration**: Backend search functionality preserved and accessible via chat
- **Intent Detection Foundation**: Basic intent detection with `IntentDetector` utility using local LLM
- **Unified UX**: Single interface for both chat and search, with recent searches in Memory tab
- **Backend**: Azure Functions with basic tag APIs, evolving toward MCP server capabilities

### Phase 2: Enhanced Intent Detection & Function Calling
- **Advanced Intent Detection**: Improve LLM-based intent detection with more sophisticated patterns
- **Function Calling Patterns**: Implement structured function calling for search operations
- **Multi-Step Reasoning**: Enable chat interface to handle complex, multi-part search queries
- **Search Context Management**: Maintain search context across conversation turns
- **Proof of Concept**: Use as testing ground for future MCP tool integration patterns

### Phase 3: MCP-Ready Chat Architecture
- **Tool-Oriented Design**: Redesign chat interface to support tool calling patterns
- **Message Routing**: Create extensible message handling that can route to different "tools"
- **Context Management**: Build conversation context management for multi-step reasoning
- **Function Calling**: Prepare infrastructure for future MCP function calling capabilities
- **Backward Compatibility**: Maintain all existing chat functionality

### Phase 4: MCP Integration Layer
- **MCP Client**: Implement MCP client capabilities in chat interface
- **Backend Connection**: Connect to backend MCP servers when they become available
- **Tool Orchestration**: Create system for managing multiple MCP tools and their interactions
- **Historical Data**: Access backend's historical HN data and vector indexing capabilities
- **Agent Behavior**: Enable multi-step reasoning: "find AI discussions from 2023, then show related startups"

### Phase 5: Full Agentic Search
- **Historical Analysis**: Search across years of HN data with powerful backend vector indexing
- **Multi-Step Reasoning**: Complex queries that require multiple API calls and context building
- **Topic Evolution**: Track how discussions evolve over time periods
- **Pattern Recognition**: Identify trends, user patterns, and emerging topics
- **Contextual Intelligence**: Build understanding across related discussions and time periods

### Key Design Principles
- **Chat-First**: All search functionality integrated into conversational interface
- **MCP-First**: Architecture designed for MCP server integration from the start
- **Tool Orchestration**: Chat interface as intelligent tool coordinator
- **Progressive Enhancement**: Evolve from current chat+search integration to full agentic capabilities
- **Future-Proof**: Can scale from simple intent detection to complex multi-step reasoning
- **Black Box Backend**: Design assumes backend is extensible black box that will expose MCP servers

### Technical Implementation Strategy
- **LLM-Based Intent Routing**: Use Llama 3.2's function calling capabilities for intent detection
- **Unified Interface**: Single chat interface handles both conversational AI and search requests
- **Extensible Architecture**: Chat interface designed to handle any number of MCP tools
- **Function Calling**: Conversation handling that can orchestrate multiple backend calls
- **Context Preservation**: Multi-step reasoning with conversation memory and context building
- **SearchService Integration**: Existing search functionality accessible through chat intent detection
- **Feature Flags**: Control rollout of agentic capabilities as backend evolves

### Modern Agentic Search Patterns (2024)
- **Hybrid Routing Systems**: Combine LLMs with traditional methods for optimal performance
- **Intent Detection Evolution**: LLMs excel at understanding user intent from natural language
- **Manual Function Calling**: WebLLM supports manual function calling with JSON parsing
- **Uncertainty-Based Routing**: Route between different approaches based on confidence scores
- **Structured Output Parsing**: Manual JSON parsing for tool calling until OpenAI API compatibility arrives

### Benefits of Chat-First Agentic Approach
- **Unified Experience**: Single interface for all user interactions eliminates context switching
- **Natural Language**: Conversational search that understands intent and context
- **Progressive Capability**: Can evolve from simple search to complex multi-step reasoning
- **Powerful Search**: Future access to historical data and sophisticated vector indexing via MCP
- **Intelligent Reasoning**: Multi-step queries that go beyond simple keyword matching
- **Scalable Architecture**: Can accommodate any backend MCP tools as they become available
- **Enhanced Discovery**: Find patterns and connections across time periods and topics

## Chrome Extension Promise Handling Guidelines

### Critical Chrome Extension Promise Error Patterns

When working with Chrome extension APIs, always handle promises properly to avoid uncaught promise rejections that create user-facing console errors.

#### Chrome Runtime Messages
**Problem**: `chrome.runtime.sendMessage()` returns a Promise that can reject with "Could not establish connection. Receiving end does not exist" when the receiving end is closed/unavailable.

**Solution**: Always add `.catch()` handlers to all `chrome.runtime.sendMessage()` calls:

```javascript
// ❌ BAD - Can cause uncaught promise rejection
chrome.runtime.sendMessage({
  type: 'SOME_MESSAGE',
  data: {}
});

// ✅ GOOD - Handles promise rejection
chrome.runtime.sendMessage({
  type: 'SOME_MESSAGE', 
  data: {}
}).catch((error) => {
  logger.debug('No listeners for message, this is expected');
});
```

#### Async Callbacks in Streaming Operations
**Problem**: When passing async callbacks to functions (like `onProgress` callbacks), the callback's Promise can reject and become uncaught.

**Solution**: Always await async callbacks and wrap them in try-catch:

```javascript
// ❌ BAD - Async callback not awaited
if (onProgress) {
  onProgress(content); // Returns unhandled Promise
}

// ✅ GOOD - Async callback properly awaited
if (onProgress) {
  await onProgress(content); // Await the Promise
}
```

#### Background Script Message Responses
**Problem**: Background scripts sending response messages back to UI can fail if UI is closed, causing uncaught rejections.

**Solution**: Combine try-catch blocks with `.catch()` handlers:

```javascript
// ✅ GOOD - Dual error handling
try {
  chrome.runtime.sendMessage({
    type: 'RESPONSE_MESSAGE',
    data: results
  }).catch((error) => {
    logger.debug('No listeners for response, this is expected');
  });
} catch (error) {
  logger.debug('Synchronous error in sendMessage');
}
```

### Transformers.js Browser Compatibility

#### Webpack Configuration for Transformers.js 3.6+
**Problem**: Newer versions include Node.js dependencies that break browser builds.

**Solution**: Use the dedicated browser build via webpack alias:

```javascript
// webpack.config.js
resolve: {
  alias: {
    "@huggingface/transformers": path.resolve(
      __dirname,
      "node_modules/@huggingface/transformers/dist/transformers.web.js"
    ),
  }
}
```

#### ONNX Runtime Logging Suppression
**Problem**: ONNX runtime produces verbose console logs that concern users.

**Solution**: Set `logSeverityLevel: 4` in pipeline session options:

```javascript
const pipeline = await pipeline("feature-extraction", "model-name", {
  device: "webgpu",
  session_options: {
    logSeverityLevel: 4, // Fatal only - suppress all non-fatal logs
  },
});
```

### Key Development Practices
- **Always use `.catch()` on Chrome extension promises** to prevent user-facing console errors
- **Await async callbacks** in streaming operations to handle their promise rejections
- **Use browser-specific builds** for ML libraries when available
- **Test extension message passing** with popup closed to verify error handling
- **Suppress non-fatal logs** in production to avoid alarming usersThis change to CLAUDE.md should NOT trigger the deployment pipeline.

## Testing Notes
- Path-ignore configuration should prevent CI triggers for documentation changes
- This edit should NOT cause version bumping or deployment
