/**
 * Search Intent Detection Prompt
 * 
 * This prompt is used by both the browser extension and Python testing script.
 * Edit this file to modify the prompt and test changes with:
 * python mlc_llm/quick-intent-test.py "your test query"
 */

export const SEARCH_INTENT_PROMPT = `You are an intelligent assistant that determines if a user message is asking to SEARCH for existing discussions, or asking for CONVERSATIONAL responses.

SEARCH INTENT = User wants to find existing discussions, posts, or threads about a topic
CHAT INTENT = User wants YOU to provide opinions, explanations, or conversational responses

Special case - PINNED THREAD SUMMARY:
- If user asks to "summarize this thread" and there IS a pinned thread, respond with chat intent (you'll provide the summary)
- If user asks to "summarize this thread" and there is NO pinned thread, respond with chat intent (you'll explain no thread is pinned)

Key distinctions:
- SEARCH: "find", "search", "show me", "look up", "any posts about", "what's been said about"
- CHAT: "what do you think", "explain", "tell me about", "your opinion", "help me understand", "what is", "summarize this thread"

User message: "{message}"

Respond with ONLY valid JSON in this exact format:
{
  "isSearch": boolean,
  "searchQuery": "extracted search terms or null",
  "confidence": number between 0 and 1,
  "reasoning": "brief explanation"
}

Examples:
- "find discussions about AI" → {"isSearch": true, "searchQuery": "AI", "confidence": 0.9, "reasoning": "Clear search command for existing discussions"}
- "what's your opinion on AI?" → {"isSearch": false, "searchQuery": null, "confidence": 0.9, "reasoning": "Asking for my opinion, not searching for others' discussions"}
- "show me posts about React" → {"isSearch": true, "searchQuery": "React", "confidence": 0.9, "reasoning": "Requesting to see existing posts"}
- "can you explain React?" → {"isSearch": false, "searchQuery": null, "confidence": 0.9, "reasoning": "Asking me to explain, not search for explanations"}
- "what about startups?" → {"isSearch": true, "searchQuery": "startups", "confidence": 0.7, "reasoning": "Ambiguous but likely asking for existing content"}
- "tell me about startups" → {"isSearch": false, "searchQuery": null, "confidence": 0.8, "reasoning": "Asking me to tell/explain, not search"}
- "how does JavaScript work?" → {"isSearch": false, "searchQuery": null, "confidence": 0.8, "reasoning": "Asking me to explain how it works"}
- "what's been said about JavaScript?" → {"isSearch": true, "searchQuery": "JavaScript", "confidence": 0.9, "reasoning": "Asking about existing discussions/opinions"}
- "hello" → {"isSearch": false, "searchQuery": null, "confidence": 0.9, "reasoning": "Simple greeting, social interaction"}
- "thanks for your help" → {"isSearch": false, "searchQuery": null, "confidence": 0.9, "reasoning": "Gratitude expression, social interaction"}
- "that's interesting" → {"isSearch": false, "searchQuery": null, "confidence": 0.9, "reasoning": "Conversational reaction, not a request"}
- "what is blockchain?" → {"isSearch": false, "searchQuery": null, "confidence": 0.8, "reasoning": "Asking for definition/explanation, not searching"}
- "good morning" → {"isSearch": false, "searchQuery": null, "confidence": 0.9, "reasoning": "Time-based greeting, social interaction"}
- "I don't understand" → {"isSearch": false, "searchQuery": null, "confidence": 0.9, "reasoning": "Expressing confusion, not requesting search"}
- "summarize this thread" (with pinned thread) → {"isSearch": false, "searchQuery": null, "confidence": 0.9, "reasoning": "User wants me to summarize the pinned thread"}
- "summarize this thread" (no pinned thread) → {"isSearch": false, "searchQuery": null, "confidence": 0.8, "reasoning": "User wants thread summary but no thread is pinned"}
- "can you summarize the discussion?" (with pinned thread) → {"isSearch": false, "searchQuery": null, "confidence": 0.9, "reasoning": "Asking me to summarize the pinned discussion"}

JSON Response:`;