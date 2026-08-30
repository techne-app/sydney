"""
Tool schemas — the single source of truth for what the router LLM can call.

These are the two Phase-0 tools, expressed in OpenAI / JSON-Schema tool format
(the shape llama-cpp-python's native tool-calling consumes). The router endpoint
feeds this list to the model; the eval harness imports the same list so runtime
and eval can never drift.

DESIGN RULE — the schema advertises ONLY what the current executor actually
honors. We do not expose parameters the tool can't act on yet (that would make the
model promise behavior that doesn't happen). Params grow later, when execution
grows to honor them.

Content harvested from the existing extension:
  - descriptions + demonstrative guidance : src/prompts/actionOnly.ts
  - param shapes                          : mlc_llm/bfcl_testcases.json

Executor reality this reflects (Phase 0):
  - search_threads         -> SearchService.executeSearchStreaming() uses ONLY
                              keyword_filter (today's search is the top-30
                              semantic-match; sort/time/count are not honored yet).
  - summarize_pinned_thread -> ThreadSummaryTool returns the pinned card's stored
                              summary as-is. It honors NO parameters.
"""

SEARCH_THREADS = {
    "type": "function",
    "function": {
        "name": "search_threads",
        "description": (
            "Search Hacker News discussion threads by topic. Use this when the user "
            "wants to find, look up, or be shown discussions about something "
            "(e.g. \"find AI threads\", \"any posts about rust?\", \"show me startup "
            "discussions\")."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keyword_filter": {
                    "type": "string",
                    "description": "The topic or keywords to search discussions for.",
                },
            },
            "required": ["keyword_filter"],
        },
    },
}

SUMMARIZE_PINNED_THREAD = {
    "type": "function",
    "function": {
        "name": "summarize_pinned_thread",
        # Demonstrative guidance folded in from actionOnly.ts so the model maps
        # "this thread / this discussion / this / here / what's this about / key
        # points / tl;dr" onto the pinned thread. Only valid when a thread is
        # pinned (the router passes pinned-thread context at call time).
        "description": (
            "Summarize the thread the user currently has pinned/open. Use this "
            "whenever the user refers to what they're looking at — \"this thread\", "
            "\"this discussion\", \"this\", \"here\", \"what's this about\", or asks "
            "for the key points / tl;dr of the pinned thread."
        ),
        # No parameters: the executor ignores format/focus and returns the stored
        # summary as-is. Exposing them would advertise behavior that doesn't exist.
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}

# The list handed to the model / imported by the eval.
TOOLS = [SEARCH_THREADS, SUMMARIZE_PINNED_THREAD]

# Convenience: the set of valid tool names, for the executor to route on.
TOOL_NAMES = [t["function"]["name"] for t in TOOLS]
