export const CONFIG = {
    BASE_URL: process.env.NODE_ENV === 'development'
        ? "https://www.techne.app/api"
        : "https://www.techne.app/api",
    ENDPOINTS: {
        STORY_TAGS: "/story-tags/",
        THREAD_TAGS: "/thread-tags/"
    },
    STYLES: {
        STORY_TAG: { 
            color: 'blue', 
            textDecoration: 'none' 
        },
        TOPIC_TAG: {
            backgroundColor: '#4A90E2',
            color: 'white'
        }
    },
    MAX_STORY_TAGS: 3,
    // Default model configuration - change this to update the default model
    DEFAULT_MODEL: "Llama-3.2-3B-Instruct-q4f16_1-MLC"
}; 