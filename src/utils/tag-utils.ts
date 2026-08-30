import { CONFIG } from '../config';
import { StoryData, ThreadData, Tag } from '../types';
import { MessageType, NewTagRequest } from '../types/messages';
import { logger } from './logger';

export async function fetchStoryTags(
    storyIds: number[],
    maxTagsPerStory?: number,
    limitTagsPerStory?: boolean,
    tagTypes?: string[]
): Promise<StoryData[]> {
    try {
        const response = await fetch(`${CONFIG.BASE_URL}${CONFIG.ENDPOINTS.STORY_TAGS}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                story_ids: storyIds,
                max_tags_per_story: maxTagsPerStory,
                limit_tags_per_story: limitTagsPerStory,
                tag_types: tagTypes
            })
        });
        if (!response.ok) {
            logger.error(`fetchStoryTags: HTTP ${response.status} ${response.statusText} from ${CONFIG.BASE_URL}${CONFIG.ENDPOINTS.STORY_TAGS}`);
            return [];
        }
        return await response.json();
    } catch (error) {
        logger.error('fetchStoryTags failed (network/parse):', error);
        return [];
    }
}

export async function fetchThreadTags(threadIds: number[]): Promise<ThreadData[]> {
    try {
        const response = await fetch(`${CONFIG.BASE_URL}${CONFIG.ENDPOINTS.THREAD_TAGS}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ thread_ids: threadIds })
        });
        if (!response.ok) {
            logger.error(`fetchThreadTags: HTTP ${response.status} ${response.statusText} from ${CONFIG.BASE_URL}${CONFIG.ENDPOINTS.THREAD_TAGS}`);
            return [];
        }
        return await response.json();
    } catch (error) {
        logger.error('fetchThreadTags failed (network/parse):', error);
        return [];
    }
}

export function addStoryTags(subtextElement: Element, storyData: StoryData): void {
    if (!subtextElement || !storyData?.tags?.length || !storyData?.tag_anchors?.length) return;

    // Add line break
    const lineBreak = document.createElement('br');
    subtextElement.appendChild(lineBreak);

    // Add "Threads:" prefix with muted styling
    const threadsPrefix = document.createElement('span');
    threadsPrefix.textContent = 'Threads: ';
    threadsPrefix.style.color = '#828282'; // HN's grey color
    subtextElement.appendChild(threadsPrefix);

    const tagsContainer = document.createElement('span');
    tagsContainer.style.color = CONFIG.STYLES.STORY_TAG.color;
    
    const tagCount = Math.min(storyData.tags.length, CONFIG.MAX_STORY_TAGS);
    for (let i = 0; i < tagCount; i++) {
        // Add pipe separator before tag (except for first tag)
        if (i > 0) {
            const pipeElement = document.createElement('span');
            pipeElement.textContent = ' | ';
            pipeElement.style.color = '#828282'; // Match HN's grey metadata style
            tagsContainer.appendChild(pipeElement);
        }

        const anchorElement = document.createElement('a');
        // Style like HN links - blue with underline on hover
        anchorElement.style.color = '#0000EE'; // HN's blue link color
        anchorElement.style.textDecoration = 'none';
        anchorElement.href = storyData.tag_anchors[i];
        anchorElement.textContent = storyData.tags[i];
        
        // Add hover effect - underline on hover, stay blue
        anchorElement.addEventListener('mouseenter', () => {
            anchorElement.style.textDecoration = 'underline';
        });
        anchorElement.addEventListener('mouseleave', () => {
            anchorElement.style.textDecoration = 'none';
        });
        
        // Add click handler
        anchorElement.addEventListener('click', (e) => {
            e.preventDefault();
            const msg: NewTagRequest = {
                type: MessageType.NEW_TAG,
                data: {
                    tag: storyData.tags[i],
                    type: storyData.tag_types[i],
                    anchor: storyData.tag_anchors[i]
                }
            };
            chrome.runtime.sendMessage(msg).catch((error) => {
                // Ignore "Receiving end does not exist" errors - this is expected when popup is closed
                logger.debug('No listeners for NEW_TAG message, this is expected');
            });
            window.open(storyData.tag_anchors[i], '_blank');
        });
        
        tagsContainer.appendChild(anchorElement);
    }
    subtextElement.appendChild(tagsContainer);
}

export function addThreadTag(element: HTMLElement | null, tags: Tag[], isThread: boolean): void {
    if (!element || !tags?.length) return;
    
    const comheadElement = element.querySelector('.comhead');
    if (!comheadElement) return;

    // Clear any existing tags
    const existingTags = comheadElement.querySelectorAll('.techne-tag');
    existingTags.forEach(tag => tag.remove());
    
    tags.forEach(tag => {
        const tagElement = document.createElement('span');
        tagElement.className = `techne-tag ${tag.type}`;
        Object.assign(tagElement.style, {
            backgroundColor: '#FF6600',
            color: 'black',
            padding: '1px 3px',
            borderRadius: '2px',
            marginLeft: '4px'
        });
        tagElement.textContent = tag.tag || '';
        comheadElement.appendChild(tagElement);
    });
}