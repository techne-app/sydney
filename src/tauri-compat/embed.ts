/**
 * Tauri-compatible version of src/background/embed.js
 *
 * Identical to the original except the CPU fallback uses device: "wasm"
 * instead of device: "cpu". WKWebView (Tauri on macOS) does not support
 * the "cpu" device — valid options are "webgpu" and "wasm" only.
 */

import { pipeline } from "@huggingface/transformers";
import { logger } from '../utils/logger';

class EmbedderSingleton {
  static fn: any = null;
  static instance: any = null;
  static promise_chain: Promise<any> | null = null;

  static async getInstance(progress_callback?: (data: any) => void) {
    return (this.fn ??= async (...args: any[]) => {
      try {
        this.instance ??= this.initializePipeline(progress_callback);
        return (this.promise_chain = (
          this.promise_chain ?? Promise.resolve()
        ).then(async () => (await this.instance)(...args)));
      } catch (error) {
        logger.error('Error in EmbedderSingleton getInstance:', error);
        throw error;
      }
    });
  }

  static async initializePipeline(progress_callback?: (data: any) => void) {
    try {
      return await pipeline(
        "feature-extraction",
        "Xenova/all-MiniLM-L6-v2",
        {
          progress_callback,
          device: "webgpu",
          session_options: { logSeverityLevel: 4 },
        } as any,
      );
    } catch (error) {
      // Use "wasm" instead of "cpu" — WKWebView doesn't support "cpu"
      logger.debug('WebGPU not available, falling back to wasm:', error);
      return await pipeline(
        "feature-extraction",
        "Xenova/all-MiniLM-L6-v2",
        {
          progress_callback,
          device: "wasm",
          session_options: { logSeverityLevel: 4 },
        } as any,
      );
    }
  }
}

export const embed_tags = async (tags: string[]) => {
  try {
    const embedder = await EmbedderSingleton.getInstance((_data: any) => {});
    const embeddings = await Promise.all(
      tags.map(async (tag: string) => {
        try {
          return await embedder(tag, { pooling: 'mean', normalize: true });
        } catch (error) {
          logger.debug('Error embedding tag:', tag, error);
          return { data: new Float32Array(384), dims: [1, 384] };
        }
      })
    );
    return embeddings;
  } catch (error) {
    logger.error('Error in embed_tags:', error);
    return tags.map(() => ({ data: new Float32Array(384), dims: [1, 384] }));
  }
};
