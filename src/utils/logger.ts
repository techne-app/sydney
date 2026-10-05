import { contextDb, SettingKeys } from './contextDb';

export enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3,
  NONE = 4
}

export interface LoggerConfig {
  level: LogLevel;
  prefix: string;
}

class Logger {
  private static instance: Logger;
  private config: LoggerConfig = {
    level: LogLevel.ERROR, // Default to ERROR only
    prefix: 'Techne'
  };
  private initialized = false;

  private constructor() {}

  static getInstance(): Logger {
    if (!Logger.instance) {
      Logger.instance = new Logger();
    }
    return Logger.instance;
  }

  async initialize(): Promise<void> {
    if (this.initialized) return;
    
    try {
      // Load log level from settings, with different defaults for dev/prod
      const defaultLevel = process.env.NODE_ENV === 'development' ? LogLevel.DEBUG : LogLevel.ERROR;
      const storedLevel = await contextDb.getSettingValue(SettingKeys.LOG_LEVEL, defaultLevel);
      
      this.config.level = storedLevel;
      this.initialized = true;
    } catch (error) {
      // Fallback to console.error if initialization fails
      console.error('Failed to initialize logger:', error);
      this.config.level = LogLevel.ERROR;
      this.initialized = true;
    }
  }

  async setLogLevel(level: LogLevel): Promise<void> {
    this.config.level = level;
    try {
      await contextDb.saveSetting(SettingKeys.LOG_LEVEL, level);
    } catch (error) {
      console.error('Failed to save log level setting:', error);
    }
  }

  async getLogLevel(): Promise<LogLevel> {
    if (!this.initialized) {
      await this.initialize();
    }
    return this.config.level;
  }

  private shouldLog(level: LogLevel): boolean {
    return level >= this.config.level;
  }

  private formatMessage(level: string, message: string, prefix?: string): string {
    const logPrefix = prefix || this.config.prefix;
    return `${logPrefix}: [${level}] ${message}`;
  }

  debug(message: string, ...args: any[]): void {
    if (this.shouldLog(LogLevel.DEBUG)) {
      console.debug(this.formatMessage('DEBUG', message), ...args);
    }
  }

  info(message: string, ...args: any[]): void {
    if (this.shouldLog(LogLevel.INFO)) {
      console.log(this.formatMessage('INFO', message), ...args);
    }
  }

  warn(message: string, ...args: any[]): void {
    if (this.shouldLog(LogLevel.WARN)) {
      console.warn(this.formatMessage('WARN', message), ...args);
    }
  }

  error(message: string, ...args: any[]): void {
    if (this.shouldLog(LogLevel.ERROR)) {
      console.error(this.formatMessage('ERROR', message), ...args);
    }
  }

  // Convenience methods that preserve existing emoji patterns
  chat(message: string, ...args: any[]): void {
    if (this.shouldLog(LogLevel.DEBUG)) {
      console.log(`💬 ${message}`, ...args);
    }
  }

  search(message: string, ...args: any[]): void {
    if (this.shouldLog(LogLevel.DEBUG)) {
      console.log(`🔍 ${message}`, ...args);
    }
  }

  model(message: string, ...args: any[]): void {
    if (this.shouldLog(LogLevel.DEBUG)) {
      console.log(`🤖 ${message}`, ...args);
    }
  }

  database(message: string, ...args: any[]): void {
    if (this.shouldLog(LogLevel.DEBUG)) {
      console.log(`💾 ${message}`, ...args);
    }
  }

  api(message: string, ...args: any[]): void {
    if (this.shouldLog(LogLevel.DEBUG)) {
      console.log(`📡 ${message}`, ...args);
    }
  }

  intent(message: string, ...args: any[]): void {
    if (this.shouldLog(LogLevel.DEBUG)) {
      console.log(`🎯 ${message}`, ...args);
    }
  }

  // Method to log without level checking (for critical errors)
  force(message: string, ...args: any[]): void {
    console.error(this.formatMessage('FORCE', message), ...args);
  }
}

// Export singleton instance
export const logger = Logger.getInstance();

// Initialize logger when module is imported
logger.initialize().catch(() => {
  // Silent fallback - logger will use default settings
});

// Export LogLevel for easy access
export { LogLevel as Level };