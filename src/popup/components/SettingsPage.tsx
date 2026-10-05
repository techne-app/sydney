import React, { useState, useEffect } from 'react';
import { contextDb, SettingKeys } from '../../utils/contextDb';
import { logger } from '../../utils/logger';

export const SettingsPage: React.FC = () => {
  const [isPersonalizationEnabled, setIsPersonalizationEnabled] = useState<boolean>(false);
  const [isSaving, setIsSaving] = useState<boolean>(false);

  // Load current settings when component mounts
  useEffect(() => {
    const loadSettings = async () => {
      try {
        const personalizationEnabled = await contextDb.getSettingValue(SettingKeys.PERSONALIZATION_ENABLED, false);
        setIsPersonalizationEnabled(personalizationEnabled);
      } catch (error) {
        logger.error('Failed to load settings:', error);
      }
    };

    loadSettings();
  }, []);

  // Handle personalization toggle change
  const handlePersonalizationToggle = async () => {
    const newValue = !isPersonalizationEnabled;
    setIsSaving(true);
    
    try {
      setIsPersonalizationEnabled(newValue); // Update UI immediately
      await contextDb.saveSetting(SettingKeys.PERSONALIZATION_ENABLED, newValue);
      logger.info(`Personalization setting updated to: ${newValue}`);
    } catch (error) {
      logger.error('Failed to save setting:', error);
      // Revert UI state on error
      setIsPersonalizationEnabled(!newValue);
    } finally {
      setIsSaving(false);
    }
  };


  return (
    <div 
      className="h-full flex flex-col"
      style={{ 
        fontFamily: 'var(--font-sans)',
        backgroundColor: 'var(--dark-bg)',
        color: 'var(--text-primary)'
      }}
    >
      <div className="flex-1 overflow-auto p-6">
        <div className="space-y-6">
          {/* Personalization Setting */}
          <div>
            <div className="flex items-center">
              <label 
                htmlFor="personalization" 
                className="text-sm mr-3"
                style={{ 
                  color: 'var(--text-primary)',
                  fontWeight: 'var(--font-weight-medium)'
                }}
              >
                Enable Personalization
              </label>
              
              {/* Toggle switch */}
              <button 
                onClick={handlePersonalizationToggle}
                disabled={isSaving}
                className="relative inline-flex items-center h-6 rounded-full w-11 focus:outline-none transition-all duration-200"
                aria-pressed={isPersonalizationEnabled}
                style={{
                  backgroundColor: isPersonalizationEnabled ? 'var(--hn-blue)' : 'var(--hn-border)'
                }}
              >
                <span 
                  className={`${
                    isPersonalizationEnabled ? 'translate-x-6' : 'translate-x-1'
                  } inline-block h-4 w-4 rounded-full bg-white transform transition-transform duration-200 ease-in-out shadow-sm`}
                ></span>
              </button>
            </div>
            
          </div>

        </div>
      </div>
    </div>
  );
};