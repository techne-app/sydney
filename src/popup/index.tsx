import React from 'react';
import { createRoot } from 'react-dom/client';
import { ChatPage } from './components/ChatPage';
import { logger } from '../utils/logger';

logger.info("Popup script starting...");

const App: React.FC = () => {
  return (
    <div className="fixed inset-0 text-base overflow-hidden">
      <ChatPage />
    </div>
  );
};

const container = document.getElementById('root');
logger.debug("Container element:", container);

if (container) {
  const root = createRoot(container);
  logger.debug("Root created, rendering components...");
  root.render(<App />);
}