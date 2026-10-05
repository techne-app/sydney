// Tauri entry point — imports CSS then loads the popup React app
import './webview'; // Must be first — patches drag and link handling before React renders
import '../styles/popup.css';
import '../popup/index';
