// Tauri entry point — imports CSS then loads the popup React app
import './chrome-shim'; // Must be first — sets up window.chrome before React renders
import '../styles/popup.css';
import '../popup/index';
