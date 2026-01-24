import React from 'react';
import ReactDOM from 'react-dom/client';
import { AppProvider } from './context/AppContext';
import App from './App';

// Import styles
import './styles/base.css';
import './styles/sidebar.css';
import './styles/controls.css';
import './styles/optimizers.css';
import './styles/panels.css';
import './styles/tooltips.css';
import './styles/theme-toggle.css';

// Apply theme immediately to prevent flash
const savedTheme = localStorage.getItem('theme') || 'dark';
document.body.setAttribute('data-theme', savedTheme);

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <AppProvider>
      <App />
    </AppProvider>
  </React.StrictMode>
);
