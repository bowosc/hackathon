// The admin panel's own entry. It is loaded only on /admin (see src/main.jsx), with its own fonts and styles.
import '@fontsource-variable/inter';
import '@fontsource/jetbrains-mono/400.css';
import '@fontsource/jetbrains-mono/500.css';
import '@fontsource/manrope/800.css';
import './console.css';

import { createRoot } from 'react-dom/client';
import Console from './Console.jsx';

document.querySelector('meta[name="theme-color"]')?.setAttribute('content', '#0e0e0e');
createRoot(document.getElementById('root')).render(<Console />);
