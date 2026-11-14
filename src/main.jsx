import { createRoot } from 'react-dom/client';
import Admin from './Admin.jsx';
import App from './App.jsx';
import './styles.css';

createRoot(document.getElementById('root')).render(window.location.pathname.startsWith('/admin') ? <Admin /> : <App />);
