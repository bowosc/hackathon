import { createRoot } from 'react-dom/client';
import App from './App.jsx';
import PurchasePage from './PurchasePage.jsx';
import './styles.css';

createRoot(document.getElementById('root')).render(/^\/purchases(\/|$)/.test(window.location.pathname) ? <PurchasePage /> : <App />);
