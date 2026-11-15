import site from '../config/site.json';

export function savedPurchases() {
  try {
    const value = JSON.parse(localStorage.getItem(site.purchases_storage_key) || '[]');
    return Array.isArray(value) ? value.filter(item => typeof item?.id === 'string' && typeof item?.access_token === 'string') : [];
  } catch { return []; }
}

export function rememberPurchase(purchase) {
  try {
    localStorage.setItem(site.purchases_storage_key, JSON.stringify([
      purchase, ...savedPurchases().filter(item => item.id !== purchase.id),
    ].slice(0, 100)));
  } catch { /* The private receipt URL also restores access without browser storage. */ }
}

export const money = (cents, currency = 'usd') => new Intl.NumberFormat('en-US', { style: 'currency', currency }).format(cents / 100);
export const fileSize = bytes => bytes < 1024 * 1024 ? `${Math.ceil(bytes / 1024)} KB` : bytes < 1024 ** 3 ? `${(bytes / 1024 ** 2).toFixed(1)} MB` : `${(bytes / 1024 ** 3).toFixed(1)} GB`;
