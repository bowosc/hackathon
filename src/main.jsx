// Two apps share this page: the public site, and the admin panel under /admin. Each is loaded only where it
// is shown, with its own styles, so neither pays for (or restyles) the other.
// (The two imports sit in a table on purpose: written as an if/else, the build folds them into one call
// and loads the wrong stylesheet.)
const apps = {
  admin: () => import('./admin/main.jsx'),
  site: () => import('./site.jsx'),
};
const admin = /^\/admin(\/|$)/.test(window.location.pathname);
if (admin) document.documentElement.style.background = '#0e0e0e';      // no white flash while the console loads
apps[admin ? 'admin' : 'site']();
