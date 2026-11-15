// The Banana Engine console: the site's admin panel. A token opens it; inside are the jobs, the footage,
// and for each job a page that follows the engine through its stages.

import { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowUpRight, Banana, Boxes, ChevronRight, Database, Film, KeyRound, LayoutGrid, LogOut, Search, Store, TriangleAlert, Workflow, X } from 'lucide-react';
import site from '../../config/site.json';
import { call, post, savedToken, saveToken, usePoll } from './lib/api.js';
import { deviceName, fmtUSD } from './lib/format.js';
import { busy } from './lib/run.js';
import { JobPage } from './Job.jsx';
import { JobsPage, NewJobSheet } from './Jobs.jsx';
import { BoxesPage, DatasetsPage, FootagePage, OverviewPage, notices } from './Pages.jsx';

const Brand = () => (
  <div className="brand">
    <Banana size={21} strokeWidth={1.6} />
    <span className="word">{site.brand}<i>.</i></span>
    <span className="prod">ENGINE</span>
  </div>
);

function Gate({ onToken, error, checking }) {
  const [value, setValue] = useState('');
  return (
    <div className="gate">
      <form onSubmit={event => { event.preventDefault(); if (value.trim()) onToken(value.trim()); }}>
        <Brand />
        <h1>Banana Engine console</h1>
        <p>Upload raw footage and watch the engine turn it into labeled clips. Enter the admin token to open it; it is set on the server as <code>ADMIN_TOKEN</code> and stays in this browser.</p>
        <input type="password" aria-label="Admin token" placeholder="Admin token" autoFocus autoComplete="current-password" value={value} onChange={event => setValue(event.target.value)} />
        {error && <div className="error" role="alert">{error}</div>}
        <button className="btn primary lg" disabled={!value.trim() || checking}>{checking ? <span className="spin sm ink" /> : <KeyRound size={15} />}Open the console</button>
        <a href="/explore">Back to the data explorer</a>
      </form>
    </div>
  );
}

// The address bar decides which page is shown; links inside the console do not reload it.
function useRoute() {
  const [path, setPath] = useState(window.location.pathname);
  useEffect(() => {
    const onPop = () => setPath(window.location.pathname);
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);
  const go = useCallback(to => { window.history.pushState(null, '', to); setPath(to); }, []);
  return [path.replace(/\/+$/, '') || '/admin', go];
}

function Link({ to, go, className, children }) {
  return <a href={to} className={className} onClick={event => {
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.button) return;
    event.preventDefault();
    go(to);
  }}>{children}</a>;
}

function Processors({ status }) {
  const { workers, engine } = status;
  const lines = [...workers.map(worker => ({ name: `Worker · ${worker.name}`, busy: worker.busy, detail: deviceName(worker.device), fal: worker.fal })),
    ...(engine.ready ? [{ name: 'This server', busy: engine.busy, detail: engine.device === 'cpu' ? 'CPU · slow' : deviceName(engine.device) }] : [])];
  if (!lines.length) return <div className="worker none"><b>No worker connected</b><br />Jobs wait until a machine with the engine asks for work.</div>;
  return lines.map(line => (
    <div key={line.name} className="worker">
      <div className="row"><span className={`dot ${line.busy ? 'busy' : 'ok'}`} /><b>{line.name}</b><span className="r">{line.busy ? 'busy' : 'idle'}</span></div>
      <div className="row" style={{ fontSize: 11 }}><span>{line.detail}</span></div>
      {line.fal && line.fal.key && line.fal.cap_usd > 0 && (
        <div style={{ display: 'grid', gap: 5 }}>
          <div className="row" style={{ fontSize: 11 }}><span>fal budget</span><span className="r num">{fmtUSD(line.fal.spent_usd)} / ${line.fal.cap_usd.toFixed(2)}</span></div>
          <div className="meter"><i style={{ width: `${Math.min(100, (100 * line.fal.spent_usd) / line.fal.cap_usd)}%` }} /></div>
        </div>
      )}
    </div>
  ));
}

export default function Console() {
  const [token, setToken] = useState(savedToken);
  const [gateError, setGateError] = useState('');
  const [status, setStatus] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [footage, setFootage] = useState([]);
  const [error, setError] = useState('');
  const [query, setQuery] = useState('');
  const [sheet, setSheet] = useState(null);         // the new-job sheet: { file } or { name } or {}
  const [path, go] = useRoute();
  const active = jobs.some(busy);
  // The media cookie: asked for once per sign-in, and again when the server says the old one is no longer good
  // (it restarted, or someone signed out elsewhere).
  const session = useRef({ token: '', name: '', opened: null });

  const signOut = useCallback(message => {
    // ends the cookie everywhere and has this browser drop the pictures and videos it cached
    if (token) call(token, '/session', { method: 'DELETE' }).catch(() => {});
    session.current = { token: '', name: '', opened: null };
    saveToken(''); setToken(''); setStatus(null); setGateError(typeof message === 'string' ? message : '');
  }, [token]);

  const refresh = useCallback(async () => {
    if (!token) return;
    try {
      const [nextStatus, nextJobs, nextFootage] = await Promise.all([call(token, '/status'), call(token, '/jobs'), call(token, '/footage')]);
      // The cookie lets this browser load the jobs' pictures and videos. Nothing is shown before it is there.
      if (session.current.token !== token || session.current.name !== nextStatus.session) {
        session.current = { token, name: nextStatus.session, opened: post(token, '/session') };
      }
      await session.current.opened.catch(failure => { session.current = { token: '', name: '', opened: null }; throw failure; });
      setStatus(nextStatus); setJobs(nextJobs.jobs); setFootage(nextFootage.footage); setGateError('');
      saveToken(token);
    } catch (failure) {
      if (failure.status === 401 || failure.status === 503) signOut(failure.message);
      else if (!status) setGateError(failure.message);
      throw failure;
    }
  }, [token, status, signOut]);
  usePoll(refresh, !token ? 3600000 : active ? 2000 : 5000);
  useEffect(() => { if (token && !status) refresh().catch(() => {}); }, [token, status, refresh]);
  useEffect(() => { document.title = 'Banana Engine · peelset'; }, []);
  useEffect(() => { setQuery(''); }, [path]);
  const reload = useCallback(() => { refresh().catch(() => {}); }, [refresh]);
  const closeSheet = useCallback(() => setSheet(null), []);

  if (!status) return <Gate onToken={value => { setGateError(''); setToken(value); }} error={gateError} checking={!!token && !gateError} />;

  const jobId = (path.match(/^\/admin\/jobs\/([^/]+)$/) || [])[1];
  const page = jobId ? 'job' : { '/admin/overview': 'overview', '/admin/footage': 'footage', '/admin/datasets': 'datasets', '/admin/boxes': 'boxes' }[path] || 'jobs';
  const job = jobId && jobs.find(item => item.id === jobId);
  const running = jobs.filter(item => item.status === 'running').length;
  const done = jobs.filter(item => item.status === 'done').length;
  const notes = notices(status);
  const names = { overview: 'Overview', jobs: 'Jobs', footage: 'Footage', datasets: 'Datasets', boxes: 'Boxes' };
  const removeFootage = async name => {
    if (!window.confirm(`Delete ${name}? Jobs already made from it keep their clips.`)) return;
    try { await call(token, `/footage/${encodeURIComponent(name)}`, { method: 'DELETE' }); reload(); } catch (failure) { setError(failure.message); }
  };
  const publish = async id => {
    try { await post(token, `/jobs/${id}/publish`); reload(); } catch (failure) { setError(failure.message); }
  };
  const nav = (to, key, Icon, label, count, live) => (
    <Link to={to} go={go} className={page === key || (key === 'jobs' && page === 'job') ? 'on' : ''}>
      <Icon size={15} />{label}{count != null && <span className={`cnt ${live ? 'hot' : ''}`}>{count}</span>}
    </Link>
  );

  return (
    <div className="shell">
      <aside className="side">
        <Brand />
        <div className="ws">
          <div className="av">R</div>
          <div><div className="n">Relentless</div><div className="s">Data refinery · {site.domain}</div></div>
        </div>
        <div className="nav-h">Banana Engine</div>
        <nav className="nav">
          {nav('/admin/overview', 'overview', LayoutGrid, 'Overview', notes.length ? <TriangleAlert size={11} style={{ color: 'var(--warn)', verticalAlign: -1 }} /> : null)}
          {nav('/admin', 'jobs', Workflow, 'Jobs', running ? `${running} running` : jobs.length, running > 0)}
          {nav('/admin/footage', 'footage', Film, 'Footage', footage.length)}
          {nav('/admin/boxes', 'boxes', Boxes, 'Boxes', 0)}
        </nav>
        <div className="nav-h" style={{ marginTop: 16 }}>Data</div>
        <nav className="nav">
          {nav('/admin/datasets', 'datasets', Database, 'Datasets', done)}
          <a href="/explore" target="_blank" rel="noreferrer"><Store size={15} />Catalog<span className="cnt">{status.library.clips}</span><ArrowUpRight size={12} className="out" style={{ marginLeft: 6 }} /></a>
        </nav>
        <div className="side-foot">
          <Processors status={status} />
          <div className="me">
            <div className="av"><KeyRound size={13} /></div>
            <div><div className="n">Admin</div><div className="s">signed in with the token</div></div>
            <button className="icon-btn" aria-label="Sign out" title="Sign out" onClick={signOut}><LogOut size={15} /></button>
          </div>
        </div>
      </aside>
      <div className="main">
        <header className="topbar">
          <div className="crumbs">
            <span>Banana Engine</span><ChevronRight size={13} />
            {page === 'job' ? <><Link to="/admin" go={go}>Jobs</Link><ChevronRight size={13} /><b className="mono" style={{ fontSize: 12 }}>{job ? job.run : jobId}</b></> : <b>{names[page]}</b>}
          </div>
          {page !== 'job' && page !== 'overview' && page !== 'boxes' && (
            <label className="search"><Search size={14} />
              <input type="search" aria-label="Search" placeholder={page === 'footage' ? 'Search footage…' : page === 'datasets' ? 'Search datasets…' : 'Search jobs…'} value={query} onChange={event => setQuery(event.target.value)} />
            </label>
          )}
        </header>
        <div className="page">
          {page === 'job' && <JobPage key={jobId} token={token} id={jobId} go={go} workers={status.workers} onChanged={reload} onError={setError} />}
          {page === 'jobs' && <JobsPage jobs={jobs} footage={footage} status={status} query={query} go={go} onFile={file => setSheet({ file })} onNew={() => setSheet({})} />}
          {page === 'overview' && <OverviewPage status={status} jobs={jobs} />}
          {page === 'footage' && <FootagePage footage={footage} jobs={jobs} status={status} query={query} onFile={file => setSheet({ file })} onNew={name => setSheet({ name })} onDelete={removeFootage} />}
          {page === 'datasets' && <DatasetsPage jobs={jobs} query={query} go={go} onPublish={publish} />}
          {page === 'boxes' && <BoxesPage />}
        </div>
      </div>
      {sheet && <NewJobSheet token={token} status={status} footage={footage} initial={sheet} onClose={closeSheet} onUploaded={reload} onError={setError}
        onStarted={started => { setSheet(null); reload(); go(`/admin/jobs/${started.id}`); }} />}
      {error && <div className="notice bad banner" role="alert"><TriangleAlert size={14} /><span>{error}</span><button className="x" aria-label="Dismiss" onClick={() => setError('')}><X size={14} /></button></div>}
    </div>
  );
}

