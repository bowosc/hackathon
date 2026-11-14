import { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowLeft, ArrowUpRight, Banana, Check, Film, KeyRound, Play, Square, Trash2, TriangleAlert, Upload, X } from 'lucide-react';
import site from '../config/site.json';
import './admin.css';

const tokenKey = 'peelset:admin-token';
const videoTypes = ['.mp4', '.mov', '.avi', '.mkv', '.m4v', '.webm'];
const size = bytes => bytes >= 2 ** 30 ? `${(bytes / 2 ** 30).toFixed(1).replace('.0', '')} GB` : `${Math.max(1, Math.round(bytes / 2 ** 20))} MB`;
const clock = seconds => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
const elapsed = (from, to) => from ? clock((new Date(to || Date.now()) - new Date(from)) / 1000) : '';
const busy = job => ['queued', 'running', 'stopping'].includes(job.status);

async function call(token, path, options = {}) {
  const response = await fetch(`/api/admin${path}`, { ...options, headers: { Authorization: `Bearer ${token}`, ...options.headers } });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw Object.assign(new Error(typeof data.detail === 'string' ? data.detail : `Request failed: ${response.status}`), { status: response.status });
  return data;
}

function Gate({ onToken, error }) {
  const [value, setValue] = useState('');
  return <form className="admin-gate" onSubmit={event => { event.preventDefault(); if (value.trim()) onToken(value.trim()); }}>
    <KeyRound size={22} strokeWidth={1.5} />
    <h2>Admin token</h2>
    <p>The token is set on the server as <code>ADMIN_TOKEN</code>. It stays in this browser.</p>
    <input type="password" aria-label="Admin token" autoFocus autoComplete="current-password" value={value} onChange={event => setValue(event.target.value)} />
    {error && <p className="admin-error" role="alert">{error}</p>}
    <button className="primary-button" disabled={!value.trim()}>Open admin</button>
  </form>;
}

function Status({ status }) {
  const { engine, storage, machine, library, limits } = status;
  const free = machine.memory_available_bytes;
  const notes = [];
  const { workers } = status;
  if (!engine.ready) notes.push(workers.length ? 'The engine is not installed on this machine. Jobs run on a connected worker.' : 'The engine is not installed on this machine. Connect a worker to process footage: on a machine with the engine, run "uv run python -m relentless.worker" with this site\'s address.');
  else if (!engine.claude_key) notes.push('ANTHROPIC_API_KEY is not set, so the engine cannot label clips. Set it on the server.');
  else if (free != null && free < limits.job_memory_bytes) notes.push(`Only ${size(free)} of memory is free on this machine right now. A job needs about ${size(limits.job_memory_bytes)} and will not start until there is room.`);
  if (storage.kind === 'container') notes.push('Uploads and processed clips are stored inside the container and are lost on the next deploy. Mount a volume at /data to keep them.');
  return <>
    <dl className="admin-tiles">
      <div><dt>Engine</dt><dd>{workers.length ? `${workers.length} worker${workers.length === 1 ? '' : 's'}` : engine.ready ? 'Ready' : 'Not installed'}</dd><span>{workers.length ? `${workers.map(worker => `${worker.name} (${worker.device === 'cpu' ? 'CPU' : 'GPU'})`).join(', ')} connected` : `${machine.cpus} CPUs${engine.device === 'cpu' ? ', no GPU' : ''}${free != null ? ` · ${size(free)} memory free` : ''}`}</span></div>
      <div><dt>Free space</dt><dd>{size(storage.free_bytes)}</dd><span>{storage.kind === 'volume' ? 'On a mounted volume' : storage.kind === 'container' ? 'Inside the container' : 'On this disk'}</span></div>
      <div><dt>Library</dt><dd>{library.clips} clips</dd><span>{library.processed_here} processed here</span></div>
      <div><dt>Labeling spend</dt><dd>${engine.claude_usd.toFixed(2)}</dd><span>Claude, on this machine</span></div>
    </dl>
    {notes.map(note => <p className="admin-note" key={note}><TriangleAlert size={14} />{note}</p>)}
  </>;
}

function Uploader({ token, limits, onDone }) {
  const [uploads, setUploads] = useState([]);
  const [over, setOver] = useState(false);
  const input = useRef(null);
  const update = (id, change) => setUploads(list => list.map(item => item.id === id ? { ...item, ...change } : item));

  async function send(file) {
    const id = `${file.name}-${Date.now()}`;
    const fail = error => update(id, { error });
    setUploads(list => [...list, { id, name: file.name, sent: 0, total: file.size }]);
    if (!videoTypes.some(type => file.name.toLowerCase().endsWith(type))) return fail(`Not a video file (${videoTypes.join(' ')}).`);
    if (file.size > limits.upload_bytes) return fail(`Larger than the ${size(limits.upload_bytes)} limit.`);
    for (let offset = 0; offset < file.size;) {
      const piece = file.slice(offset, offset + limits.chunk_bytes);
      const query = new URLSearchParams({ name: file.name, total: file.size, offset });
      let reply, attempt = 0;
      while (!reply) {
        try { reply = await call(token, `/footage?${query}`, { method: 'PUT', body: piece }); }
        catch (error) { if (error.status || ++attempt > 3) return fail(error.message); await new Promise(resolve => setTimeout(resolve, 1500 * attempt)); }
      }
      offset += piece.size;
      update(id, { sent: offset });
      if (reply.done) { update(id, { done: true }); onDone(); }
    }
  }
  const add = files => [...files].forEach(send);

  return <section className="admin-card">
    <h2>Upload raw footage</h2>
    <button type="button" className={`drop-zone ${over ? 'is-over' : ''}`} onClick={() => input.current.click()}
      onDragOver={event => { event.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
      onDrop={event => { event.preventDefault(); setOver(false); add(event.dataTransfer.files); }}>
      <Upload size={22} strokeWidth={1.5} /><strong>Drop a video here, or choose a file</strong>
      <span>{videoTypes.join(' ')} · up to {size(limits.upload_bytes)}</span>
    </button>
    <input ref={input} type="file" hidden multiple accept={videoTypes.join(',')} onChange={event => { add(event.target.files); event.target.value = ''; }} />
    {uploads.map(item => <div className="upload-row" key={item.id}>
      <span className="upload-name">{item.name}</span>
      {item.error ? <span className="admin-error">{item.error}</span> : <progress max={item.total} value={item.sent} aria-label={`Uploading ${item.name}`} />}
      <span className="upload-state">{item.error ? '' : item.done ? 'Uploaded' : `${Math.round(100 * item.sent / item.total)}%`}</span>
      {(item.done || item.error) && <button className="icon-button" aria-label="Dismiss" onClick={() => setUploads(list => list.filter(other => other.id !== item.id))}><X size={14} /></button>}
    </div>)}
  </section>;
}

function Footage({ item, token, here, workers, slow, last, onChange, onError }) {
  const [open, setOpen] = useState(false);
  const [place, setPlace] = useState('');   // '' = whatever is best right now
  const where = place || (workers.length || !here ? 'worker' : 'server');
  const small = slow && where === 'server';
  // Without a GPU the engine takes minutes per clip, so the suggested job is smaller there.
  const [options, setOptions] = useState({ start: 0, seconds: Math.min(small ? 30 : 60, Math.floor(item.seconds || 60)), max_clips: small ? 3 : 4, quality: 'fast' });
  const [working, setWorking] = useState(false);
  const set = (key, value) => setOptions(previous => ({ ...previous, [key]: value }));
  async function run(event) {
    event.preventDefault(); setWorking(true);
    try {
      await call(token, '/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ footage: item.name, ...options, where }) });
      setOpen(false); onChange();
    } catch (error) { onError(error.message); } finally { setWorking(false); }
  }
  async function remove() {
    if (!window.confirm(`Delete ${item.name}?`)) return;
    try { await call(token, `/footage/${encodeURIComponent(item.name)}`, { method: 'DELETE' }); onChange(); } catch (error) { onError(error.message); }
  }
  return <li className="footage-row">
    <div className="footage-line">
      <Film size={16} strokeWidth={1.6} />
      <div className="footage-name"><strong>{item.name}</strong><span>{[item.seconds && clock(item.seconds), item.width && `${item.width}×${item.height}`, item.fps && `${item.fps} fps`, size(item.bytes)].filter(Boolean).join(' · ')}</span></div>
      <button className="secondary-button" onClick={() => setOpen(value => !value)} aria-expanded={open}><Play size={13} />Process</button>
      <button className="icon-button" aria-label={`Delete ${item.name}`} onClick={remove}><Trash2 size={15} /></button>
    </div>
    {open && <form className="run-options" onSubmit={run}>
      <label>Start (s)<input type="number" min="0" max={item.seconds || undefined} step="1" value={options.start} onChange={event => set('start', Number(event.target.value))} /></label>
      <label>Length (s)<input type="number" min="5" max="3600" step="1" value={options.seconds} onChange={event => set('seconds', Number(event.target.value))} /></label>
      <label>Clips to refine<input type="number" min="1" max="24" value={options.max_clips} onChange={event => set('max_clips', Number(event.target.value))} /></label>
      <label>Upscale to<select value={options.quality} onChange={event => set('quality', event.target.value)}><option value="fast">720p</option><option value="full">1080p (slower)</option></select></label>
      <label>Run on<select value={where} onChange={event => setPlace(event.target.value)}><option value="worker">{workers.length ? `${workers.map(worker => worker.name).join(', ')} (worker)` : 'A worker (none yet)'}</option><option value="server" disabled={!here}>{here ? `This server${slow ? ' (slow)' : ''}` : 'This server (no engine)'}</option></select></label>
      <button className="primary-button" disabled={working}>{working ? 'Starting…' : 'Run the engine'}</button>
      <p>The engine finds the moments where hands work on objects, cuts and upscales them, and labels each clip. Accepted clips join the catalog.{where === 'worker' && !workers.length && ' The job waits until a worker connects.'}{small && ' This server has no GPU, so a job here takes many times the length of the footage.'}{last && ` The last job took ${elapsed(last.started_at, last.finished_at)} for ${Math.round(last.source.seconds)} s of footage and ${last.clips.planned} clips${last.worker ? ` on ${last.worker}` : ' on this server'}.`}</p>
    </form>}
  </li>;
}

function Job({ job, token, onChange, onError }) {
  const current = job.stages.find(stage => stage.status === 'running') || [...job.stages].reverse().find(stage => stage.status === 'done');
  const act = async (path, method) => { try { await call(token, path, { method }); onChange(); } catch (error) { onError(error.message); } };
  const span = job.source.seconds ? `${clock(job.source.start_s)} to ${clock(job.source.start_s + job.source.seconds)}` : `from ${clock(job.options.start)}`;
  const remove = () => { if (window.confirm('Remove this job and the clips it added to the catalog?')) act(`/jobs/${job.id}`, 'DELETE'); };
  return <li className={`job is-${job.status}`}>
    <div className="job-head">
      <div><strong>{job.footage}</strong><span>{span} · up to {job.options.max_clips} clips · {job.options.quality === 'fast' ? '720p' : '1080p'} · {job.worker ? `on ${job.worker}` : job.options.where === 'worker' ? 'waiting for a worker' : 'on this server'}</span></div>
      <span className="job-status">{job.status === 'done' && <Check size={12} />}{job.status}{job.started_at && ` · ${elapsed(job.started_at, job.finished_at)}`}</span>
      {busy(job) ? <button className="secondary-button" disabled={job.status === 'stopping'} onClick={() => act(`/jobs/${job.id}/stop`, 'POST')}><Square size={11} />Stop</button>
        : <button className="icon-button" aria-label="Remove this job and its clips" title="Remove this job and its clips" onClick={remove}><Trash2 size={15} /></button>}
    </div>
    {!!job.stages.length && <ol className="job-stages" aria-label="Engine stages">{job.stages.map(stage => <li key={stage.key} className={`is-${stage.status}`} title={`${stage.title}${stage.detail ? `: ${stage.detail}` : ''}`}><span style={{ width: `${Math.round(100 * (stage.status === 'done' ? 1 : stage.progress || 0))}%` }} /></li>)}</ol>}
    {current && <p className="job-detail"><strong>{current.title}</strong>{current.detail ? ` · ${current.detail}` : job.status === 'running' ? ' · working' : ''}</p>}
    {job.status === 'done' && <p className="job-result">{job.clips.accepted ? <>{job.clips.accepted} clip{job.clips.accepted === 1 ? '' : 's'} added to the catalog{job.clips.rejected ? `, ${job.clips.rejected} rejected by the labeler` : ''}{job.claude_usd != null ? ` · $${job.claude_usd.toFixed(3)} of labeling` : ''}. <a href="/explore?sort=newest">See them in the explorer<ArrowUpRight size={12} /></a></> : 'No clip passed the labeler, so nothing was added to the catalog.'}</p>}
    {job.error && <p className="admin-error">{job.error}</p>}
    {job.log && job.status !== 'stopped' && <pre className="job-log">{job.log}</pre>}
  </li>;
}

export default function Admin() {
  const [token, setToken] = useState(() => localStorage.getItem(tokenKey) || '');
  const [gateError, setGateError] = useState('');
  const [status, setStatus] = useState(null);
  const [footage, setFootage] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [error, setError] = useState('');
  const active = jobs.some(busy);

  const refresh = useCallback(async () => {
    if (!token) return;
    try {
      const [nextStatus, nextFootage, nextJobs] = await Promise.all([call(token, '/status'), call(token, '/footage'), call(token, '/jobs')]);
      setStatus(nextStatus); setFootage(nextFootage.footage); setJobs(nextJobs.jobs); setGateError('');
      localStorage.setItem(tokenKey, token);
    } catch (failure) {
      if (failure.status === 401 || failure.status === 503) { localStorage.removeItem(tokenKey); setToken(''); setStatus(null); setGateError(failure.message); }
      else setError(failure.message);
    }
  }, [token]);

  useEffect(() => { document.title = 'Peelset — Admin'; }, []);
  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => {
    if (!token) return;
    const timer = setInterval(refresh, active ? 2000 : 10000);
    return () => clearInterval(timer);
  }, [token, active, refresh]);

  return <div className="admin">
    <header className="admin-header">
      <a className="brand" href="/"><Banana className="brand-mark" strokeWidth={1.6} aria-hidden="true" />{site.brand}<span className="brand-period">.</span></a>
      <nav><a href="/explore"><ArrowLeft size={13} />Data explorer</a>{status && <button onClick={() => { localStorage.removeItem(tokenKey); setToken(''); setStatus(null); }}>Sign out</button>}</nav>
    </header>
    <main>
      <p className="eyebrow"><span className="eyebrow-line" />ADMIN</p>
      <h1>Raw footage in. Labeled clips out.</h1>
      {!status ? <Gate onToken={setToken} error={gateError} /> : <>
        <Status status={status} />
        {error && <p className="admin-error admin-banner" role="alert">{error}<button className="icon-button" aria-label="Dismiss" onClick={() => setError('')}><X size={14} /></button></p>}
        <Uploader token={token} limits={status.limits} onDone={refresh} />
        <section className="admin-card">
          <h2>Footage<span>{footage.length}</span></h2>
          {footage.length ? <ul className="footage-list">{footage.map(item => <Footage key={item.name} item={item} token={token} here={status.engine.ready && status.engine.claude_key} workers={status.workers} slow={status.engine.device === 'cpu'} last={jobs.find(job => job.status === 'done' && job.source.seconds)} onChange={refresh} onError={setError} />)}</ul>
            : <p className="admin-empty">Nothing uploaded yet.</p>}
        </section>
        <section className="admin-card">
          <h2>Engine jobs<span>{jobs.length}</span></h2>
          {jobs.length ? <ul className="job-list">{jobs.map(job => <Job key={job.id} job={job} token={token} onChange={refresh} onError={setError} />)}</ul>
            : <p className="admin-empty">No jobs yet. Upload footage, then press Process.</p>}
        </section>
      </>}
    </main>
  </div>;
}
