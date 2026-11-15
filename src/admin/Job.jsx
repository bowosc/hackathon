// One job: its header, the row of the engine's eight stages, and under it either the live view of a
// stage or, once the job is done, the dataset it made.

import { Component, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Check, Download, FileText, MoreHorizontal, Square, Trash2, Undo2, Upload, X } from 'lucide-react';
import { call, post, usePoll } from './lib/api.js';
import { fmtClock, fmtUSD, plural, withDevice } from './lib/format.js';
import { makeClock, useLatest } from './lib/media.jsx';
import { LABELED, STAGE, STAGES, UPSCALED, busy, followed, passedVariants, readRun } from './lib/run.js';
import { ClipDrawer, Results, Toast } from './Results.jsx';
import { LiveView, SideColumn, stageLine } from './Stages.jsx';

// The clip a clip stage shows: the one picked by hand, else the one the engine most recently finished
// while this page was open, else the one it is on, else the first there is.
function heroClip(run, focus, picked, latest) {
  const { clips } = run;
  const get = id => clips.find(clip => clip.id === id);
  if (picked && get(picked)) return get(picked);
  if (focus === 'upscale') return get(latest.upscale) || clips.find(clip => clip.status === 'upscaling') || clips.find(clip => clip.files.upscaled) || clips[0] || null;
  if (focus === 'label') return get(latest.label) || clips.find(clip => clip.status === 'labeling') || clips.find(clip => clip.label) || clips.find(clip => clip.files.upscaled) || null;
  if (focus === 'multiply') return get(latest.multiply) || clips.find(clip => clip.multiply === 'running') || clips.find(clip => (clip.variants || []).length) || clips.find(clip => clip.multiply) || null;
  return null;
}

// The engine's own clock, ticking between two of its reports.
function useElapsed(run) {
  const base = useRef({ elapsed: 0, at: 0 });
  const [, redraw] = useState(0);
  const reported = run.state.elapsed || 0;
  if (reported !== base.current.elapsed) base.current = { elapsed: reported, at: performance.now() };
  useEffect(() => {
    if (!run.active) return undefined;
    const timer = setInterval(() => redraw(n => n + 1), 500);
    return () => clearInterval(timer);
  }, [run.active]);
  return run.active && reported ? reported + (performance.now() - base.current.at) / 1000 : reported;
}

const title = run => {
  const { source, job } = run;
  if (source.camera) return `Camera ${source.camera}${source.recorded ? ` · ${source.recorded.slice(0, 16)}` : ''}`;
  return job.footage.replace(/\.[^.]+$/, '').replace(/[-_]+/g, ' ');
};

function Steps({ run, focus, canOpen, onOpen }) {
  return (
    <div className="steps">
      {STAGES.map(stage => {
        const st = run.by[stage.key] || { status: 'pending' };
        const Icon = stage.icon;
        const reached = canOpen && ['running', 'done', 'error', 'stopped'].includes(st.status);
        return (
          <button key={stage.key} className={`step ${st.status || 'pending'} ${focus === stage.key ? 'focus' : ''}`} disabled={!reached} onClick={() => onOpen(stage.key)}
            title={reached ? `Show the ${stage.title.toLowerCase()} stage` : undefined}>
            <div className="top">
              <Icon size={13} className="ic" />
              <span className="name"><span className="long">{stage.title}</span><span className="short">{stage.short}</span></span>
              <span className="st">
                {st.status === 'done' && <span className="okc"><Check size={10} strokeWidth={3} /></span>}
                {st.status === 'running' && <span className="spin sm" />}
                {st.status === 'error' && <span className="badc"><X size={10} strokeWidth={3} /></span>}
                {st.status === 'stopped' && <Square size={9} fill="currentColor" style={{ color: 'var(--muted)' }} />}
                {(!st.status || st.status === 'pending') && <span className="pend" />}
              </span>
            </div>
            <div className="m">{stageLine(run, stage.key)}</div>
            {st.status === 'running' && <i className="pbar" style={{ width: `${Math.min(1, st.progress || 0) * 100}%` }} />}
          </button>
        );
      })}
    </div>
  );
}

function JobView({ detail, token, go, site, workers, reload, onChanged, onError }) {
  const run = useMemo(() => readRun(detail), [detail]);
  const { job } = run;
  const [pinned, setPinned] = useState(null);      // a stage opened by hand; null follows the engine, or shows the dataset of a finished job
  const [picked, setPicked] = useState(null);      // a clip opened by hand in a clip stage
  const [open, setOpen] = useState(null);          // the clip in the drawer
  const [toast, setToast] = useState(false);
  const [menu, setMenu] = useState(false);
  const [doing, setDoing] = useState('');
  const clock = useMemo(makeClock, []);
  const latest = {
    upscale: useLatest(run.clips, clip => UPSCALED.includes(clip.status) && !!clip.files.upscaled),
    label: useLatest(run.clips, clip => LABELED.includes(clip.status)),
    multiply: useLatest(run.clips, clip => (clip.variants || []).length > 0),
  };
  const elapsed = useElapsed(run);
  const done = job.status === 'done';
  const reported = run.reported;
  const live = followed(run.stages, job.status);
  const focus = reported ? pinned || (done ? null : live) : null;
  const hero = heroClip(run, focus, picked, latest);
  const accepted = run.clips.filter(clip => clip.status === 'accepted');
  const variants = accepted.reduce((sum, clip) => sum + passedVariants(clip).length, 0);
  const shown = open && run.clips.find(clip => clip.id === open);
  useEffect(() => { setPicked(null); }, [focus]);
  useEffect(() => {
    if (!menu) return undefined;
    const close = () => setMenu(false);
    window.addEventListener('click', close);
    return () => window.removeEventListener('click', close);
  }, [menu]);
  const hideToast = useCallback(() => setToast(false), []);
  const closeDrawer = useCallback(() => setOpen(null), []);

  const act = async (name, work) => {
    setDoing(name);
    try { await work(); await reload(); onChanged(); }
    catch (error) { onError(error.message); }
    finally { setDoing(''); }
  };
  const openStage = key => setPinned(key === pinned || (!done && key === live) ? null : key);
  const remove = async () => {
    const made = run.clips.length ? ` and the ${plural(run.clips.length, 'clip')} it made` : '';
    if (!window.confirm(`Delete this job${made}?${job.published ? ' They are in the public catalog now and will leave it.' : ''}`)) return;
    try { await call(token, `/jobs/${job.id}`, { method: 'DELETE' }); onChanged(); go('/admin'); }
    catch (error) { onError(error.message); }
  };
  const from = run.source.start_s != null ? run.source.start_s : job.options.start || 0;
  const window_ = run.total ? `${fmtClock(from)}–${fmtClock(from + run.total)}` : `from ${fmtClock(from)}`;
  const where = job.worker || job.options.where !== 'worker' ? withDevice(job.worker || 'this server', run.device) : 'a worker';

  return (
    <>
      <div className="job">
        <div style={{ display: 'grid', gap: 10 }}>
          <div className="job-h">
            <div className="thumb" style={{ backgroundImage: `url("/api/admin/footage/${encodeURIComponent(job.footage)}/film")` }} />
            <div className="ttl">
              <h1><span>{title(run)}</span>
                {done ? <span className="status done"><Check size={13} strokeWidth={2.6} />Completed</span>
                  : job.status === 'running' ? <span className="status run"><span className="spin sm" />{reported && STAGE[live] ? STAGE[live].title : 'Starting'}</span>
                    : job.status === 'stopping' ? <span className="status off"><span className="spin sm w" />Stopping</span>
                      : job.status === 'queued' ? <span className="status off">{job.options.where === 'worker' ? 'Waiting for a worker' : 'Queued'}</span>
                        : job.status === 'failed' ? <span className="status bad"><X size={13} strokeWidth={2.6} />Failed</span>
                          : <span className="status off">Stopped</span>}
              </h1>
              <div className="meta"><span className="mono" style={{ fontSize: 11 }}>{job.run}</span><i />{job.footage}<i />{window_}
                {run.source.size && <><i />{run.size[0]}×{run.size[1]}{run.source.fps ? ` · ${Math.round(run.source.fps)} fps` : ''}</>}<i />{where}</div>
            </div>
            <div className="r">
              <div className="kpi"><span className="k">Elapsed</span><span className="v">{fmtClock(elapsed)}</span></div>
              <div className="vsep" />
              <div className="kpi wide-only"><span className="k">Spend</span><span className="v">{fmtUSD((run.costs.anthropic_usd || 0) + (run.costs.fal_usd || 0))}</span></div>
              <div className="vsep wide-only" />
              <div className="kpi"><span className="k">Clips accepted</span><span className="v">{accepted.length}</span></div>
              <div className="vsep" />
              {busy(job) && <button className="btn" disabled={job.status === 'stopping' || doing === 'stop'} onClick={() => act('stop', () => post(token, `/jobs/${job.id}/stop`))}><Square size={11} fill="currentColor" />Stop</button>}
              {done && <a className="btn" href={`/api/admin/jobs/${job.id}/file/catalog.json`} download={`${job.run}-catalog.json`} title="Every clip's labels and where it comes from, as JSON"><Download size={14} />Export</a>}
              {done && (job.published
                ? <span className="btn ok" role="status"><Check size={14} strokeWidth={2.6} />Published</span>
                : <button className="btn primary" disabled={!accepted.length || doing === 'publish'} title={accepted.length ? undefined : 'No clip was accepted, so there is nothing to publish'}
                  onClick={() => act('publish', async () => { await post(token, `/jobs/${job.id}/publish`); setToast(true); })}>
                  {doing === 'publish' ? <span className="spin sm ink" /> : <Upload size={14} />}Publish to catalog</button>)}
              <div className="menu-wrap">
                <button className="icon-btn" aria-label="More actions" aria-expanded={menu} onClick={event => { event.stopPropagation(); setMenu(value => !value); }}><MoreHorizontal size={16} /></button>
                {menu && (
                  <div className="menu" role="menu">
                    {run.url('run.log') && <a role="menuitem" href={run.url('run.log')} target="_blank" rel="noreferrer"><FileText size={14} />Open the engine's log</a>}
                    {job.published && <button role="menuitem" onClick={() => act('unpublish', () => post(token, `/jobs/${job.id}/unpublish`))}><Undo2 size={14} />Take out of the catalog</button>}
                    {!busy(job) && <button role="menuitem" className="danger" onClick={remove}><Trash2 size={14} />Delete job and its clips</button>}
                    {busy(job) && <button role="menuitem" disabled style={{ opacity: .5 }}><Trash2 size={14} />Stop the job to delete it</button>}
                  </div>
                )}
              </div>
            </div>
          </div>
          {job.error && <div className="notice bad"><X size={14} /><span>{job.error}</span></div>}
        </div>
        <Steps run={run} focus={focus} canOpen={reported} onOpen={openStage} />
        <div style={{ minHeight: 0 }}>
          {!reported ? <Idle job={job} site={site} workers={workers} />
            : focus ? (
              <div className="body">
                <LiveView run={run} focus={focus} following={!pinned} onFollow={() => setPinned(null)} onResults={done && pinned ? () => setPinned(null) : null}
                  hero={hero} picked={picked} onPick={setPicked} clock={clock} />
                <SideColumn run={run} focus={focus} hero={hero} fresh={!!hero && !picked && hero.id === latest.label} clock={clock} />
              </div>
            ) : <Results run={run} onOpen={setOpen} />}
        </div>
      </div>
      {shown && <ClipDrawer run={run} clip={shown} onClose={closeDrawer} />}
      {toast && <Toast clips={accepted.length} variants={variants} onClose={hideToast} />}
    </>
  );
}

// A job the engine has not reported on yet: waiting for its turn, for a worker, or over before it began.
function Idle({ job, site, workers }) {
  const ready = workers.filter(worker => !worker.busy).map(worker => worker.name);
  const waits = job.status === 'queued' && job.options.where === 'worker';
  const text = waits && ready.length ? ['Waiting for a worker', `${ready[0]} is connected and will take the job in a moment.`]
    : waits ? ['Waiting for a worker', workers.length ? 'Every connected worker is busy. The job starts when one is free.' : 'The job starts as soon as a machine with the engine asks for work. To make a Mac a worker, run this in its engine folder:']
    : job.status === 'queued' ? ['Waiting for its turn', 'The engine on this server runs one job at a time. This one starts when the job before it ends.']
      : job.status === 'running' ? [job.worker ? `${job.worker} has the job` : 'Starting the engine', job.worker ? 'It is fetching the footage. The first stage shows up here in a moment.' : 'Loading the engine. The first stage shows up here in a moment.']
        : job.status === 'failed' ? ['The job did not get going', 'Nothing was processed. The message above says why.']
          : ['Stopped before it started', 'Nothing was processed.'];
  return (
    <div className="card" style={{ height: '100%', display: 'grid', placeItems: 'center' }}>
      <div className="empty">
        <div className="ico">{job.status === 'running' || job.status === 'queued' ? <span className="spin" /> : <X size={18} />}</div>
        <b>{text[0]}</b>
        <p>{text[1]}</p>
        {waits && !workers.length && <pre className="cmd">uv run python -m relentless.worker {site}</pre>}
        {job.log && <pre className="cmd" style={{ textAlign: 'left', maxWidth: 720 }}>{job.log}</pre>}
      </div>
    </div>
  );
}

// If a run's state is not what the engine writes (a worker gone wrong), the page says so instead of going blank.
class Guard extends Component {
  state = { failed: false };

  static getDerivedStateFromError() { return { failed: true }; }

  render() { return this.state.failed ? this.props.fallback : this.props.children; }
}

export function JobPage({ token, id, go, workers, onChanged, onError }) {
  const [detail, setDetail] = useState(null);
  const [gone, setGone] = useState(false);
  const last = useRef(null);       // the frame of the running stage this page already holds
  const load = useCallback(async () => {
    const have = last.current ? `?have=${encodeURIComponent(`${last.current.stage}:${last.current.seq}`)}` : '';
    try {
      const next = await call(token, `/jobs/${id}${have}`);
      if (next.now && next.now.same) next.now = last.current; else last.current = next.now;
      setDetail(next);
    } catch (error) {
      if (error.status === 404) setGone(true);
      throw error;
    }
  }, [token, id]);
  usePoll(load, !detail || busy(detail.job) ? 900 : 6000);
  if (gone) {
    return <div className="jobs"><div className="card"><div className="empty"><div className="ico"><X size={18} /></div><b>No such job</b>
      <p>It may have been deleted, or the site was redeployed without a volume for its data.</p>
      <button className="btn" onClick={() => go('/admin')}>Back to jobs</button></div></div></div>;
  }
  if (!detail) return <div className="center"><div className="waiting"><span className="spin" />Loading the job…</div></div>;
  const broken = (
    <div className="jobs"><div className="card"><div className="empty"><div className="ico"><X size={18} /></div><b>This job's run cannot be shown</b>
      <p>What the run reported is not in the shape the engine writes. The job itself is untouched: it can still be deleted from the list.</p>
      <button className="btn" onClick={() => go('/admin')}>Back to jobs</button></div></div></div>
  );
  return (
    <Guard fallback={broken}>
      <JobView detail={detail} token={token} go={go} site={window.location.origin} workers={workers} reload={load} onChanged={onChanged} onError={onError} />
    </Guard>
  );
}
