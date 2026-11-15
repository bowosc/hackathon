// The Jobs page (drop footage, see what the engine has made so far, open a job) and the sheet that
// starts a new job: upload, a look at the file, the pipeline's few choices, an estimate, Start.

import { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowUpRight, Check, ChevronDown, Clock, Coins, Cpu, Crop, FileVideo, Film, ImageUpscale, Layers, Lock, Plus, Scissors, ShieldCheck, Sparkles, Tags, Upload, X } from 'lucide-react';
import { post, upload, videoTypes } from './lib/api.js';
import { fmtBytes, fmtClock, fmtUSD, fmtWhen, jobSeconds, labelerName, plural, withDevice } from './lib/format.js';
import { STAGE, STAGES, followed } from './lib/run.js';

export const matches = (query, ...texts) => !query.trim() || texts.some(text => (text || '').toLowerCase().includes(query.trim().toLowerCase()));

export function DropZone({ limits, onFile, slim, children }) {
  const [over, setOver] = useState(false);
  const input = useRef(null);
  return (
    <div className={`drop ${over ? 'over' : ''} ${slim ? 'slim' : ''}`}
      onDragOver={event => { event.preventDefault(); setOver(true); }}
      onDragLeave={event => { if (!event.currentTarget.contains(event.relatedTarget)) setOver(false); }}
      onDrop={event => { event.preventDefault(); setOver(false); if (event.dataTransfer.files.length) onFile(event.dataTransfer.files[0]); }}>
      <button className="zone" onClick={() => input.current.click()}>
        <div className="ico"><Upload size={slim ? 16 : 20} /></div>
        <div className="t1">{over ? 'Release to upload' : 'Drop raw footage here'}</div>
        <div className="t2">MP4, AVI, MOV or MKV up to {fmtBytes(limits.upload_bytes)} · or <u>browse files</u></div>
      </button>
      <input ref={input} type="file" hidden accept={videoTypes.join(',')} onChange={event => { if (event.target.files.length) onFile(event.target.files[0]); event.target.value = ''; }} />
      {children}
    </div>
  );
}

export function JobStatus({ job }) {
  if (job.status === 'done') return <><span className="chip good"><Check size={11} />Completed</span>{job.published && <span className="chip yellow" style={{ marginLeft: 6 }}>Published</span>}</>;
  if (job.status === 'running') {
    const stage = STAGE[followed(job.stages || [], job.status)];
    return <span className="chip blue"><span className="spin sm" />{stage ? stage.title : 'Starting'}</span>;
  }
  if (job.status === 'queued') return <span className="chip outline">{job.options.where === 'worker' ? 'Waiting for a worker' : 'Queued'}</span>;
  if (job.status === 'stopping') return <span className="chip outline"><span className="spin sm w" />Stopping</span>;
  if (job.status === 'failed') return <span className="chip bad"><X size={11} />Failed</span>;
  return <span className="chip outline">Stopped</span>;
}

export function JobsPage({ jobs, footage, status, query, go, onFile, onNew }) {
  const totals = status.totals;
  const rows = jobs.filter(job => matches(query, job.run, job.footage, job.status, job.worker));
  return (
    <div className="jobs">
      <div className="page-h">
        <div>
          <h1>Jobs</h1>
          <p>Upload raw footage. The Banana Engine turns it into upscaled, labeled hand–object clips.</p>
        </div>
        <div className="r">
          <a className="btn" href="/explore" target="_blank" rel="noreferrer"><ArrowUpRight size={14} />Open the catalog</a>
          <button className="btn primary" onClick={onNew}><Plus size={15} />New job</button>
        </div>
      </div>

      <DropZone limits={status.limits} onFile={onFile}>
        <div className="div" />
        <div className="pipe">
          <div className="h">Pipeline<span className="chip yellow">Hand–object interactions</span></div>
          <div className="pipe-steps">
            {STAGES.map(stage => { const Icon = stage.icon; return <span key={stage.key}><Icon size={12} />{stage.short}</span>; })}
          </div>
          <div className="pipe-note"><ShieldCheck size={13} />Faces are blurred before a clip is cut, upscaled or shown to a model.</div>
        </div>
      </DropZone>

      <div className="stats">
        <div className="card stat"><div className="k"><Film size={12} />Footage processed</div><div className="big">{(totals.footage_seconds / 60).toFixed(1)}<small>min</small></div><div className="s">{plural(totals.jobs, 'finished job')}</div></div>
        <div className="card stat"><div className="k"><Check size={12} />Clips accepted</div><div className="big">{totals.clips}</div><div className="s">labeled, upscaled, face-blurred</div></div>
        <div className="card stat"><div className="k"><Layers size={12} />Verified variants</div><div className="big">{totals.variants}</div><div className="s">object swaps that passed the check</div></div>
        <div className="card stat"><div className="k"><Coins size={12} />Spend per clip</div><div className="big">{totals.clips ? ((100 * totals.usd) / totals.clips).toFixed(1) : '–'}<small>¢</small></div><div className="s">Claude + fal, all jobs</div></div>
      </div>

      <div className="card" style={{ overflow: 'hidden' }}>
        <div className="card-h"><h3>Recent jobs</h3><span className="sub">{query.trim() ? `${rows.length} of ${plural(jobs.length, 'job')}` : plural(jobs.length, 'job')}</span></div>
        {rows.length ? (
          <div className="table-wrap">
            <table className="t">
              <thead><tr><th>Job</th><th>Footage</th><th>Clips</th><th>Variants</th><th>Labeler</th><th>Spend</th><th>Duration</th><th>Status</th><th>Started</th></tr></thead>
              <tbody>
                {rows.map(job => {
                  const seconds = jobSeconds(job, footage);
                  const spend = (job.claude_usd || 0) + (job.fal_usd || 0);
                  const began = !!(job.stages && job.stages.length);
                  return (
                    <tr key={job.id} className={`go ${job.status === 'running' ? 'new' : ''}`} tabIndex={0} onClick={() => go(`/admin/jobs/${job.id}`)}
                      onKeyDown={event => { if (event.key === 'Enter') go(`/admin/jobs/${job.id}`); }}>
                      <td className="name"><div className="cell-job"><div className="job-ico"><FileVideo size={14} /></div><div><span className="mono" style={{ fontSize: 11.5 }}>{job.run}</span><span className="fn">{job.footage}</span></div></div></td>
                      <td className="num">{seconds != null ? fmtClock(seconds) : '–'}</td>
                      <td className="num">{began ? job.clips.accepted : '–'}</td>
                      <td className="num">{job.variants || '–'}</td>
                      <td>{labelerName(job.labeler)}</td>
                      <td className="num">{began ? fmtUSD(spend) : '–'}</td>
                      <td className="num">{job.elapsed ? fmtClock(job.elapsed) : '–'}</td>
                      <td><JobStatus job={job} /></td>
                      <td className="num" style={{ color: 'var(--muted)' }}>{fmtWhen(job.started_at || job.created_at)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty">
            <div className="ico"><FileVideo size={18} /></div>
            <b>{jobs.length ? 'No job matches that search' : 'No jobs yet'}</b>
            <p>{jobs.length ? 'Try the name of a file or of a run.' : 'Drop a video above. The engine will work through it stage by stage, and you can watch.'}</p>
          </div>
        )}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ the new-job sheet
const LENGTHS = [15, 30, 60, 120, 300, 600];

// One choice of the pipeline: a tile that shows the current value, with a native select laid over it.
function Choice({ icon: Icon, name, value, current, options, onChange, locked, plain }) {
  return (
    <div className={`opt ${locked ? 'locked' : ''} ${plain ? 'plain' : ''}`}>
      <div className="k"><Icon size={12} />{name}</div>
      <div className="v">{locked && !plain && <Lock size={12} />}<span>{value}</span>{!locked && <ChevronDown size={13} className="dd" />}</div>
      {!locked && (
        <select aria-label={name} value={String(current)} onChange={event => onChange(event.target.value)}>
          {options.map(option => <option key={option.value} value={String(option.value)} disabled={option.disabled}>{option.label}</option>)}
        </select>
      )}
    </div>
  );
}

const codecName = codec => ({ h264: 'H.264', hevc: 'H.265', mpeg4: 'MPEG-4', vp9: 'VP9', vp8: 'VP8', av1: 'AV1' }[codec] || (codec || '–').toUpperCase());

export function NewJobSheet({ token, status, footage, initial, onClose, onStarted, onUploaded, onError }) {
  const [item, setItem] = useState(() => (initial.name ? footage.find(entry => entry.name === initial.name) || null : null));
  const [sending, setSending] = useState(null);       // { name, total, sent, error }
  const [starting, setStarting] = useState(false);
  const input = useRef(null);
  const turn = useRef(0);          // which upload this sheet is showing; an older one that finishes is ignored
  const limits = useRef(status.limits);
  limits.current = status.limits;

  const send = useCallback(async file => {
    const mine = ++turn.current;
    setItem(null);
    setSending({ name: file.name, total: file.size, sent: 0 });
    try {
      const entry = await upload(token, file, limits.current, sent => { if (turn.current === mine) setSending(now => (now ? { ...now, sent } : now)); });
      onUploaded();
      if (turn.current !== mine) return;
      setItem(entry);
      setSending(null);
    } catch (error) {
      if (turn.current === mine) setSending(now => (now ? { ...now, error: error.message } : now));
    }
  }, [token, onUploaded]);
  const dropped = useRef(false);
  useEffect(() => {      // the file the sheet was opened with is sent once, whatever else changes meanwhile
    if (initial.file && !dropped.current) { dropped.current = true; send(initial.file); }
  }, [initial.file, send]);
  useEffect(() => {
    const onKey = event => { if (event.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  // where it can run
  const here = status.engine.ready && status.engine.claude_key;
  const workers = status.workers;
  const [where, setWhere] = useState('');             // '': whatever is best right now
  const place = where || (workers.length || !here ? 'worker' : 'server');
  const worker = workers[0];
  const onServer = place === 'server';
  const slow = onServer && status.engine.device === 'cpu';      // no GPU: many times the length of the footage
  const canSwap = !onServer && workers.some(entry => entry.fal && entry.fal.key);
  const length = item && item.seconds ? item.seconds : 0;

  // what to do with it; null means "the usual for this file and this machine"
  const [start, setStart] = useState(0);
  const [span, setSpan] = useState(null);
  const [clips, setClips] = useState(null);
  const [quality, setQuality] = useState(null);
  const [swaps, setSwaps] = useState(0);
  const left = Math.max(0, length - start);
  const usual = slow ? Math.min(30, left) : left > 600 ? 300 : left;
  const seconds = Math.min(span == null ? usual : span === 'all' ? left : span, left || Infinity);
  const whole = length > 0 && start === 0 && seconds >= length - 0.05;
  const maxClips = clips || (slow ? 3 : seconds >= 240 ? 12 : seconds >= 100 ? 8 : 4);
  const size = quality || (slow ? 'fast' : 'full');
  const variants = canSwap ? Math.min(swaps, maxClips) : 0;

  // measured on an M1 Pro (worker) and on the 2-CPU server: tracking, then cutting and upscaling per clip
  const minutes = slow ? (seconds * 30 + maxClips * 200 + 60) / 60 : (seconds / 2.6 + maxClips * (size === 'full' ? 18 : 9) + 12 + variants * 70) / 60;
  const claude = maxClips * 0.0021 + Math.ceil((maxClips * 4) / 6) * 0.0004 + variants * 0.0007;
  const fal = variants * 0.11;

  const begin = async () => {
    setStarting(true);
    try {
      const job = await post(token, '/jobs', { footage: item.name, start, seconds: whole ? null : Math.round(seconds * 10) / 10, max_clips: maxClips, quality: size, where: place, multiply: variants });
      onStarted(job);
    } catch (error) {
      onError(error.message);
      setStarting(false);
    }
  };

  const sent = sending ? sending.sent / Math.max(1, sending.total) : 1;
  const reading = sending && !sending.error && sending.sent >= sending.total;
  const ready = !!item && length > 0;
  const processor = onServer ? withDevice('This server', status.engine.device) : worker ? withDevice(worker.name, worker.device) : 'The next worker to connect';
  const labeler = labelerName(onServer || !worker ? status.engine.labeler : worker.labeler || status.engine.labeler);

  return (
    <>
      <div className="scrim" onClick={onClose} />
      <aside className="sheet" role="dialog" aria-label="New job">
        <div className="sheet-h">
          <div><h2>New job</h2><div className="sub">Raw footage → Banana Engine → labeled clips</div></div>
          <button style={{ marginLeft: 'auto' }} className="icon-btn" onClick={onClose} aria-label="Close"><X size={16} /></button>
        </div>
        <div className="sheet-b">
          <div className="sec">
            <div className="sec-h">Footage</div>
            {(sending || item) ? (
              <div className="file">
                <div className="fi"><FileVideo size={18} /></div>
                <div style={{ minWidth: 0 }}>
                  <div className="n">{sending ? sending.name : item.name}</div>
                  <div className="s num">
                    {sending && sending.error ? <span className="error">{sending.error}</span>
                      : sending ? <><span>Uploading {Math.round(sent * 100)}%</span><span>{(sending.sent / 1e6).toFixed(1)} of {fmtBytes(sending.total)}</span></>
                        : <><span className="ok"><Check size={12} />Uploaded</span><span>{fmtBytes(item.bytes)}</span></>}
                  </div>
                </div>
                <button className="chip outline" onClick={() => { turn.current += 1; setSending(null); setItem(null); }}>Change</button>
                <div className="bar"><i className={sending ? '' : 'done'} style={{ width: `${sent * 100}%` }} /></div>
              </div>
            ) : (
              <>
                <button className="file" onClick={() => input.current.click()}>
                  <div className="fi"><Upload size={18} /></div>
                  <div><div className="n">Choose a video file</div><div className="s">MP4, AVI, MOV or MKV up to {fmtBytes(status.limits.upload_bytes)}</div></div>
                </button>
                {footage.slice(0, 6).map(entry => (
                  <button key={entry.name} className="file" onClick={() => setItem(entry)}>
                    <div className="fi"><FileVideo size={18} /></div>
                    <div style={{ minWidth: 0 }}><div className="n">{entry.name}</div><div className="s num"><span>Uploaded {fmtWhen(entry.uploaded_at)}</span><span>{entry.seconds ? fmtClock(entry.seconds) : ''}</span><span>{fmtBytes(entry.bytes)}</span></div></div>
                  </button>
                ))}
              </>
            )}
            <input ref={input} type="file" hidden accept={videoTypes.join(',')} onChange={event => { if (event.target.files.length) send(event.target.files[0]); event.target.value = ''; }} />
            {reading && <div className="reading"><span className="spin sm" />Reading the file…</div>}
            {ready && (
              <>
                <div className="probe">
                  <div><div className="k">Duration</div><div className="v num">{fmtClock(item.seconds)}</div></div>
                  <div><div className="k">Resolution</div><div className="v num">{item.width}×{item.height}</div></div>
                  <div><div className="k">Frame rate</div><div className="v num">{item.fps ? `${Math.round(item.fps)} fps` : '–'}</div></div>
                  <div><div className="k">Codec</div><div className="v">{codecName(item.codec)}</div></div>
                </div>
                <div className="film" style={{ backgroundImage: `url("/api/admin/footage/${encodeURIComponent(item.name)}/film")` }} />
                {item.source && <div className="src-line"><ShieldCheck size={13} />Source: {item.source.dataset} · camera {item.source.camera} · recorded {item.source.recorded} · {item.source.license}</div>}
              </>
            )}
          </div>

          {ready && (
            <div className="sec">
              <div className="sec-h">Pipeline<span className="chip yellow">Hand–object interactions</span></div>
              <div className="opts">
                <div className="opt wide">
                  <div className="k"><Scissors size={12} />Stretch to process</div>
                  <div className="pair">
                    <span><small>from</small></span>
                    <input type="number" aria-label="Start, in seconds" min="0" max={Math.max(0, Math.floor(length - 1))} step="1" value={start}
                      onChange={event => setStart(Math.min(Math.max(0, Number(event.target.value) || 0), Math.max(0, length - 1)))} />
                    <span><small>s, for</small></span>
                    <span className="sel"><span>{whole ? `the whole file (${fmtClock(length)})` : seconds >= left - 0.05 ? `the rest (${fmtClock(seconds)})` : seconds >= 60 ? `${seconds / 60} min` : `${Math.round(seconds)} s`}</span><ChevronDown size={12} />
                      <select aria-label="Length" value={span == null ? '' : String(span)} onChange={event => setSpan(event.target.value === 'all' ? 'all' : Number(event.target.value))}>
                        <option value="" disabled>Length</option>
                        <option value="all">{start ? 'To the end' : 'The whole file'} ({fmtClock(left)})</option>
                        {LENGTHS.filter(value => value < left).map(value => <option key={value} value={value}>{value >= 60 ? `${value / 60} min` : `${value} s`}</option>)}
                      </select>
                    </span>
                  </div>
                </div>
                <Choice icon={Crop} name="Clips to refine" value={`${maxClips} best moments`} current={maxClips} onChange={value => setClips(Number(value))}
                  options={[2, 3, 4, 6, 8, 12, 16, 24].map(n => ({ value: n, label: `${n} best moments` }))} />
                {canSwap
                  ? <Choice icon={Sparkles} name="Object-swap variants" value={variants ? plural(variants, 'clip') : 'Off'} current={variants} onChange={value => setSwaps(Number(value))}
                    options={[0, 1, 2, 3].map(n => ({ value: n, label: n ? `${plural(n, 'clip')} · about ${fmtUSD(n * 0.11)} of fal credit` : 'Off' }))} />
                  : <Choice icon={Sparkles} name="Object-swap variants" value={onServer ? 'Off · made on a worker' : 'Off · no worker with a fal key'} locked plain />}
                <Choice icon={ImageUpscale} name="Upscale" value={`4× Real-ESRGAN · up to ${size === 'full' ? '1080p' : '720p'}`} current={size} onChange={setQuality}
                  options={[{ value: 'fast', label: 'Up to 720p' }, { value: 'full', label: 'Up to 1080p (about twice as slow)' }]} />
                <Choice icon={Tags} name="Labeler" value={labeler} locked plain />
                <Choice icon={ShieldCheck} name="Face blur" value="Always on" locked />
                <Choice icon={Cpu} name="Worker" value={processor} current={place} onChange={setWhere}
                  options={[{ value: 'worker', label: worker ? `${workers.map(entry => entry.name).join(', ')} (worker${workers.length > 1 ? 's' : ''})` : 'A worker (none is connected yet)' },
                    { value: 'server', label: !here ? 'This server (no engine)' : status.engine.device === 'cpu' ? 'This server (CPU, slow)' : 'This server', disabled: !here }]} />
              </div>
            </div>
          )}

          {ready && (
            <div className="est">
              <Clock size={15} />
              <span>Estimate <b>≈ {minutes < 1.5 ? '1 min' : `${Math.round(minutes)} min`}</b> on {onServer ? 'this server' : worker ? 'this worker' : 'an Apple-silicon worker'}</span>
              <span style={{ color: 'var(--faint)' }}>·</span>
              <span><b>≈ {fmtUSD(claude + fal)}</b> (Claude {(claude * 100).toFixed(1)}¢{variants ? `, fal ${Math.round(fal * 100)}¢` : ''})</span>
            </div>
          )}
          {ready && slow && <div className="hint">This server has no GPU and shares its two processors with other sites, so it takes many times the length of the footage. A worker is the fast way: on a Mac with the engine, run <code>uv run python -m relentless.worker {window.location.origin}</code>.</div>}
          {ready && !onServer && !worker && <div className="hint">No worker is connected, so the job will wait. On a machine with the engine, run <code>uv run python -m relentless.worker {window.location.origin}</code>.</div>}
          {item && !length && <div className="hint error">This file's length could not be read, so it cannot be processed.</div>}
        </div>
        <div className="sheet-f">
          <span style={{ fontSize: 11.5, color: 'var(--muted)' }}>Clips wait on the job's page until you publish them.</span>
          <span style={{ flex: 1 }} />
          <button className="btn" onClick={onClose}>Cancel</button>
          <button className="btn primary lg" disabled={!ready || starting || seconds < 1} onClick={begin}>
            {starting ? <span className="spin sm ink" /> : <Sparkles size={15} />}{starting ? 'Starting…' : 'Start job'}
          </button>
        </div>
      </aside>
    </>
  );
}
