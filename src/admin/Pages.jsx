// The smaller pages of the console: what the site has to work with (Overview), the uploaded footage,
// the datasets finished jobs made, and the boxes that will send footage in on their own.

import { ArrowUpRight, Boxes as BoxesIcon, Check, Cpu, Database, FileVideo, HardDrive, Plus, Sparkles, Store, Trash2, TriangleAlert } from 'lucide-react';
import { deviceName, fmtBytes, fmtClock, fmtUSD, fmtWhen, labelerName, plural } from './lib/format.js';
import { DropZone, matches } from './Jobs.jsx';

export const workerCommand = () => `cd engine && uv run python -m relentless.worker ${window.location.origin}`;

// Things about this deployment that the person running it should know.
export function notices(status) {
  const { engine, storage, machine, limits, workers } = status;
  const free = machine.memory_available_bytes;
  const notes = [];
  if (!engine.ready && !workers.length) notes.push('The engine is not installed on this server and no worker is connected, so footage can be uploaded but not processed yet.');
  if (engine.ready && !engine.claude_key) notes.push('ANTHROPIC_API_KEY is not set on this server, so jobs that run here cannot label clips. A worker uses its own key.');
  if (engine.ready && engine.claude_key && free != null && free < limits.job_memory_bytes) notes.push(`Only ${fmtBytes(free)} of memory is free on this server right now. A job here needs about ${fmtBytes(limits.job_memory_bytes)} and will not start until there is room. A worker is not affected.`);
  if (storage.kind === 'container') notes.push('Uploads, jobs and their clips are stored inside the container and are lost on the next deploy. Mount a volume at /data to keep them.');
  return notes;
}

export function OverviewPage({ status, jobs }) {
  const { engine, storage, machine, library, workers, totals } = status;
  const notes = notices(status);
  const running = jobs.filter(job => job.status === 'running').length;
  return (
    <div className="jobs">
      <div className="page-h"><div><h1>Overview</h1><p>What the Banana Engine has to work with here, and what it has made.</p></div></div>
      {notes.map(note => <div key={note} className="notice"><TriangleAlert size={14} /><span>{note}</span></div>)}
      <div className="tiles4">
        <div className="card stat"><div className="k"><Cpu size={12} />Processors</div><div className="big">{workers.length + (engine.ready ? 1 : 0)}</div><div className="s">{plural(workers.length, 'worker')}{engine.ready ? ' + this server' : ''} · {running ? `${running} running` : 'idle'}</div></div>
        <div className="card stat"><div className="k"><HardDrive size={12} />Free space</div><div className="big">{fmtBytes(storage.free_bytes).replace(/ (GB|MB)$/, '')}<small>{fmtBytes(storage.free_bytes).slice(-2)}</small></div><div className="s">{storage.kind === 'volume' ? 'on a mounted volume' : storage.kind === 'container' ? 'inside the container' : 'on this disk'}</div></div>
        <div className="card stat"><div className="k"><Store size={12} />Catalog</div><div className="big">{library.clips}<small>clips</small></div><div className="s">{library.processed_here} published from jobs here</div></div>
        <div className="card stat"><div className="k"><Sparkles size={12} />Spend so far</div><div className="big">{fmtUSD(totals.usd)}</div><div className="s">Claude + fal, over {plural(totals.jobs, 'finished job')}</div></div>
      </div>
      <div className="two-col">
        <div className="card" style={{ overflow: 'hidden' }}>
          <div className="card-h"><h3>Where jobs run</h3><span className="sub">a worker is another machine with the engine, usually one with a GPU</span></div>
          <div className="rowlist">
            {workers.map(worker => (
              <div key={worker.name}>
                <span className={`dot ${worker.busy ? 'busy' : 'ok'}`} />
                <span className="n">{worker.name}</span>
                <span className="s">{deviceName(worker.device)} · labels with {labelerName(worker.labeler)}{worker.fal && worker.fal.key ? ` · fal ${fmtUSD(worker.fal.spent_usd)} of $${worker.fal.cap_usd.toFixed(2)} used` : ' · no fal key, so no variants'}</span>
                <span className="r">{worker.busy ? 'busy' : 'idle'}</span>
              </div>
            ))}
            <div>
              <span className={`dot ${!engine.ready ? '' : engine.busy ? 'busy' : 'ok'}`} />
              <span className="n">This server</span>
              <span className="s">{engine.ready ? `${machine.cpus} CPUs${engine.device === 'cpu' ? ', no GPU' : ''}${machine.memory_available_bytes != null ? ` · ${fmtBytes(machine.memory_available_bytes)} of memory free` : ''} · labels with ${labelerName(engine.labeler)}` : 'the engine is not installed here'}</span>
              <span className="r">{!engine.ready ? 'off' : engine.busy ? 'busy' : 'idle'}</span>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="card-h"><h3>Connect a worker</h3></div>
          <div className="pad">
            <div className="hint">On a machine that has this repository and the engine set up (see <code>engine/README.md</code>), with <code>PEELSET_ADMIN_TOKEN</code> in its <code>.env</code>:</div>
            <pre className="cmd">{workerCommand()}</pre>
            <div className="hint">It asks this site for jobs, fetches the footage, runs the engine on its own GPU, shows each stage here while it works, and sends the clips back.</div>
          </div>
        </div>
      </div>
    </div>
  );
}

export function FootagePage({ footage, jobs, status, query, onFile, onNew, onDelete }) {
  const rows = footage.filter(entry => matches(query, entry.name));
  return (
    <div className="jobs">
      <div className="page-h">
        <div><h1>Footage</h1><p>Raw recordings uploaded to this site. One recording can feed any number of jobs.</p></div>
      </div>
      <DropZone limits={status.limits} onFile={onFile} slim />
      <div className="card" style={{ overflow: 'hidden' }}>
        <div className="card-h"><h3>Uploaded footage</h3><span className="sub">{plural(footage.length, 'file')} · {fmtBytes(footage.reduce((sum, entry) => sum + entry.bytes, 0))}</span></div>
        {rows.length ? (
          <div className="table-wrap">
            <table className="t">
              <thead><tr><th>File</th><th>Preview</th><th>Length</th><th>Resolution</th><th>Size</th><th>Jobs</th><th>Uploaded</th><th /></tr></thead>
              <tbody>
                {rows.map(entry => (
                  <tr key={entry.name}>
                    <td className="name"><div className="cell-job"><div className="job-ico"><FileVideo size={14} /></div><div>{entry.name}{entry.source && <span className="fn">{entry.source.dataset} · camera {entry.source.camera} · {entry.source.license}</span>}</div></div></td>
                    <td><div className="strip" style={{ backgroundImage: `url("/api/admin/footage/${encodeURIComponent(entry.name)}/film")` }} /></td>
                    <td className="num">{entry.seconds ? fmtClock(entry.seconds) : '–'}</td>
                    <td className="num">{entry.width ? `${entry.width}×${entry.height}${entry.fps ? ` · ${Math.round(entry.fps)} fps` : ''}` : '–'}</td>
                    <td className="num">{fmtBytes(entry.bytes)}</td>
                    <td className="num">{jobs.filter(job => job.footage === entry.name).length || '–'}</td>
                    <td className="num" style={{ color: 'var(--muted)' }}>{fmtWhen(entry.uploaded_at)}</td>
                    <td className="acts"><div>
                      <button className="btn" onClick={() => onNew(entry.name)}><Plus size={14} />New job</button>
                      <button className="icon-btn" aria-label={`Delete ${entry.name}`} title="Delete this file" onClick={() => onDelete(entry.name)}><Trash2 size={15} /></button>
                    </div></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty"><div className="ico"><FileVideo size={18} /></div><b>{footage.length ? 'No file matches that search' : 'Nothing uploaded yet'}</b>
            <p>{footage.length ? 'Try part of a file name.' : 'Drop a video above to upload it and start a job.'}</p></div>
        )}
      </div>
    </div>
  );
}

export function DatasetsPage({ jobs, query, go, onPublish }) {
  const rows = jobs.filter(job => job.status === 'done' && matches(query, job.run, job.footage));
  return (
    <div className="jobs">
      <div className="page-h">
        <div><h1>Datasets</h1><p>What finished jobs made. A dataset is private to this console until it is published to the catalog.</p></div>
        <div className="r"><a className="btn" href="/explore?sort=newest" target="_blank" rel="noreferrer"><ArrowUpRight size={14} />Open the catalog</a></div>
      </div>
      <div className="card" style={{ overflow: 'hidden' }}>
        <div className="card-h"><h3>Datasets</h3><span className="sub">{plural(rows.length, 'dataset')} · {rows.filter(job => job.published).length} in the catalog</span></div>
        {rows.length ? (
          <div className="table-wrap">
            <table className="t">
              <thead><tr><th>Dataset</th><th>Clips</th><th>Not kept</th><th>Variants</th><th>Labeler</th><th>Made</th><th>Catalog</th><th /></tr></thead>
              <tbody>
                {rows.map(job => (
                  <tr key={job.id} className="go" tabIndex={0} onClick={() => go(`/admin/jobs/${job.id}`)} onKeyDown={event => { if (event.key === 'Enter') go(`/admin/jobs/${job.id}`); }}>
                    <td className="name"><div className="cell-job"><div className="job-ico"><Database size={14} /></div><div><span className="mono" style={{ fontSize: 11.5 }}>{job.run}</span><span className="fn">{job.footage}</span></div></div></td>
                    <td className="num">{job.clips.accepted}</td>
                    <td className="num">{job.clips.planned - job.clips.accepted || '–'}</td>
                    <td className="num">{job.variants || '–'}</td>
                    <td>{labelerName(job.labeler)}</td>
                    <td className="num" style={{ color: 'var(--muted)' }}>{fmtWhen(job.finished_at)}</td>
                    <td>{job.published ? <span className="chip yellow"><Check size={11} />Published</span> : <span className="chip outline">Private</span>}</td>
                    <td className="acts"><div>{!job.published && job.clips.accepted > 0 && <button className="btn" onClick={event => { event.stopPropagation(); onPublish(job.id); }}>Publish</button>}</div></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty"><div className="ico"><Database size={18} /></div><b>No datasets yet</b><p>Every job that finishes leaves its clips here: labeled, upscaled, with the face of everyone in them blurred.</p></div>
        )}
      </div>
    </div>
  );
}

export function BoxesPage() {
  return (
    <div className="jobs">
      <div className="page-h"><div><h1>Boxes</h1><p>A box sits next to a store's CCTV recorder and sends its footage to the engine.</p></div></div>
      <div className="card">
        <div className="empty">
          <div className="ico"><BoxesIcon size={18} /></div>
          <b>No box is connected</b>
          <p>None has been set up with this site yet. Until one is, footage comes in by upload: drop a recording on the Jobs page.</p>
        </div>
      </div>
    </div>
  );
}

