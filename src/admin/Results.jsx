// What a finished job made: the numbers, how much of the footage each stage kept, and the dataset of clips.

import { useEffect, useMemo, useState } from 'react';
import { ArrowUpRight, Check, MapPin, Sparkles, X } from 'lucide-react';
import { fmtCents, fmtClock, fmtUSD, labelerName, plural, withDevice } from './lib/format.js';
import { Video, makeClock } from './lib/media.jsx';
import { passedVariants } from './lib/run.js';
import { Actions, ClipPlayer } from './Stages.jsx';

const FUNNEL = [
  ['raw', 'Raw footage', value => `${Math.round(value)} s`], ['motion', 'With motion', value => `${Math.round(value)} s`],
  ['people', 'People in view', value => `${value} tracks`], ['activity', 'Hands busy', value => `${Math.round(value)} s`],
  ['accepted', 'Clips kept', value => plural(value, 'clip')],
];

function DataCard({ run, clip, index, onOpen }) {
  const [hover, setHover] = useState(false);
  const label = clip.label || {};
  const kept = clip.status === 'accepted';
  const poster = run.url(clip.files.poster), video = run.url(clip.files.upscaled);
  const tiers = clip.tiers || [];
  return (
    <button className={`dcard ${kept ? '' : 'no'}`} style={{ animationDelay: `${Math.min(index * 40, 500)}ms` }} onClick={() => onOpen(clip.id)}
      onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)} onFocus={() => setHover(true)} onBlur={() => setHover(false)}>
      <div className="m">
        {poster && <img src={poster} alt="" loading="lazy" />}
        {hover && video && <video src={video} muted loop autoPlay playsInline />}
        <span className={`vchip ${kept ? '' : 'no'}`}>{kept ? <Check size={10} /> : <X size={10} />}{label.main_object || clip.id}</span>
        <span className="vchip dur num">{clip.seconds.toFixed(1)} s</span>
      </div>
      <div className="b">
        <div className="c">{label.caption || clip.reject_reason || 'Not labeled'}</div>
        <div className="f">
          {kept ? <span className="tier">Bulk</span> : <span className="tier out">{clip.status === 'rejected' ? 'Rejected' : 'Not kept'}</span>}
          {tiers.includes('labeled+') && <span className="tier">Labeled+</span>}
          {tiers.includes('studio') && <span className="tier studio">Studio</span>}
          {passedVariants(clip).length > 0 && <span className="chip yellow"><Sparkles size={9} />variant</span>}
          {label.quality != null && <span className="q num">q {label.quality}/10</span>}
        </div>
      </div>
    </button>
  );
}

export function Results({ run, onOpen }) {
  const { state, clips, total, costs, job, size, source } = run;
  const funnel = Object.fromEntries((state.funnel || []).map(row => [row.key, row]));
  const kept = clips.filter(clip => clip.status === 'accepted').sort((a, b) => (b.label.quality - a.label.quality) || a.id.localeCompare(b.id));
  const others = clips.filter(clip => clip.status !== 'accepted');
  const variants = kept.reduce((sum, clip) => sum + passedVariants(clip).length, 0);
  const rows = FUNNEL.filter(([key]) => funnel[key] && funnel[key].share != null);
  const tier = name => kept.filter(clip => (clip.tiers || []).includes(name)).length;
  return (
    <div className="res">
      <div className="res-top">
        <div className="sumtiles">
          <div><div className="k">Footage in</div><div className="big">{fmtClock(total)}<small>min</small></div><div className="s">{size[0]}×{size[1]}{source.fps ? ` · ${Math.round(source.fps)} fps` : ''}</div></div>
          <div><div className="k">Clips out</div><div className="big">{kept.length}{variants > 0 && <small>+ {plural(variants, 'variant')}</small>}</div><div className="s">all labeled, 4× upscaled</div></div>
          <div><div className="k">Processing</div><div className="big">{fmtClock(state.elapsed)}</div><div className="s">on {withDevice(job.worker || 'this server', run.device)}</div></div>
          <div><div className="k">Spend</div><div className="big">{fmtUSD((costs.anthropic_usd || 0) + (costs.fal_usd || 0))}</div><div className="s">Claude {fmtCents(costs.anthropic_usd)} · fal {fmtCents(costs.fal_usd)}</div></div>
        </div>
        <div className="card fun">
          <div className="t">Footage kept at each stage<span>share of raw pixel-seconds</span></div>
          {rows.map(([key, name, show], i) => {
            const share = funnel[key].share;
            return (
              <div key={key} className="frow">
                <span title={show(funnel[key].value)}>{name}</span>
                <div className="bar"><i style={{ width: `${Math.max(0.6, share * 100)}%`, animationDelay: `${i * 80}ms` }} /></div>
                <span className="val num">{(share * 100).toFixed(share < 0.1 ? 1 : 0)}%</span>
              </div>
            );
          })}
        </div>
      </div>
      <div className="card ds">
        <div className="card-h">
          <h3>Dataset</h3><span className="sub">{job.run} · {plural(kept.length, 'clip')}{variants ? ` · ${plural(variants, 'verified variant')}` : ''}{others.length ? ` · ${others.length} not kept` : ''}</span>
          <div className="r">
            <span className="chip outline">Bulk {kept.length}</span>
            {tier('labeled+') > 0 && <span className="chip outline">Labeled+ {tier('labeled+')}</span>}
            {tier('studio') > 0 && <span className="chip yellow">Studio {tier('studio')}</span>}
            <span className="chip outline">Sort: quality</span>
          </div>
        </div>
        {clips.length ? (
          <div className="ds-grid">
            {[...kept, ...others].map((clip, i) => <DataCard key={clip.id} run={run} clip={clip} index={i} onOpen={onOpen} />)}
          </div>
        ) : <div className="empty"><b>No clips came out of this footage</b><p>The engine found no moment where a hand works on an object. Try a different stretch of the recording.</p></div>}
      </div>
    </div>
  );
}

// One clip in full: the clip with its labels drawn on it, the label card, where it comes from, its variants.
export function ClipDrawer({ run, clip, onClose }) {
  const clock = useMemo(makeClock, []);
  const label = clip.label || {};
  const tiers = clip.tiers || [];
  const source = run.source;
  useEffect(() => {
    const onKey = event => { if (event.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);
  return (
    <>
      <div className="scrim" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-label={`Clip ${clip.id}`}>
        <div className="sheet-h">
          <div style={{ minWidth: 0 }}>
            <h2>{clip.id}{label.main_object ? ` · ${label.main_object}` : ''}</h2>
            <div className="sub num">{fmtClock(clip.start_s)}–{fmtClock(clip.start_s + clip.seconds)} in the source · person {clip.track}
              {clip.native_size && clip.size ? ` · ${clip.native_size[0]}×${clip.native_size[1]} → ${clip.size[0]}×${clip.size[1]}` : ''}</div>
          </div>
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 8, alignItems: 'center' }}>
            {clip.status === 'accepted' ? <span className="tier">Bulk</span> : <span className="tier out">{clip.status === 'rejected' ? 'Rejected' : clip.status}</span>}
            {tiers.includes('labeled+') && <span className="tier">Labeled+</span>}
            {tiers.includes('studio') && <span className="tier studio">Studio</span>}
            <button className="icon-btn" onClick={onClose} aria-label="Close"><X size={16} /></button>
          </div>
        </div>
        <div className="drawer-b">
          <ClipPlayer run={run} clip={clip} clock={clock} />
          {label.caption && <div className="cap">{label.caption}</div>}
          {clip.status !== 'accepted' && clip.reject_reason && (
            <div className="verdict no"><X size={16} /><div><div className="t">Not in the dataset</div><div className="s">{clip.reject_reason}</div></div></div>
          )}
          {label.labels && label.labels.length > 0 && <Actions labels={label.labels} clock={clock} />}
          <div className="lineage">
            <span className="chip outline"><MapPin size={11} />{source.dataset ? `${source.dataset} · camera ${source.camera}` : source.file}</span>
            <span className="chip outline num">frames {clip.frames[0]}–{clip.frames[1]}</span>
            <span className="chip outline num">crop {clip.crop.join(', ')}</span>
            <span className="chip outline">person {clip.track} · {(clip.hands || []).join(' + ')} hand</span>
            <span className="chip outline">faces blurred</span>
            {label.quality != null && <span className="chip outline num">quality {label.quality}/10</span>}
            {clip.label && <span className="chip outline">labeled by {labelerName(run.engine.labeler)}</span>}
          </div>
          {(clip.variants || []).map(variant => (
            <div key={variant.id} className="varrow">
              <div className="vid">{run.url(clip.files.poster) && <img className="full" src={run.url(clip.files.poster)} alt="" />}</div>
              <div className="vid">{run.url(variant.files.video) && <Video src={run.url(variant.files.video)} />}</div>
              <div className="txt">
                <b>Variant · {(variant.edit || '').replace(' -> ', ' → ')}</b>
                <span>{variant.passed ? 'Passed' : 'Did not pass'} the check {variant.verify ? `${variant.verify.score}/10` : ''}{variant.passed ? ' · labels carried over' : ' · not in the dataset'}</span>
                {variant.passed && <span className="chip yellow"><Sparkles size={9} />Studio tier</span>}
              </div>
            </div>
          ))}
        </div>
      </aside>
    </>
  );
}

export function Toast({ clips, variants, onClose }) {
  useEffect(() => {
    const timer = setTimeout(onClose, 9000);
    return () => clearTimeout(timer);
  }, [onClose]);
  return (
    <div className="toast" role="status">
      <div className="ti"><Check size={14} strokeWidth={2.8} /></div>
      <div><div className="t">Published to Catalog</div><div className="s">{plural(clips, 'clip')}{variants ? ` and ${plural(variants, 'variant')}` : ''} {clips + variants === 1 ? 'is' : 'are'} now searchable</div></div>
      <a href="/explore?sort=newest" target="_blank" rel="noreferrer">Open<ArrowUpRight size={13} /></a>
    </div>
  );
}
