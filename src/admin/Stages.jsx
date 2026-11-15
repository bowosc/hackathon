// The job page's live view: one picture per engine stage, drawn from what the run has written so far
// (its state, the frame a stage is on, and the previews in live/), plus the cards beside it.

import { memo, useEffect, useMemo, useRef, useState } from 'react';
import { Check, Clock, Hand, Loader, Sparkles, Tags, X } from 'lucide-react';
import { useJson } from './lib/api.js';
import { clamp, deviceName, fmtClock, fmtDur, fmtInt, fmtUSD, labelerName, lerp, plural } from './lib/format.js';
import { Sprite, Video, useClock, useInStep, useTyped, useVideoTime } from './lib/media.jsx';
import { LABELED, STAGE, UPSCALED, momentOf, passedVariants } from './lib/run.js';

const SKELETON = [[5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]];
const ticks = total => Array.from({ length: 6 }, (_, i) => fmtClock((total * i) / 5));
const verb = name => (name || '').replace(/_/g, ' ');

// Footage stages: the video takes the height the chart under it leaves, at the footage's own shape.
// (The layer is a size container; 102px is the chart with its heading and the gap.)
function footageBox([w, h]) {
  const width = `min(100cqw, calc((100cqh - 102px) * ${(w / h).toFixed(4)}))`;
  return { box: { aspectRatio: `${w} / ${h}`, width }, below: { width } };
}

// A clip stage: the clip takes the height that `under` pixels and the row(s) of clip tiles leave.
function clipBox(run, clip, under = 0) {
  const [w, h] = clip.size || [clip.crop[2], clip.crop[3]];
  const rows = Math.max(1, Math.ceil(run.clips.length / 12));
  const tiles = `${rows} * (100cqw - 66px) / 12 * 0.5625 + ${(rows - 1) * 6 + 10 + under}px`;
  return { aspectRatio: `${w} / ${h}`, width: `min(100cqw, calc((100cqh - (${tiles})) * ${(w / h).toFixed(4)}))` };
}

const Waiting = ({ children, spin }) => <div className="center"><div className="waiting">{spin && <span className="spin" />}<span>{children}</span></div></div>;

export function LiveView({ run, focus, following, onFollow, onResults, hero, picked, onPick, clock }) {
  const stage = STAGE[focus];
  const Icon = stage ? stage.icon : Loader;
  const Layer = LAYERS[focus];
  return (
    <div className="card live">
      <div className="card-h">
        <h3>{run.active && following ? 'Live view' : 'Stage view'}</h3>
        {stage && <span className="chip blue"><Icon size={11} />{stage.title}</span>}
        <span className="sub">{stage ? stage.blurb : 'Waiting for the engine'}</span>
        <div className="r">
          {run.active && !following && <button className="chip yellow" onClick={onFollow}>Follow the engine</button>}
          {onResults && <button className="chip yellow" onClick={onResults}>Back to the dataset</button>}
          <span className="chip outline"><span className="dot ok" style={{ width: 6, height: 6, boxShadow: 'none' }} />Faces blurred</span>
        </div>
      </div>
      <div className="live-b">
        <div className="layer" key={focus || 'none'}>
          {Layer ? <Layer run={run} hero={hero} picked={picked} onPick={onPick} clock={clock} /> : <Waiting spin={run.active}>Starting the engine…</Waiting>}
        </div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ 1. motion
const MotionChart = memo(function MotionChart({ timeline, top }) {
  const n = timeline.length, points = [];
  timeline.forEach((value, i) => { if (value != null) points.push([(i / Math.max(1, n - 1)) * 1000, 60 - Math.min(1, value / top) * 56]); });
  if (points.length < 2) return null;
  const line = points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' L');
  return (
    <>
      <path d={`M${points[0][0]},60 L${line} L${points[points.length - 1][0]},60 Z`} fill="rgba(57,135,229,.22)" />
      <path d={`M${line}`} fill="none" stroke="var(--blue)" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
    </>
  );
});

function MotionLayer({ run }) {
  const { now, kit, state, total } = run;
  const stage = run.by.motion || {};
  const running = stage.status === 'running';
  const live = running && now && now.stage === 'motion' ? now : null;
  const motion = state.motion;
  const video = useRef(null);
  const at = useVideoTime(video);
  const scan = run.url(kit.scan && kit.scan.file);
  const t = running ? (live ? live.t : 0) : scan ? at : 0;
  const timeline = running ? (live ? live.timeline : []) : motion ? motion.timeline : [];
  const threshold = (motion && motion.threshold) || 0.00035;
  const top = Math.max(threshold * 6, 0.002, ...timeline.filter(value => value != null));
  const rate = Math.min(8, Math.max(1, Math.round(total / 20)));
  const { box, below } = footageBox(run.size);
  return (
    <div className="vstack">
      <div className="vid" style={box}>
        {live && live.image ? <img className="full" src={live.image} alt="" />
          : !running && scan ? <Video videoRef={video} src={scan} rate={rate} />
            : <div className="ph"><div className="waiting">{running && <span className="spin" />}{running ? 'Reading the footage…' : run.active ? 'The replay of the scan is written when the clips are cut.' : 'This run has no replay of the scan.'}</div></div>}
        <div className="vchip" style={{ left: 10, top: 10 }}>
          {running ? <><span className="rec" />Scanning · 10 samples/s at 480 px</> : scan ? <>Scan replay · {rate}× speed</> : <>Motion filter</>}
        </div>
        <div className="vchip num" style={{ right: 10, top: 10 }}>{fmtClock(t)} / {fmtClock(total)}</div>
        {running && <div className="vbar" style={{ width: `${clamp(stage.progress || 0) * 100}%` }} />}
      </div>
      <div style={below}>
        <div className="chart-h"><b>Motion energy</b><span>share of pixels changing between samples</span>
          <div className="legend"><span><i style={{ background: 'var(--blue)' }} />kept</span><span><i style={{ background: '#3a3a38' }} />idle, dropped</span></div></div>
        <svg className="chart" viewBox="0 0 1000 66" preserveAspectRatio="none">
          <line x1="0" x2="1000" y1={60 - (threshold / top) * 56} y2={60 - (threshold / top) * 56} stroke="rgba(250,178,25,.6)" strokeDasharray="4 4" vectorEffect="non-scaling-stroke" />
          <MotionChart timeline={timeline} top={top} />
          {motion && <rect x="0" y="62" width="1000" height="4" rx="2" fill="#3a3a38" />}
          {motion && motion.spans.map(([a, b], i) => <rect key={i} x={(a / total) * 1000} y="62" width={((b - a) / total) * 1000} height="4" rx="2" fill="var(--blue)" />)}
          {total > 0 && <line x1={(t / total) * 1000} x2={(t / total) * 1000} y1="0" y2="66" stroke="#fff" strokeWidth="1.2" vectorEffect="non-scaling-stroke" opacity=".8" />}
        </svg>
        <div className="axis">{ticks(total).map((tick, i) => <span key={i}>{tick}</span>)}</div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ 2-4. tracking, activity, hand zoom on the footage
// A person's located hands between two samples. Undefined for a run from before the engine located hands
// (its page draws the wrists instead); a hand located in only one of the two samples shows for its half.
function handsBetween(a, b, k) {
  if (!a) return undefined;
  const out = {};
  for (const side of ['l', 'r']) {
    const ha = a[side], hb = b && b[side];
    if (ha && hb) out[side] = ha.map((v, j) => lerp(v, hb[j], k));
    else if (ha && k < 0.5) out[side] = ha;
    else if (hb && k >= 0.5) out[side] = hb;
  }
  return out;
}

// The people at footage second `ft`, between the two samples around it.
function peopleAt(tracks, ft) {
  const samples = tracks.samples;
  if (!samples.length) return [];
  let lo = 0, hi = samples.length - 1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (samples[mid].t <= ft) lo = mid; else hi = mid - 1; }
  const a = samples[lo], b = samples[Math.min(lo + 1, samples.length - 1)];
  const k = b.t > a.t ? clamp((ft - a.t) / (b.t - a.t)) : 0;
  const next = new Map(b.people.map(person => [person.id, person]));
  const out = [];
  for (const pa of a.people) {
    const pb = next.get(pa.id);
    if (pb) out.push({ ...pa, box: pa.box.map((v, j) => lerp(v, pb.box[j], k)), busy: k < 0.5 ? pa.busy : pb.busy,
      kp: pa.kp.map((q, j) => [lerp(q[0], pb.kp[j][0], k), lerp(q[1], pb.kp[j][1], k), Math.min(q[2], pb.kp[j][2])]),
      hands: handsBetween(pa.hands, pb.hands, k) });
    else if (k < 0.5) out.push(pa);
  }
  for (const pb of b.people) if (k >= 0.5 && !a.people.some(person => person.id === pb.id)) out.push(pb);
  return out;
}

function wristTrail(tracks, id, ft, which) {
  const points = [];
  for (const sample of tracks.samples) {
    if (sample.t > ft) break;
    if (sample.t < ft - 0.6) continue;
    const person = sample.people.find(q => q.id === id);
    if (person && person.kp[which][2] > 0.3) points.push(person.kp[which]);
  }
  return points;
}

// One person on the footage: box, skeleton, a tag, and the hands when they are busy: a box around each hand
// the engine located (nothing for a hand it could not locate), or the wrists for a run from before it
// located hands. `u` scales text and dots to the footage (they were sized for a frame 1,920 pixels wide).
function Person({ person, u, busyOn, tracks, ft }) {
  const [x1, y1, x2, y2] = person.box;
  const unclear = person.clear === false;      // too dark, foggy or hidden for the engine to read this person here
  const busy = busyOn && person.busy && !unclear;
  const tag = unclear ? `#${person.id} · unclear` : busy ? `#${person.id} · hands busy` : busyOn ? `#${person.id} · idle` : `#${person.id}`;
  return (
    <g opacity={unclear || (busyOn && !person.busy) ? 0.45 : 1}>
      {SKELETON.map(([a, b], j) => (person.kp[a][2] > 0.35 && person.kp[b][2] > 0.35
        ? <line key={j} x1={person.kp[a][0]} y1={person.kp[a][1]} x2={person.kp[b][0]} y2={person.kp[b][1]} stroke={busy ? 'rgba(140,190,250,.75)' : 'rgba(255,255,255,.45)'} strokeWidth="1.4" vectorEffect="non-scaling-stroke" />
        : null))}
      <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} rx={6 * u} fill={busy ? 'rgba(57,135,229,.10)' : 'none'} stroke={busy ? 'rgb(90,158,240)' : 'rgba(240,239,236,0.85)'}
        strokeWidth={busy ? 2 : 1.4} strokeDasharray={unclear || (busyOn && !person.busy) ? '6 5' : 'none'} vectorEffect="non-scaling-stroke" />
      <g transform={`translate(${x1}, ${Math.max(0, y1 - 30 * u)})`}>
        <rect width={(tag.length * 9.7 + 18) * u} height={26 * u} rx={5 * u} fill={busy ? 'rgba(42,110,200,.92)' : 'rgba(18,18,18,.78)'} />
        <text x={9 * u} y={18.5 * u} fontSize={17 * u} fontWeight="600" fill="#fff" fontFamily="Inter Variable, Inter, sans-serif">{tag}</text>
      </g>
      {busy && person.hands && ['l', 'r'].map(side => {
        const hand = person.hands[side];
        if (!hand) return null;
        const trail = tracks ? wristTrail(tracks, person.id, ft, side === 'l' ? 9 : 10) : [];
        return (
          <g key={side}>
            {trail.length > 1 && <polyline points={trail.map(q => `${q[0]},${q[1]}`).join(' ')} fill="none" stroke="rgba(120,180,255,.55)" strokeWidth="2.2" vectorEffect="non-scaling-stroke" strokeLinecap="round" />}
            <rect x={hand[0]} y={hand[1]} width={hand[2] - hand[0]} height={hand[3] - hand[1]} rx={4 * u} fill="rgba(90,158,240,.28)" stroke="#cfe3ff" strokeWidth="1.8" vectorEffect="non-scaling-stroke" />
            <text x={(hand[0] + hand[2]) / 2} y={hand[1] - 5 * u} textAnchor="middle" fontSize={14 * u} fontWeight="700" fill="#cfe3ff" stroke="rgba(18,18,18,.7)" strokeWidth={3 * u} paintOrder="stroke" fontFamily="Inter Variable, Inter, sans-serif">{side === 'l' ? 'L' : 'R'}</text>
          </g>
        );
      })}
      {busy && !person.hands && [9, 10].map(w => {
        if (person.kp[w][2] < 0.3) return null;
        const trail = tracks ? wristTrail(tracks, person.id, ft, w) : [];
        return (
          <g key={w}>
            {trail.length > 1 && <polyline points={trail.map(q => `${q[0]},${q[1]}`).join(' ')} fill="none" stroke="rgba(120,180,255,.55)" strokeWidth="2.2" vectorEffect="non-scaling-stroke" strokeLinecap="round" />}
            <circle cx={person.kp[w][0]} cy={person.kp[w][1]} r={13 * u} fill="rgba(90,158,240,.4)" />
            <circle cx={person.kp[w][0]} cy={person.kp[w][1]} r={6 * u} fill="#cfe3ff" stroke="var(--blue)" strokeWidth="2" vectorEffect="non-scaling-stroke" />
          </g>
        );
      })}
    </g>
  );
}

function Moment({ cand, u }) {
  const [x, y, w, h] = cand.crop;
  const lost = cand.located === false;      // the second look found no hand in it: it did not become a clip
  const label = cand.clip ? `moment → ${cand.clip}` : lost ? 'moment · no hand located' : `moment · ${cand.seconds.toFixed(1)} s`;
  return (
    <g opacity={lost ? 0.5 : 1}>
      <rect x={x} y={y} width={w} height={h} rx={5 * u} fill="rgba(240,122,60,.08)" stroke="var(--orange-hi)" strokeWidth="2.2" strokeDasharray={lost ? '7 6' : 'none'} vectorEffect="non-scaling-stroke" />
      {!lost && [[x, y], [x + w, y], [x, y + h], [x + w, y + h]].map(([cx, cy], j) => (
        <path key={j} d={`M${cx + (j % 2 ? -22 : 22) * u},${cy} L${cx},${cy} L${cx},${cy + (j < 2 ? 22 : -22) * u}`} fill="none" stroke="#ffb184" strokeWidth="3.4" vectorEffect="non-scaling-stroke" />
      ))}
      <g transform={`translate(${x}, ${y + h + 6 * u})`}>
        <rect width={(label.length * 9.4 + 18) * u} height={26 * u} rx={5 * u} fill="rgba(217,89,38,.95)" />
        <text x={9 * u} y={18.5 * u} fontSize={16.5 * u} fontWeight="600" fill="#fff" fontFamily="Inter Variable, Inter, sans-serif">{label}</text>
      </g>
    </g>
  );
}

// One row per person along the footage: when they are in view, when their hands are busy, their hand moments.
const Raster = memo(function Raster({ people, candidates, total, busy, moments }) {
  const rows = new Map(people.map((person, i) => [person.id, i]));
  const rh = 72 / Math.max(1, people.length);
  const x = seconds => (seconds / total) * 1000;
  return (
    <>
      {people.map((person, i) => <rect key={person.id} x={x(person.start_s)} y={i * rh} width={Math.max(1.5, x(person.seconds))} height={rh} fill="#64635d" />)}
      {busy && people.flatMap((person, i) => person.active.map(([a, b], j) => <rect key={`${person.id}-${j}`} x={x(a)} y={i * rh} width={Math.max(1.2, x(b - a))} height={rh} fill="var(--blue-hi)" />))}
      {moments && candidates.map((cand, k) => <rect key={k} x={x(cand.start_s)} y={(rows.get(cand.track) || 0) * rh - 0.6} width={Math.max(1.6, x(cand.seconds))} height={Math.max(rh * 0.8, 0.45) + 1.2} fill="var(--orange-hi)" />)}
    </>
  );
});

function TrackLayer({ run, phase, extra }) {
  const { now, kit, total, people, candidates } = run;
  const stage = run.by.people || {};
  const tracking = phase === 'people' && stage.status === 'running';
  const live = tracking && now && now.stage === 'people' ? now : null;
  const video = useRef(null);
  const at = useVideoTime(video);
  const preview = kit.preview;
  const src = run.url(preview && preview.file);
  const tracks = useJson(run.url(preview && preview.tracks));
  // footage time of the picture on screen: the preview's frame whose turn has come
  const shownAt = preview && preview.fps ? Math.floor(at * preview.fps + 1e-3) / preview.fps : at;
  const ft = live ? live.t : src ? preview.start_s + shownAt : 0;
  const shown = live ? live.people : tracks && src ? peopleAt(tracks, ft) : [];
  const moments = phase === 'zoom' && tracks && src ? tracks.candidates.filter(cand => ft >= cand.start_s && ft <= cand.start_s + cand.seconds)
    .map(cand => ({ ...cand, located: (candidates[cand.k] || {}).located })) : [];
  const [w, h] = run.size, u = w / 1920;
  const busyOn = phase !== 'people';
  const { box, below } = footageBox(run.size);
  const counts = live ? live.counts : [];
  const most = Math.max(1, ...counts.filter(n => n != null));
  return (
    <div className="vstack">
      <div className="vid" style={box}>
        {live && live.image ? <img className="full" src={live.image} alt="" />
          : !tracking && src ? <Video videoRef={video} src={src} />
            : <div className="ph"><div className="waiting">{tracking && <span className="spin" />}{tracking ? 'Loading the pose model…' : run.active ? 'The footage for this view is written when the clips are cut, a few moments from now.' : 'This run has no preview of the footage.'}</div></div>}
        <svg className="ov" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
          {shown.map(person => <Person key={person.id} person={person} u={u} busyOn={busyOn} tracks={live ? null : tracks} ft={ft} />)}
          {moments.map(cand => <Moment key={cand.k} cand={cand} u={u} />)}
        </svg>
        <div className="vchip" style={{ left: 10, top: 10 }}>
          {phase === 'people' && <>{tracking && <span className="rec" />}Pose model · {(run.engine.pose_model || 'yolo26s-pose').replace('-pose.pt', '').toUpperCase()} · ByteTrack</>}
          {phase === 'activity' && <>Wrist speed ÷ torso length</>}
          {phase === 'zoom' && <>2–8 s crops around hand bursts</>}
        </div>
        {(live || src) && <div className="vchip num" style={{ right: 10, top: 10 }}>
          {phase === 'people' ? `${shown.length} in view` : `${shown.filter(person => person.busy && person.clear !== false).length} of ${shown.length} busy`} · {fmtClock(ft)}.{Math.floor((ft % 1) * 10)}
        </div>}
        {tracking && <div className="vbar" style={{ width: `${clamp(stage.progress || 0) * 100}%` }} />}
      </div>
      <div style={below}>
        {tracking ? (
          <>
            <div className="chart-h"><b>{plural(live ? live.seen : 0, 'person track')} so far</b><span>how many people are in view, along the footage</span>
              <div className="legend"><span className="num">{live ? fmtInt(live.samples) : 0} frames through the model</span></div></div>
            <svg className="chart" viewBox="0 0 1000 72" preserveAspectRatio="none">
              {counts.map((n, i) => (n ? <rect key={i} x={(i / counts.length) * 1000} y={72 - (n / most) * 66} width={1000 / counts.length - 0.6} height={(n / most) * 66} fill="#64635d" /> : null))}
              {total > 0 && <line x1={(ft / total) * 1000} x2={(ft / total) * 1000} y1="0" y2="72" stroke="var(--blue-hi)" strokeWidth="1.4" vectorEffect="non-scaling-stroke" />}
            </svg>
          </>
        ) : (
          <>
            <div className="chart-h">
              <b>{plural(people.length, 'person track')}</b><span>one row per person-appearance, across all {fmtClock(total)}</span>
              <div className="legend">
                <span><i style={{ background: '#64635d' }} />in view</span>
                {busyOn && <span><i style={{ background: 'var(--blue-hi)' }} />hands busy</span>}
                {phase === 'zoom' && <span><i style={{ background: 'var(--orange-hi)' }} />moment</span>}
                {extra}
              </div>
            </div>
            <svg className="chart" viewBox="0 0 1000 72" preserveAspectRatio="none">
              {src && total > 0 && <rect x={(preview.start_s / total) * 1000} y="0" width={(preview.seconds / total) * 1000} height="72" fill="rgba(255,255,255,.05)" />}
              {total > 0 && <Raster people={people} candidates={candidates} total={total} busy={busyOn} moments={phase === 'zoom'} />}
              {src && total > 0 && <line x1={(ft / total) * 1000} x2={(ft / total) * 1000} y1="0" y2="72" stroke="#fff" strokeWidth="1" vectorEffect="non-scaling-stroke" opacity=".7" />}
            </svg>
          </>
        )}
        <div className="axis">{ticks(total).map((tick, i) => <span key={i}>{tick}</span>)}</div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ 4b. every hand moment
function Moments({ run, url, toggle }) {
  const sheet = run.kit.cands;
  const all = run.candidates;
  const seconds = all.reduce((sum, cand) => sum + cand.seconds, 0);
  return (
    <div className="fill">
      <div className="head-row">
        <span className="n num">{fmtInt(all.length)}</span>
        <span className="a">hand moments found</span>
        <span className="b num">{Math.round(seconds)} s of clip time · 2–8 s each · 16:9{sheet.index.length < all.length ? ` · the ${sheet.index.length} most active shown` : ''}</span>
        <div className="r"><span className="chip orange"><Hand size={11} />from {plural(new Set(all.map(cand => cand.track)).size, 'person', 'people')}</span>{toggle}</div>
      </div>
      <div className="scroll">
        <div className="cand-grid">
          {sheet.index.map((k, i) => {
            const cand = all[k];
            if (!cand) return null;
            const rated = cand.rating != null;
            return (
              <div key={k} className={`cand ${cand.rating === 0 || cand.located === false ? 'out' : ''} ${cand.clip ? 'pick' : ''}`} style={{ animationDelay: `${Math.min(i * 5, 900)}ms` }}
                title={`#${k} · person ${cand.track} · ${fmtClock(cand.start_s)}${rated ? ` · rated ${cand.rating}/3: ${cand.what}` : ''}${cand.clip ? ` · became clip ${cand.clip}` : ''}${cand.located === false ? ' · no hand could be located in it' : ''}`}>
                <Sprite url={url} cols={sheet.cols} count={sheet.index.length} at={i} style={{ position: 'absolute', inset: 0 }} />
                {cand.clip && <span className="c num">{cand.clip}</span>}
                <span className="d num">{cand.seconds.toFixed(1)}s</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function ZoomLayer({ run }) {
  const sheet = run.kit.cands, url = run.url(sheet && sheet.file);
  const [view, setView] = useState(null);      // null: the stills, once the engine has made them
  const shown = view || (url ? 'moments' : 'footage');
  const toggle = url && (
    <span className="seg-toggle">
      <button className={shown === 'moments' ? 'on' : ''} onClick={() => setView('moments')}>All moments</button>
      <button className={shown === 'footage' ? 'on' : ''} onClick={() => setView('footage')}>On the footage</button>
    </span>
  );
  return shown === 'moments' && url ? <Moments run={run} url={url} toggle={toggle} /> : <TrackLayer run={run} phase="zoom" extra={toggle} />;
}

// ------------------------------------------------------------------ 5. quick screen
function ScreenLayer({ run }) {
  const sheet = run.kit.screen, url = run.url(sheet && sheet.file);
  const stage = run.by.screen || {};
  if (stage.status === 'skipped') return <Waiting>The quick screen was skipped: clips were picked by how much the hands move.</Waiting>;
  if (!url) return <Waiting spin={stage.status === 'running'}>{stage.status === 'running' ? 'Taking four stills of each of the most active moments…' : 'This run has no stills of the screened moments.'}</Waiting>;
  const rows = sheet.index.map(k => ({ k, ...run.candidates[k] })).filter(cand => cand.seconds != null);
  const rated = rows.filter(cand => cand.rating != null);
  const pass = rated.filter(cand => cand.rating > 0).length;
  return (
    <div className="fill">
      <div className="head-row">
        <span className="n num">{rated.length}<small> / {rows.length}</small></span>
        <span className="a">most active moments rated</span>
        <span className="b">6 per contact sheet · {sheet.frames} frames each</span>
        <div className="r"><span className="chip good num"><Check size={11} />{pass} pass</span><span className="chip bad num"><X size={11} />{rated.length - pass} drop</span></div>
      </div>
      <div className="scroll">
        <div className="rows">
          {rows.map((cand, i) => {
            const done = cand.rating != null;
            return (
              <div key={cand.k} className={`srow ${done && cand.rating === 0 ? 'out' : ''}`} style={{ animationDelay: `${Math.min(i * 14, 500)}ms` }}>
                <Sprite className="strip4" url={url} cols={1} count={sheet.index.length} at={i} />
                <div className="who num"><b>#{cand.k}</b> · person {cand.track}<br />{fmtClock(cand.start_s)} · {cand.seconds.toFixed(1)} s</div>
                <div style={{ minWidth: 0 }}>
                  {done ? <div className="what" title={cand.what}>{cand.what}</div>
                    : <div className="what wait">{stage.status === 'running' ? <><span className="spin sm" />Claude is looking…</> : 'not rated'}</div>}
                  <div className="rate">{[1, 2, 3].map(r => <i key={r} className={done && cand.rating >= r ? 'on' : ''} />)}</div>
                </div>
                <div>{done && (cand.rating > 0 ? <span className="chip good"><Check size={11} />Pass</span> : <span className="chip bad"><X size={11} />Drop</span>)}</div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ 6. cut + upscale
function ClipTiles({ run, hero, onPick, mode }) {
  const sheet = run.kit.cands, sheetUrl = run.url(sheet && sheet.file);
  return (
    <div className="tiles">
      {run.clips.map(clip => {
        const poster = run.url(clip.files.poster);
        const at = run.candAt[momentOf(run, clip)];
        const working = run.active && (mode === 'label' ? clip.status === 'labeling' : clip.status === 'upscaling');
        const good = mode === 'label' ? clip.status === 'accepted' : UPSCALED.includes(clip.status);
        const bad = mode === 'label' ? clip.status === 'rejected' || clip.status === 'error' : false;
        return (
          <button key={clip.id} className={`tile ${hero && clip.id === hero.id ? 'focus' : ''}`} onClick={() => onPick(clip.id)} title={`${clip.id} · ${clip.status}`}>
            {poster ? <img src={poster} alt="" />
              : sheetUrl && at != null ? <Sprite className="dim" url={sheetUrl} cols={sheet.cols} count={sheet.index.length} at={at} /> : null}
            {working && mode !== 'label' && <i className="pb" style={{ width: `${(clip.progress || 0) * 100}%` }} />}
            <span className="st">
              {good ? <span className="okc sm"><Check size={8} strokeWidth={3.2} /></span>
                : bad ? <span className="badc sm"><X size={8} strokeWidth={3.2} /></span>
                  : working ? <span className="spin sm w" style={{ width: 9, height: 9 }} /> : null}
            </span>
            <span className="lbl num">{clip.id}</span>
          </button>
        );
      })}
    </div>
  );
}

// A 2.4× look at the busy hand: the pixels of the source crop next to the upscaler's output.
function Loupe({ a, b, clip, hands }) {
  const first = useRef(null), second = useRef(null);
  const centre = useMemo(() => {
    const track = hands && (hands[(clip.hands || [])[0]] || hands.right || hands.left);
    if (!track || !track.x.length) return [0.5, 0.5];
    // only the frames in which the hand is located: elsewhere its position is a stand-in
    const at = track.x.map((_, i) => i).filter(i => !track.seen || track.seen[i]);
    if (!at.length) return [0.5, 0.5];
    const mean = values => at.reduce((sum, i) => sum + values[i], 0) / at.length;
    return [mean(track.x), mean(track.y)];
  }, [hands, clip]);
  useEffect(() => {
    let frame;
    const draw = () => {
      for (const [video, canvas] of [[a.current, first.current], [b.current, second.current]]) {
        if (!video || !canvas || video.readyState < 2) continue;
        const vw = video.videoWidth, vh = video.videoHeight;
        const rw = 0.26, rh = (rw * 0.75 * vw) / vh;      // a 4:3 window, a quarter of the clip wide
        if (canvas.width !== 294) { canvas.width = 294; canvas.height = 220; }
        canvas.getContext('2d').drawImage(video, clamp(centre[0] - rw / 2, 0, 1 - rw) * vw, clamp(centre[1] - rh / 2, 0, 1 - rh) * vh,
          rw * vw, rh * vh, 0, 0, canvas.width, canvas.height);
      }
      frame = requestAnimationFrame(draw);
    };
    frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, [a, b, centre]);
  return (
    <div className="loupe-pair">
      <div className="lp"><canvas ref={first} /><span>Source</span></div>
      <div className="lp"><canvas ref={second} /><span>4× upscaled</span></div>
    </div>
  );
}

// The source crop and the upscaled clip in one frame, with a divider to drag.
function Compare({ run, clip, style }) {
  const raw = useRef(null), upscaled = useRef(null), frame = useRef(null);
  const [split, setSplit] = useState(0.5);
  useInStep(raw, [upscaled]);
  const hands = useJson(run.url(clip.files.hands));
  const move = event => {
    const box = frame.current.getBoundingClientRect();
    setSplit(clamp((event.clientX - box.left) / box.width, 0.03, 0.97));
  };
  const native = clip.native_size || [clip.crop[2], clip.crop[3]];
  return (
    <div ref={frame} className="vid cmp" style={style} onPointerDown={event => { event.currentTarget.setPointerCapture(event.pointerId); move(event); }}
      onPointerMove={event => { if (event.buttons) move(event); }}>
      <Video videoRef={raw} src={run.url(clip.files.raw)} />
      <div style={{ position: 'absolute', inset: 0, clipPath: `inset(0 ${100 - split * 100}% 0 0)` }}>
        <Video videoRef={upscaled} src={run.url(clip.files.upscaled)} />
      </div>
      <div className="cmp-line" style={{ left: `calc(${split * 100}% - 1px)` }} />
      <div className="cmp-knob" style={{ left: `${split * 100}%` }}>⇆</div>
      <div className="cmp-tag" style={{ right: 10 }}>Source crop<small className="num">{native[0]}×{native[1]}</small></div>
      <div className="cmp-tag" style={{ left: 10 }}>4× Real-ESRGAN<small className="num">{clip.size ? `${clip.size[0]}×${clip.size[1]}` : ''}</small></div>
      <Loupe a={raw} b={upscaled} clip={clip} hands={hands} />
      {clip.upscale_s != null && <div className="vchip" style={{ right: 12, bottom: 12 }}>
        <Check size={12} style={{ color: 'var(--good-text)' }} />{clip.id} upscaled in {Math.round(clip.upscale_s)} s · temporal blend on static areas
      </div>}
    </div>
  );
}

function UpscaleLayer({ run, hero, onPick }) {
  const stage = run.by.upscale || {};
  if (!hero) return <Waiting spin={stage.status === 'running'}>{stage.status === 'running' ? 'Taking a second look at the chosen moments: locating the hands…' : 'No clips were cut in this run.'}</Waiting>;
  const raw = run.url(hero.files.raw), upscaled = run.url(hero.files.upscaled);
  const waiting = run.clips.filter(clip => clip.status === 'cut').length;
  const box = clipBox(run, hero);
  return (
    <div className="stack">
      {raw && upscaled ? <Compare key={hero.id} run={run} clip={hero} style={box} /> : (
        <div className="vid" style={box}>
          {raw ? <Video key={hero.id} src={raw} />
            : <div className="ph"><div className="waiting">{run.active && <span className="spin" />}{run.active
              ? `Cutting ${plural(run.clips.length, 'clip')} out of the footage, faces blurred${stage.progress ? ` · ${Math.round(stage.progress * 100)}%` : ''}…` : 'This clip was not cut.'}</div></div>}
          {raw && <div className="cmp-tag" style={{ right: 10 }}>Source crop<small className="num">{hero.crop[2]}×{hero.crop[3]}</small></div>}
          {run.active && hero.status === 'upscaling' && <div className="vchip num" style={{ left: '50%', bottom: 12, transform: 'translateX(-50%)' }}>
            <span className="spin sm w" />Upscaling {hero.id} · {Math.round((hero.progress || 0) * 100)}% · frame {Math.round((hero.progress || 0) * (hero.frames[1] - hero.frames[0]))} of {hero.frames[1] - hero.frames[0]}
          </div>}
          {run.active && hero.status === 'cut' && <div className="vchip" style={{ left: '50%', bottom: 12, transform: 'translateX(-50%)' }}>
            <Clock size={12} />{hero.id} cut from blurred frames · {waiting > 1 ? `${plural(waiting - 1, 'clip')} ahead of it` : 'next to be upscaled'}
          </div>}
        </div>
      )}
      <ClipTiles run={run} hero={hero} onPick={onPick} mode="upscale" />
    </div>
  );
}

// ------------------------------------------------------------------ 7. label
// The hands to draw on a clip at `time`. A hand is drawn only in the frames where the engine located it.
// A run whose hands were located by the whole-body model has a box of its own size for every frame, and
// both hands are shown, the busy ones stronger. An older run has one box size and only its busy hands are
// worth showing (the others are a guess from the wrist); in the frames where that run's own tracking did
// not see the wrist the box used to be drawn anyway, on a guess, and is now left out there too.
export function handBoxes(hands, time, busyHands) {
  if (!hands || !hands.frames) return [];
  // the frame that is on screen at `time`: the one whose turn has come, not the nearest one (rounding ran
  // up to half a frame ahead of the picture, which shows on a fast hand in 15 fps footage)
  const frame = Math.min(hands.frames - 1, Math.max(0, Math.floor(time * hands.fps + 1e-3)));
  const busy = new Set(busyHands || hands.busy || []);
  const out = [];
  for (const side of ['left', 'right']) {
    const track = hands[side];
    if (!track || !Array.isArray(track.x) || track.x[frame] == null) continue;
    if (Array.isArray(track.seen) && !track.seen[frame]) continue;
    const located = Array.isArray(track.w) && Array.isArray(track.h);
    if (!located && !busy.has(side)) continue;
    const size = hands.box_size || [0.1, 0.1];
    out.push({ side, busy: busy.has(side), x: track.x[frame], y: track.y[frame],
      w: located ? track.w[frame] : size[0], h: located ? track.h[frame] : size[1] });
  }
  // two hands on the same spot: the second label goes under its box, so both can be read
  if (out.length === 2 && Math.abs(out[0].x - out[1].x) < (out[0].w + out[1].w) / 2 + 0.06
    && Math.abs(out[0].y - out[1].y) < (out[0].h + out[1].h) / 2 + 0.05) out[1].below = true;
  return out;
}

// A clip with what its label card says drawn on it: the busy hand, and the action at each moment.
// `clock` lets the cards beside the clip follow it, and jump to a moment in it.
export function ClipPlayer({ run, clip, style, clock, labels }) {
  const video = useRef(null);
  const time = useVideoTime(video);
  const hands = useJson(run.url(clip.files.hands));
  const src = run.url(clip.files.upscaled);
  useEffect(() => {
    if (!clock) return undefined;
    clock.seek = seconds => { if (video.current) { video.current.currentTime = seconds; video.current.play().catch(() => {}); } };
    return () => { clock.seek = null; };
  }, [clock]);
  useEffect(() => { if (clock) clock.set(time); }, [clock, time]);
  const boxes = handBoxes(hands, time, clip.hands);
  const segments = labels || (clip.label ? clip.label.labels : []);
  const length = clip.seconds;
  const [w, h] = clip.size || [clip.crop[2], clip.crop[3]];
  return (
    <div className="player" style={style ? { width: style.width } : undefined}>
      <div className="vid" style={{ aspectRatio: `${w} / ${h}` }}>
        {src ? <Video key={src} videoRef={video} src={src} /> : <div className="ph"><div className="waiting"><span className="spin" />Upscaling this clip…</div></div>}
        <svg className="ov" viewBox="0 0 1000 1000" preserveAspectRatio="none">
          {boxes.map(box => <rect key={box.side} x={(box.x - box.w / 2) * 1000} y={(box.y - box.h / 2) * 1000} width={box.w * 1000} height={box.h * 1000} rx="10" fill="none"
            stroke={box.side === 'right' ? 'var(--orange-hi)' : 'var(--blue-hi)'} strokeWidth={box.busy ? 2 : 1.3} strokeDasharray={box.busy ? 'none' : '5 4'}
            opacity={box.busy ? 1 : 0.8} vectorEffect="non-scaling-stroke" />)}
        </svg>
        {boxes.map(box => (
          <div key={box.side} className="vchip" style={{ left: `${(box.x - box.w / 2) * 100}%`,
            top: box.below ? `calc(${(box.y + box.h / 2) * 100}% + 6px)` : `calc(${(box.y - box.h / 2) * 100}% - 26px)`, height: 20, fontSize: 10.5,
            opacity: box.busy ? 1 : 0.75, background: box.side === 'right' ? 'rgba(217,89,38,.92)' : 'rgba(42,110,200,.92)' }}>
            {box.side === 'right' ? 'R' : 'L'} hand
          </div>
        ))}
        <div className="vchip num" style={{ right: 10, bottom: 10 }}>{clip.id} · {time.toFixed(1)} / {length.toFixed(1)} s</div>
      </div>
      <div className="segs">
        {segments.map((segment, i) => (
          <button key={i} className={`seg ${time >= segment.start && time < segment.end ? 'on' : ''}`} title={segment.action}
            style={{ left: `calc(${(segment.start / length) * 100}% + 1px)`, width: `calc(${((segment.end - segment.start) / length) * 100}% - 2px)` }}
            onClick={() => { if (video.current) video.current.currentTime = segment.start; }}>
            {verb(segment.verb)}
          </button>
        ))}
        <div className="playhead" style={{ left: `${clamp(time / length) * 100}%` }} />
      </div>
    </div>
  );
}

function LabelLayer({ run, hero, onPick, clock }) {
  const stage = run.by.label || {};
  if (stage.status === 'skipped') return <Waiting>Labeling was turned off for this run.</Waiting>;
  if (!hero) return <Waiting spin={run.active}>{run.active ? 'Waiting for the first upscaled clip…' : 'No clips were labeled in this run.'}</Waiting>;
  return (
    <div className="stack">
      <ClipPlayer key={hero.id} run={run} clip={hero} style={clipBox(run, hero, 34)} clock={clock} />
      <ClipTiles run={run} hero={hero} onPick={onPick} mode="label" />
    </div>
  );
}

// The action rows of a label card. They light up with the clip beside them, and a click jumps to that moment.
export function Actions({ labels, clock }) {
  const time = useClock(clock);
  return (
    <div className="acts">
      {labels.map((segment, i) => (
        <button key={i} className={`act ${time >= segment.start && time < segment.end ? 'on' : ''}`} onClick={() => clock.seek && clock.seek(segment.start)}>
          <span className="tm num">{segment.start.toFixed(1)}–{segment.end.toFixed(1)} s</span>
          <span className="vb">{verb(segment.verb)}</span>
          <span className="ds" title={segment.action}>{segment.action}</span>
          <span className="hd">{segment.hand}</span>
        </button>
      ))}
    </div>
  );
}

function LabelCard({ run, clip, fresh, clock }) {
  const label = clip && clip.label;
  const labeled = run.clips.filter(item => LABELED.includes(item.status)).length;
  const [caption, typing] = useTyped(label ? label.caption : '', fresh && !!label);
  const model = labelerName(run.engine.labeler);
  return (
    <div className="card lcard">
      <div className="stage-card" style={{ padding: 0 }}>
        <div className="h">
          <div className="ico"><Tags size={16} /></div>
          <div style={{ minWidth: 0 }}><div className="t">Label card{clip ? ` · ${clip.id}` : ''}</div><div className="s">{model} · frames of the clip and its surroundings</div></div>
          <span className="chip blue num">{labeled}/{run.clips.length}</span>
        </div>
      </div>
      {!clip ? <div className="cap" style={{ color: 'var(--muted)' }}>No clip to label yet.</div>
        : !label ? (
          <div className="cap" style={{ color: 'var(--muted)', display: 'flex', gap: 8, alignItems: 'center' }}>
            {run.active && clip.status === 'labeling' && <span className="spin sm" />}
            {clip.status === 'error' ? clip.reject_reason || 'Labeling failed for this clip.' : !run.active ? 'The run ended before this clip was labeled.'
              : clip.status === 'labeling' ? 'Claude is watching the clip…' : 'Waiting for its turn with the labeler.'}
          </div>
        ) : (
          <>
            <div className="cap">{caption}{typing && <span className="caret" />}</div>
            <div className="chips">{label.objects.map((object, i) => <span key={object} className={`chip ${object === label.main_object || (!label.main_object && i === 0) ? 'orange' : ''}`}>{object}</span>)}</div>
            <Actions labels={label.labels} clock={clock} />
            <dl className="kv">
              <dt>Setting</dt><dd title={label.scene.setting}>{label.scene.setting}</dd>
              <dt>Lighting</dt><dd title={label.scene.lighting}>{label.scene.lighting}</dd>
              <dt>Quality</dt><dd className="q"><span className="qbar">{Array.from({ length: 10 }, (_, i) => <i key={i} className={i < label.quality ? 'on' : ''} />)}</span>{label.quality}/10</dd>
            </dl>
            <div className={`verdict ${clip.status === 'rejected' ? 'no' : ''}`}>
              {clip.status === 'rejected' ? <X size={16} /> : <Check size={16} />}
              <div><div className="t">{clip.status === 'rejected' ? 'Rejected' : 'Accepted'}</div>
                <div className="s">{clip.status === 'rejected' ? clip.reject_reason || 'not a hand-object interaction' : 'a hand visibly handles an object'}</div></div>
              <div className="r num">{clip.label_usd ? fmtUSD(clip.label_usd) : 'from cache'}</div>
            </div>
          </>
        )}
    </div>
  );
}

// ------------------------------------------------------------------ 8. multiply
// What a clip's object is swapped for: from the variant once it is made, from the engine's plan before that.
// Nothing is promised before the plan: the engine may find the clip not fit for a swap at all.
const swapOf = clip => {
  const variant = (clip.variants || [])[0];
  if (variant && variant.edit) return variant.edit.split(' -> ');
  const label = clip.label || {};
  if (clip.swap && Array.isArray(clip.swap.to) && clip.swap.to.length) return [clip.swap.from || label.main_object || 'object', clip.swap.to[0]];
  return [label.main_object || 'object', clip.multiply === 'skipped' ? 'not swapped' : '…'];
};

function MultiplyLayer({ run, hero, onPick }) {
  const stage = run.by.multiply || {};
  const all = run.clips.filter(clip => clip.multiply || (clip.variants || []).length);
  // clips the engine looked at and left alone: not fit for a swap, with its reason
  const passedOver = all.filter(clip => clip.multiply === 'skipped' && !(clip.variants || []).length);
  const rows = all.filter(clip => !passedOver.includes(clip));
  if (!all.length) {
    return <Waiting spin={stage.status === 'running'}>{stage.status === 'running' ? 'Choosing the clips to make variants of…'
      : run.options.multiply ? 'No clip of this run was fit for an object swap: a swap needs a loose item that a hand clearly holds.' : 'Object-swap variants were not asked for in this job. Turn them on when you start a job on a worker with a fal key.'}</Waiting>;
  }
  return (
    <div className="vrows">
      {passedOver.length > 0 && (
        <div className="vrow" style={{ cursor: 'default' }}>
          <div className="h"><b>Not swapped</b><span className="r"><span className="chip outline num">{passedOver.length} looked at</span></span></div>
          {passedOver.map(clip => (
            <div key={clip.id} className="h" style={{ cursor: 'pointer' }} onClick={() => onPick(clip.id)} title={clip.multiply_note || ''}>
              <span className="mono" style={{ color: hero && hero.id === clip.id ? 'var(--text)' : 'var(--muted)', fontSize: 11 }}>{clip.id}</span>
              <b>{(clip.label && clip.label.main_object) || 'object'}</b>
              <span style={{ color: 'var(--muted)', minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{clip.multiply_note || 'not fit for a swap'}</span>
            </div>
          ))}
        </div>
      )}
      {rows.map(clip => {
        const variant = (clip.variants || [])[0];
        const [from, to] = swapOf(clip);
        const made = run.url(variant && variant.files.video);
        const original = run.url(variant && variant.files.original) || run.url(clip.files.upscaled);
        return (
          <div key={clip.id} className="vrow" onClick={() => onPick(clip.id)}>
            <div className="h">
              <span className="mono" style={{ color: hero && hero.id === clip.id ? 'var(--text)' : 'var(--muted)', fontSize: 11 }}>{clip.id}</span>
              <b>{from}</b><span className="arrow">→</span><b className="to">{to}</b>
              <span className="r">
                {variant ? (variant.passed ? <span className="chip good"><Check size={11} />Passed · {variant.verify ? variant.verify.score : '?'}/10</span>
                  : <span className="chip bad"><X size={11} />Rejected · {variant.verify ? variant.verify.score : '?'}/10</span>)
                  : !run.active ? <span className="chip outline">Not made</span>
                    : clip.multiply === 'running' ? <span className="chip outline"><span className="spin sm" />Generating, then checking</span>
                    : clip.multiply === 'planning' ? <span className="chip outline"><span className="spin sm" />Checking that a swap makes sense</span>
                    : clip.multiply === 'stopped' ? <span className="chip bad">Stopped at the fal budget</span>
                      : <span className="chip outline">Queued</span>}
              </span>
            </div>
            <div className="two">
              <div className="vid">{original && <Video src={original} />}<div className="cmp-tag" style={{ left: 8, top: 8 }}>Original</div></div>
              <div className="vid">
                {made ? <Video src={made} /> : !run.active ? <div className="ph"><div><span>No variant was made before the run ended.</span></div></div> : (
                  <div className="ph"><div>
                    <Sparkles size={18} />
                    <span>Video edit model, through fal</span>
                    <small>{clip.multiply === 'planning' ? 'First: is this a loose item the hand holds, and what of the same kind could stand in for it?'
                      : `“Replace only the ${from} in the hand with ${/^[aeiou]/i.test(to) ? 'an' : 'a'} ${to}. Keep the hand, the timing and the motion; keep blurred faces blurred.”`}</small>
                  </div></div>
                )}
                {made && <div className="cmp-tag y" style={{ left: 8, top: 8 }}>Variant · {to}</div>}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function CheckCard({ run, clip }) {
  // clips that are being swapped or were: one the engine looked at and passed over is not waiting for a check
  const rows = run.clips.filter(item => (item.multiply && item.multiply !== 'skipped') || (item.variants || []).length);
  const made = rows.filter(item => (item.variants || []).length);
  const variant = clip && (clip.variants || [])[0];
  const verify = (variant && variant.verify) || {};
  const [, to] = clip ? swapOf(clip) : ['', ''];
  // where the hands are in the variant against the original, measured by the hand model (newer runs)
  const paths = variant && variant.hand_paths && variant.hand_paths.variant;
  const before = variant && variant.hand_paths && variant.hand_paths.original;
  // the hands that can be followed in the original; one of them missing from the variant counts as moved
  const measured = ['left', 'right'].filter(side => paths && before && before[side] && before[side].same >= 0.5);
  const kept = measured.every(side => paths[side] && paths[side].same >= before[side].same - 0.25);
  const items = [
    ['Edit applied', verify.edit_applied, `${to} in every frame`],
    ['Motion preserved', verify.motion_preserved && kept, measured.length
      ? (kept ? 'hands where the original has them, measured' : 'the hands are not where they were, measured') : 'same hand path and timing',
      measured.map(side => `${side} hand in place in ${paths[side] ? Math.round(100 * paths[side].same) : 0}% of the frames (${Math.round(100 * before[side].same)}% when the untouched original is measured the same way)`).join('; ')],
    ['Hands intact', verify.hands_intact, 'no extra or missing fingers'],
    ['Faces still blurred', verify.faces_blurred, 'privacy holds after the edit'],
    ...(verify.only_edit_changed != null ? [['Nothing else changed', verify.only_edit_changed, 'clothing, gloves, background as they were']] : []),
    ...(verify.plausible != null ? [['Makes sense', verify.plausible, 'belongs in the scene, held as a person would']] : []),
  ];
  if (clip && clip.multiply === 'skipped' && !variant) {
    return (
      <div className="card stage-card">
        <div className="h">
          <div className="ico y"><Sparkles size={16} /></div>
          <div style={{ minWidth: 0 }}><div className="t">Not swapped · {clip.id}</div>
            <div className="s">The engine looked at this clip before spending anything on it</div></div>
        </div>
        <div className="verdict no"><X size={16} />
          <div><div className="t">Not fit for an object swap</div><div className="s">{clip.multiply_note || 'a swap needs a loose item that a hand clearly holds'}</div></div></div>
      </div>
    );
  }
  return (
    <div className="card stage-card">
      <div className="h">
        <div className="ico y"><Sparkles size={16} /></div>
        <div style={{ minWidth: 0 }}><div className="t">Variant check{clip ? ` · ${clip.id}` : ''}</div>
          <div className="s">{labelerName((variant && variant.verify_model) || run.engine.labeler)} compares the variant with the original</div></div>
        <span className="chip blue num">{made.length}/{rows.length}</span>
      </div>
      <div className="checks">
        {items.map(([name, ok, detail, more]) => (
          <div key={name} className="check" title={more || undefined}>
            <span className={`ci ${!variant ? 'wait' : ok ? 'ok' : 'no'}`}>
              {!variant ? (run.active && clip && clip.multiply === 'running' ? <span className="spin sm" style={{ width: 9, height: 9 }} /> : null) : ok ? <Check size={11} strokeWidth={3} /> : <X size={11} strokeWidth={3} />}
            </span>
            <span style={{ color: variant ? 'var(--text)' : 'var(--muted)' }}>{name}</span>
            <span className="r">{detail}</span>
          </div>
        ))}
      </div>
      {variant && (
        <>
          <div className={`verdict ${variant.passed ? '' : 'no'}`}>
            {variant.passed ? <Check size={16} /> : <X size={16} />}
            <div><div className="t">{variant.passed ? 'Passed' : 'Rejected'} · {verify.score}/10</div><div className="s">{variant.passed ? 'labels carried over · Studio tier' : 'left out of the dataset'}</div></div>
            <div className="r num">fal {fmtUSD(variant.cost_usd)}</div>
          </div>
          {verify.notes && <div className="quote">“{verify.notes}”</div>}
        </>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ right column
export function SideColumn({ run, focus, hero, fresh, clock }) {
  const card = focus === 'label' ? <LabelCard run={run} clip={hero} fresh={fresh} clock={clock} />
    : focus === 'multiply' ? <CheckCard run={run} clip={hero} />
      : <StageCard run={run} focus={focus} />;
  return (
    <div className="side-col">
      {card}
      <LogCard run={run} />
    </div>
  );
}

function StageCard({ run, focus }) {
  const stage = STAGE[focus];
  if (!stage) return <div className="card stage-card"><div className="h"><div className="ico"><Loader size={16} /></div><div><div className="t">Starting</div><div className="s">The engine has not reported a stage yet</div></div></div></div>;
  const Icon = stage.icon;
  const st = run.by[focus] || {};
  const stats = st.stats || {};
  const done = st.status === 'done';
  const p = done ? 1 : clamp(st.progress || 0);
  const { total, now, state, people, candidates, clips } = run;
  const motion = state.motion;
  const device = deviceName(run.device);
  const dash = '—';
  let metrics = [], tech = [];
  if (focus === 'motion') {
    const live = now && now.stage === 'motion' ? now : null;
    const scanned = done ? total : live ? live.t : p * total;
    metrics = [['Scanned', fmtClock(scanned), `/ ${fmtClock(total)}`], ['With motion', motion ? Math.round(motion.kept_seconds) : dash, motion ? 's' : ''],
      ['Idle, dropped', motion ? Math.round(motion.seconds - motion.kept_seconds) : dash, motion ? 's' : ''],
      ['Speed', done && st.seconds ? `${Math.round(total / st.seconds)}×` : dash, done ? 'real time' : '']];
    tech = ['frame differencing', '480 px · 10 samples/s', 'CPU'];
  } else if (focus === 'people') {
    const live = now && now.stage === 'people' ? now : null;
    const kept = motion ? motion.kept_seconds : total;
    // a track is one appearance of a person: someone who is hidden for a moment or walks out and back counts again
    metrics = [['Person tracks', done ? people.length : live ? live.seen : 0, ''], ['Frames through model', fmtInt(done ? stats.samples : live ? live.samples : 0), ''],
      ['Speed', stats.cached ? 'cached' : done && stats.model_seconds ? `${(kept / stats.model_seconds).toFixed(1)}×` : dash, done && !stats.cached ? 'real time' : ''],
      ['Dropped', done && stats.dropped_short != null ? `${stats.dropped_short} + ${stats.dropped_small}` : dash, done ? 'brief + small' : '']];
    tech = [(run.engine.pose_model || 'yolo26s-pose.pt').replace('.pt', ''), 'ByteTrack', device, stats.imgsz ? `${stats.imgsz} px input` : null];
  } else if (focus === 'activity') {
    const busySeconds = stats.active_s != null ? stats.active_s : people.reduce((sum, person) => sum + person.active_s, 0);
    const personTime = stats.person_s != null ? stats.person_s : people.reduce((sum, person) => sum + person.seconds, 0);
    metrics = [['Hands busy', fmtInt(busySeconds), 's'], ['Person-time', fmtInt(personTime), 's'], ['Share busy', personTime ? Math.round((100 * busySeconds) / personTime) : dash, '%'],
      ['Idle or walking', fmtInt(personTime - busySeconds), 's dropped']];
    tech = ['wrist speed ÷ torso length', 'stricter while walking', 'no API'];
  } else if (focus === 'zoom') {
    metrics = [['Moments', fmtInt(candidates.length), ''], ['Clip time', fmtInt(candidates.reduce((sum, cand) => sum + cand.seconds, 0)), 's'], ['Length', '2–8', 's'], ['Frame', '16:9', 'crop']];
    tech = ['hand path + 0.6 torso margin', 'raised or reaching hands', 'no API'];
  } else if (focus === 'screen') {
    const all = run.kit.screen ? run.kit.screen.index.length : stats.screened || 0;
    const rated = candidates.filter(cand => cand.rating != null);
    metrics = [['Calls', done ? stats.calls != null ? stats.calls : Math.ceil(all / 6) : Math.ceil(rated.length / 6), `/ ${Math.ceil(all / 6)}`], ['Rated', rated.length, `/ ${all}`],
      ['Passed', rated.filter(cand => cand.rating > 0).length, ''], ['Claude spend', stats.cached ? 'cached' : stats.usd != null ? fmtUSD(stats.usd) : dash, '']];
    tech = [run.engine.labeler, 'effort: low', 'contact sheets'];
  } else if (focus === 'upscale') {
    const finished = clips.filter(clip => UPSCALED.includes(clip.status));
    const timed = finished.filter(clip => clip.upscale_s != null);
    metrics = [['Upscaled', finished.length, `/ ${clips.length}`], ['Per clip', timed.length ? (timed.reduce((sum, clip) => sum + clip.upscale_s, 0) / timed.length).toFixed(1) : dash, timed.length ? 's avg' : ''],
      ['Output', `${run.options.max_height || 1080}p`, 'max'], ['Faces', 'blurred', 'before cut']];
    tech = [/x4plus/.test(run.engine.upscaler || '') ? 'RealESRGAN_x4plus' : 'realesr-general-x4v3', 'temporal blend',
      stats.hand_model ? `hands: ${stats.hand_model}` : null, device];
  }
  return (
    <div className="card stage-card">
      <div className="h">
        <div className="ico"><Icon size={16} /></div>
        <div style={{ minWidth: 0 }}><div className="t">{stage.title}</div><div className="s">{stage.blurb}</div></div>
        <span className="num">
          {done ? <span className="chip good"><Check size={11} />{fmtDur(st.seconds)}</span>
            : st.status === 'running' ? <span className="chip blue">{Math.round(p * 100)}%</span>
              : st.status === 'error' ? <span className="chip bad">Failed</span>
                : <span className="chip outline">{st.status === 'skipped' ? 'Skipped' : st.status === 'stopped' ? `Stopped at ${Math.round(p * 100)}%` : 'Queued'}</span>}
        </span>
      </div>
      <div className="metrics">
        {metrics.map(([name, value, unit]) => <div key={name}><span className="k">{name}</span><span className="v">{value}<small>{unit}</small></span></div>)}
      </div>
      {st.more && <div className="quote">{st.more.charAt(0).toUpperCase() + st.more.slice(1)}.</div>}
      <div className="tech">{tech.filter(Boolean).map(item => <span key={item} className="chip outline mono">{item}</span>)}</div>
    </div>
  );
}

const tidy = message => message.replace(/\(([^()]*)\)$/, '').replace(/^run (\S+): /, 'job $1 · ').replace(/\n[\s\S]*$/, '').slice(0, 180);

function LogCard({ run }) {
  const lines = run.state.log || [];
  const elapsed = run.state.elapsed || 0;
  const text = !lines.length && run.job.log ? run.job.log.split('\n') : [];      // a worker that sends no state: its last lines
  return (
    <div className="card log">
      <div className="card-h"><h3>Engine log</h3><span className="sub mono" style={{ fontSize: 10.5 }}>{run.job.run}</span>
        <div className="r"><span className="chip outline num">{plural(lines.length || text.length, 'event')}</span></div></div>
      <div className="log-b">
        {[...lines].reverse().slice(0, 80).map(line => (
          <div key={`${line.t}-${line.msg}`} className={`log-l ${run.active && elapsed - line.t < 6 ? 'new' : ''}`}>
            <span className="ts num">{fmtClock(line.t)}</span>
            <span className="msg" title={line.msg}>
              {/accepted|done|kept|passed/.test(line.msg) && !/FAILED|failed/.test(line.msg) && <span className="ok">● </span>}
              {/FAILED|failed|rejected/.test(line.msg) && <span className="no">● </span>}
              {tidy(line.msg)}
            </span>
          </div>
        ))}
        {[...text].reverse().map((line, i) => <div key={i} className="log-l"><span className="ts" /><span className="msg" title={line}>{line}</span></div>)}
      </div>
    </div>
  );
}

const LAYERS = {
  motion: MotionLayer,
  people: props => <TrackLayer {...props} phase="people" />,
  activity: props => <TrackLayer {...props} phase="activity" />,
  zoom: ZoomLayer,
  screen: ScreenLayer,
  upscale: UpscaleLayer,
  label: LabelLayer,
  multiply: MultiplyLayer,
};

// The short line under each stage's name in the row of stages.
export function stageLine(run, key) {
  const st = run.by[key] || {};
  const stats = st.stats || {};
  if (!st.status || st.status === 'pending') return 'Queued';
  if (st.status === 'skipped') return key === 'multiply' ? 'Off' : 'Skipped';
  if (st.status === 'error') return 'Failed';
  if (st.status === 'stopped') return `Stopped at ${Math.round((st.progress || 0) * 100)}%`;
  if (!run.reported) return st.detail || (st.status === 'done' ? 'Done' : 'Working');      // only a worker's short report to go on
  const done = st.status === 'done', time = fmtDur(st.seconds);
  const { clips, candidates, people, total, now } = run;
  switch (key) {
    case 'motion':
      return done ? <><b>{Math.round(run.state.motion ? run.state.motion.kept_seconds : total)} s</b> kept · {time}</> : <>scanning {fmtClock((st.progress || 0) * total)}</>;
    case 'people':
      return <><b>{done ? people.length : now && now.stage === 'people' ? now.seen : 0}</b> tracks · {done ? time : `${Math.round((st.progress || 0) * 100)}%`}</>;
    case 'activity': {
      const personTime = stats.person_s != null ? stats.person_s : people.reduce((sum, person) => sum + person.seconds, 0);
      const busyTime = stats.active_s != null ? stats.active_s : people.reduce((sum, person) => sum + person.active_s, 0);
      return done && personTime ? <><b>{Math.round((100 * busyTime) / personTime)}%</b> busy · {time}</> : <>measuring wrists</>;
    }
    case 'zoom':
      return done ? <><b>{candidates.length}</b> moments · {time}</> : <>finding moments</>;
    case 'screen': {
      const rated = candidates.filter(cand => cand.rating != null);
      const all = run.kit.screen ? run.kit.screen.index.length : stats.screened || rated.length;
      return done ? <><b>{rated.filter(cand => cand.rating > 0).length}</b>/{all} pass · {time}</> : <>{rated.length}/{all || '…'} rated</>;
    }
    case 'upscale': {
      const finished = clips.filter(clip => UPSCALED.includes(clip.status)).length;
      if (done) return <><b>{finished}</b> clips · {time}</>;
      return clips.length && clips.every(clip => clip.status === 'planned') ? <>cutting {clips.length} clips</> : <><b>{finished}</b>/{clips.length} upscaled</>;
    }
    case 'label': {
      const labeled = clips.filter(clip => LABELED.includes(clip.status)).length;
      return done ? <><b>{clips.filter(clip => clip.status === 'accepted').length}</b> accepted · {time}</> : <><b>{labeled}</b>/{clips.length} labeled</>;
    }
    case 'multiply': {
      const made = clips.reduce((sum, clip) => sum + (clip.variants || []).length, 0);
      const verified = clips.reduce((sum, clip) => sum + passedVariants(clip).length, 0);
      if (done && !made) return <>no clip fit for a swap</>;
      return done ? <><b>{verified}</b>/{made} verified · {time}</> : <>{made}/{clips.filter(clip => clip.multiply && clip.multiply !== 'skipped').length} checked</>;
    }
    default:
      return null;
  }
}
