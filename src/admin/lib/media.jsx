// Videos and stills in the console. Videos play on their own; whatever is drawn over them (people, hands,
// the place in a clip) reads the video's clock on every frame.

import { useEffect, useRef, useState, useSyncExternalStore } from 'react';

// The current time of a <video>, re-read on every animation frame while it changes.
export function useVideoTime(ref) {
  const [time, setTime] = useState(0);
  useEffect(() => {
    let frame;
    const tick = () => {
      if (ref.current) setTime(ref.current.currentTime);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [ref]);
  return time;
}

// Keep other videos of the same clip in step with one of them.
export function useInStep(lead, others) {
  useEffect(() => {
    let frame;
    const tick = () => {
      const a = lead.current;
      if (a) {
        for (const other of others) {
          const b = other.current;
          if (!b || b.readyState < 1) continue;
          if (Math.abs(b.currentTime - a.currentTime) > 0.12) b.currentTime = a.currentTime;
          if (a.paused !== b.paused) { if (a.paused) b.pause(); else b.play().catch(() => {}); }
        }
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  });
}

// A muted, looping video. `rate` speeds it up (the scan of a long recording plays several times faster).
export function Video({ videoRef, src, rate = 1, ...rest }) {
  const own = useRef(null);
  const ref = videoRef || own;
  useEffect(() => { if (ref.current) ref.current.playbackRate = rate; }, [ref, rate, src]);
  return <video ref={ref} src={src} muted loop autoPlay playsInline preload="auto" disablePictureInPicture {...rest} />;
}

// One still out of a sheet of same-sized stills (the engine tiles them into one picture).
export function Sprite({ url, cols, count, at, className = '', style }) {
  const rows = Math.ceil(count / cols), x = at % cols, y = Math.floor(at / cols);
  return <div className={`sprite ${className}`} style={{
    ...style, backgroundImage: `url(${url})`, backgroundSize: `${cols * 100}% ${rows * 100}%`,
    backgroundPosition: `${cols > 1 ? (100 * x) / (cols - 1) : 0}% ${rows > 1 ? (100 * y) / (rows - 1) : 0}%`,
  }} />;
}

// Text that types itself out once, when `fresh` (it arrived while the page was open).
export function useTyped(text, fresh, ms = 1300) {
  const [shown, setShown] = useState(fresh ? 0 : text.length);
  useEffect(() => {
    if (!fresh) { setShown(text.length); return; }
    let frame, start;
    const tick = now => {
      start = start || now;
      const k = Math.min(1, (now - start) / ms);
      setShown(Math.floor((1 - (1 - k) ** 2) * text.length));
      if (k < 1) frame = requestAnimationFrame(tick);
    };
    setShown(0);
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [text, fresh, ms]);
  return [text.slice(0, shown), shown < text.length];
}

// The id of the clip that most recently came to pass `test` while this page was open (null until one does).
export function useLatest(clips, test) {
  const seen = useRef(null), latest = useRef(null);
  const ids = clips.filter(test).map(clip => clip.id);
  if (seen.current === null) seen.current = new Set(ids);      // what was already there when the page opened
  else for (const id of ids) if (!seen.current.has(id)) { seen.current.add(id); latest.current = id; }
  return latest.current;
}

// The time of one clip, shared between its player and the cards beside it. The player sets it (and says how
// to jump to a moment); the cards read it.
export function makeClock() {
  let time = 0;
  const listeners = new Set();
  return {
    seek: null,
    get: () => time,
    set: value => { if (value !== time) { time = value; listeners.forEach(listener => listener()); } },
    subscribe: listener => { listeners.add(listener); return () => listeners.delete(listener); },
  };
}
export const useClock = clock => useSyncExternalStore(clock.subscribe, clock.get);
