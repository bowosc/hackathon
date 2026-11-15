// Small, pure helpers shared by every screen of the console.

export const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
export const lerp = (a, b, k) => a + (b - a) * k;

export const fmtClock = s => {
  s = Math.max(0, s || 0);
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s - m * 60)).padStart(2, '0')}`;
};
export const fmtDur = s => (s == null ? '' : s < 10 ? `${s.toFixed(1)} s` : s < 60 ? `${Math.round(s)} s` : fmtClock(s));
export const fmtUSD = v => (v == null ? '–' : !v ? '$0.00' : v < 0.1 ? `$${v.toFixed(4)}` : `$${v.toFixed(2)}`);
export const fmtCents = v => `${(100 * (v || 0)).toFixed(v && v < 0.095 ? 1 : 0).replace(/\.0$/, '')}¢`;
export const fmtInt = n => Math.round(n || 0).toLocaleString('en-US');
export const fmtBytes = b => (b >= 2 ** 30 ? `${(b / 2 ** 30).toFixed(1)} GB` : `${(b / 1e6).toFixed(b < 1e7 ? 1 : 0)} MB`);
export const plural = (n, one, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

export const labelerName = m => (m || '').replace('claude-', 'Claude ').replace(/-(\d)-(\d)$/, ' $1.$2')
  .replace('haiku', 'Haiku').replace('opus', 'Opus').replace('sonnet', 'Sonnet') || '–';
export const deviceName = d => ({ mps: 'Apple GPU (MPS)', cuda: 'NVIDIA GPU (CUDA)', cpu: 'CPU', auto: 'its own GPU or CPU' }[d] || d || 'unknown device');
export const deviceShort = d => ({ mps: 'Apple GPU', cuda: 'NVIDIA GPU', cpu: 'CPU' }[d] || '');
// 'm1-pro · Apple GPU', or just the name when the device is not known.
export const withDevice = (name, d) => (deviceShort(d) ? `${name} · ${deviceShort(d)}` : name);

// 11/14 14:13 in the reader's own time zone.
export const fmtWhen = iso => {
  if (!iso) return '–';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '–';
  const two = n => String(n).padStart(2, '0');
  return `${two(d.getMonth() + 1)}/${two(d.getDate())} ${two(d.getHours())}:${two(d.getMinutes())}`;
};

// The stretch of footage a job covers, before its run has said so itself.
export const jobSeconds = (job, footage) => {
  if (job.source && job.source.seconds) return job.source.seconds;
  if (job.options.seconds) return job.options.seconds;
  const file = footage.find(item => item.name === job.footage);
  return file && file.seconds ? Math.max(0, file.seconds - (job.options.start || 0)) : null;
};
