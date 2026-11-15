// Talking to the admin API (backend/admin.py). Every call carries the admin token; pictures and videos
// are loaded by the browser itself, which shows a cookie the server gave this page for them.

import { useEffect, useRef, useState } from 'react';

const tokenKey = 'peelset:admin-token';
export const savedToken = () => localStorage.getItem(tokenKey) || '';
export const saveToken = token => (token ? localStorage.setItem(tokenKey, token) : localStorage.removeItem(tokenKey));

export async function call(token, path, options = {}) {
  const response = await fetch(`/api/admin${path}`, { ...options, headers: { Authorization: `Bearer ${token}`, ...options.headers } });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = typeof data.detail === 'string' ? data.detail : `Request failed: ${response.status}`;
    throw Object.assign(new Error(message), { status: response.status });
  }
  return data;
}

export const post = (token, path, body) => call(token, path, body === undefined ? { method: 'POST' }
  : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

export const videoTypes = ['.mp4', '.mov', '.avi', '.mkv', '.m4v', '.webm'];

// One piece of a file. `onSent` hears how much of the piece has left this machine.
function putPiece(token, path, piece, onSent) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open('PUT', `/api/admin${path}`);
    request.setRequestHeader('Authorization', `Bearer ${token}`);
    request.upload.onprogress = event => onSent(event.loaded);
    request.onload = () => {
      let data = {};
      try { data = JSON.parse(request.responseText); } catch { /* not JSON: the status says enough */ }
      if (request.status >= 200 && request.status < 300) resolve(data);
      else reject(Object.assign(new Error(typeof data.detail === 'string' ? data.detail : `Request failed: ${request.status}`), { status: request.status }));
    };
    request.onerror = () => reject(new Error('The connection dropped during the upload.'));
    request.send(piece);
  });
}

// Send a file in pieces. `onProgress` hears the bytes sent so far; the result is the footage entry, which the
// server makes after the last piece (it reads the video's length and size, and takes ten small frames of it).
export async function upload(token, file, limits, onProgress) {
  if (!videoTypes.some(type => file.name.toLowerCase().endsWith(type))) throw new Error(`Not a video file (${videoTypes.join(' ')}).`);
  if (file.size > limits.upload_bytes) throw new Error(`Larger than the ${Math.round(limits.upload_bytes / 2 ** 20)} MB limit.`);
  for (let offset = 0; offset < file.size;) {
    const piece = file.slice(offset, offset + limits.chunk_bytes);
    const query = new URLSearchParams({ name: file.name, total: file.size, offset });
    const start = offset;
    let reply, attempt = 0;
    while (!reply) {
      try { reply = await putPiece(token, `/footage?${query}`, piece, sent => onProgress(start + Math.min(sent, piece.size))); }
      catch (error) {      // a refusal is final; a dropped connection gets the same piece again
        if (error.status || ++attempt > 3) throw error;
        await new Promise(resolve => setTimeout(resolve, 1500 * attempt));
      }
    }
    offset += piece.size;
    onProgress(offset);
    if (reply.done) return reply.footage;
  }
  throw new Error('The upload ended before the server had the whole file.');
}

// Run `load` now and then every `ms`, one call at a time. A failed call is tried again at the next turn.
export function usePoll(load, ms) {
  const latest = useRef(load);
  latest.current = load;
  useEffect(() => {
    let live = true, timer;
    const turn = async () => {
      try { await latest.current(); } catch { /* the next turn tries again */ }
      if (live) timer = setTimeout(turn, ms);
    };
    turn();
    return () => { live = false; clearTimeout(timer); };
  }, [ms]);
}

// A JSON file of a job's run (tracks, hand positions), fetched once per address.
const fetched = new Map();
export function useJson(url) {
  const [, redraw] = useState(0);
  useEffect(() => {
    if (!url || fetched.has(url)) return;
    let live = true;
    fetched.set(url, null);
    fetch(url).then(response => (response.ok ? response.json() : null)).catch(() => null).then(data => {
      if (data) fetched.set(url, data); else fetched.delete(url);
      if (fetched.size > 60) fetched.delete(fetched.keys().next().value);
      if (live) redraw(n => n + 1);
    });
    return () => { live = false; };
  }, [url]);
  return (url && fetched.get(url)) || null;
}
