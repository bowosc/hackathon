export async function getJSON(url, signal) {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`Request failed: ${response.status}`);
  return response.json();
}

export async function postJSON(url, body, signal) {
  const response = await fetch(url, {
    method: 'POST', signal,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Please check the request fields and try again.');
  return data;
}

export function download(data, filename) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
