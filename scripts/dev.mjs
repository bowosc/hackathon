import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';

const python = process.platform === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python';
if (!existsSync(python)) {
  console.error('Create .venv and install requirements.txt first. See readme.md.');
  process.exit(1);
}

const children = [
  spawn(python, ['-m', 'uvicorn', 'backend.main:app', '--reload', '--host', '127.0.0.1', '--port', process.env.API_PORT || '8000'], { stdio: 'inherit' }),
  spawn(process.execPath, ['node_modules/vite/bin/vite.js'], { stdio: 'inherit' }),
];
let stopping = false;
function stop(code = 0) {
  if (stopping) return;
  stopping = true;
  children.forEach(child => child.kill('SIGTERM'));
  process.exitCode = code;
}
children.forEach(child => {
  child.on('error', error => { console.error(error.message); stop(1); });
  child.on('exit', code => stop(code || 0));
});
process.on('SIGINT', () => stop());
process.on('SIGTERM', () => stop());
