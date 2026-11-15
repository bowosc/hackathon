// What the console knows about the engine's eight stages, and how to read a run's state.

import { Activity, Copy, Crop, Hand, ImageUpscale, ScanEye, Tags, Users } from 'lucide-react';

export const STAGES = [
  { key: 'motion', title: 'Motion filter', short: 'Motion', blurb: 'Drop the parts where nothing moves', icon: Activity },
  { key: 'people', title: 'Person tracking', short: 'Tracking', blurb: 'Everyone in view followed, one number per person, faces located and blurred', icon: Users },
  { key: 'activity', title: 'Activity filter', short: 'Activity', blurb: "Keep the time when a person's hands are busy", icon: Hand },
  { key: 'zoom', title: 'Hand zoom', short: 'Hand zoom', blurb: 'A 2–8 s, 16:9 crop around each burst of hand activity', icon: Crop },
  { key: 'screen', title: 'Quick screen', short: 'Screen', blurb: 'Claude rates contact sheets, weak moments are dropped', icon: ScanEye },
  { key: 'upscale', title: 'Cut + upscale', short: 'Upscale', blurb: 'Clips cut from blurred frames, 4× super-resolution', icon: ImageUpscale },
  { key: 'label', title: 'Label', short: 'Label', blurb: 'Claude writes the label card, rejects non-interactions', icon: Tags },
  { key: 'multiply', title: 'Multiply', short: 'Multiply', blurb: 'Object-swap variants, labels carried over, each checked', icon: Copy },
];
export const STAGE = Object.fromEntries(STAGES.map(stage => [stage.key, stage]));

export const busy = job => ['queued', 'running', 'stopping'].includes(job.status);
export const UPSCALED = ['upscaled', 'labeling', 'accepted', 'rejected', 'error'];
export const LABELED = ['accepted', 'rejected'];

// The stage a job page follows: the earliest one still running; between two stages, the one about to start;
// for a run that is over, the last one it reached.
export function followed(stages, status) {
  const failed = stages.find(stage => stage.status === 'error' || stage.status === 'stopped');
  if (failed) return failed.key;
  const running = stages.find(stage => stage.status === 'running');
  if (running) return running.key;
  if (status === 'running' || status === 'stopping') {
    const next = stages.find(stage => stage.status === 'pending');
    if (next) return next.key;
  }
  const reached = [...stages].reverse().find(stage => stage.status === 'done');
  return reached ? reached.key : stages.length ? stages[0].key : null;
}

const list = value => (Array.isArray(value) ? value.filter(item => item && typeof item === 'object') : []);
const record = value => (value && typeof value === 'object' && !Array.isArray(value) ? value : {});

// Everything the job page's parts read, from one answer of GET /api/admin/jobs/<id>.
// A state can come from a worker, so its lists and records are checked before anything is drawn from them.
export function readRun(detail) {
  const { job, files } = detail;
  const state = record(detail.state);
  const active = job.status === 'running' || job.status === 'stopping';
  // a run that was stopped, or died, leaves its last stage marked as running: nothing is running any more
  const stages = list(list(state.stages).length ? state.stages : job.stages)
    .map(stage => (!active && stage.status === 'running' ? { ...stage, status: 'stopped' } : stage));
  const source = record(state.source);
  const candidates = list(state.candidates);
  const kit = record(detail.kit);
  // the frame a stage is on: the people in it are drawn only if they have a box and all their keypoints
  const now = detail.now && Array.isArray(detail.now.people)
    ? { ...detail.now, people: list(detail.now.people).filter(person => Array.isArray(person.box) && person.box.length === 4
      && Array.isArray(person.kp) && person.kp.length >= 17 && person.kp.every(Array.isArray)) }
    : detail.now;
  const clips = list(state.clips).filter(clip => typeof clip.id === 'string' && Array.isArray(clip.crop) && Array.isArray(clip.frames))
    .map(clip => ({ ...clip, files: record(clip.files), variants: list(clip.variants).map(variant => ({ ...variant, files: record(variant.files) })) }));
  return {
    job, state: {
      ...state, log: list(state.log),
      motion: state.motion && Array.isArray(state.motion.timeline)
        ? { ...state.motion, spans: Array.isArray(state.motion.spans) ? state.motion.spans.filter(Array.isArray) : [] } : null,
    },
    kit, now, stages, source,
    by: Object.fromEntries(stages.map(stage => [stage.key, stage])),
    size: (Array.isArray(source.size) && source.size) || (job.source && job.source.size) || [1920, 1080],
    total: source.seconds || (job.source && job.source.seconds) || 0,
    clips, people: list(state.people).filter(person => Array.isArray(person.active)), candidates,
    engine: record(state.engine), options: record(state.options), costs: record(state.costs),
    reported: list(state.stages).length > 0,
    active,
    device: job.device || '',
    // where each hand moment's still sits in the sheets the engine made
    candAt: Object.fromEntries(((kit.cands && kit.cands.index) || []).map((k, i) => [k, i])),
    // a file of the run by its path in the run folder; nothing while the file is not there yet
    url: rel => (rel && files[rel] != null ? `/api/admin/jobs/${job.id}/file/${rel}?v=${files[rel]}` : null),
  };
}

// The hand moment a clip was made of (its number in the list of candidates).
// one number per person: a person the tracker lost and found again has several tracks but one number
export const personOf = item => (item && item.person != null && item.person >= 0 ? item.person : item ? item.track ?? item.id : null);
export const peopleCount = people => new Set(people.map(person => personOf(person))).size;

export const momentOf = (run, clip) => run.candidates.findIndex(cand => cand.clip === clip.id);

export const passedVariants = clip => (clip.variants || []).filter(variant => variant.passed);
