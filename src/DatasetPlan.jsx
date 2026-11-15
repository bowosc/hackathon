import { useState } from 'react';
import { ArrowRight, Check, ChevronDown, Film, FolderPlus, Play, SlidersHorizontal } from 'lucide-react';
import DatasetReview from './DatasetReview.jsx';
import { formatHours } from './format.js';
import settings from '../config/settings.json';
import commerce from '../config/commerce.json';
import './dataset.css';

const changes = [['upscale', 'Upscale resolution'], ['object_swap', 'Replace objects'], ['lighting', 'Vary lighting'], ['background', 'Change background']];

export function DatasetFolder({ clips }) {
  return <span className="dataset-folder" aria-hidden="true">
    <svg className="folder-back" viewBox="0 0 144 112" fill="none"><path d="M9 62V24a7 7 0 0 1 7-7h35l11 5h65a7 7 0 0 1 7 7v33Z" fill="#e5d492" stroke="#c9b873" /></svg>
    {clips.filter(clip => clip.image).slice(0, 3).map((clip, index) => <span className={`folder-video folder-video-${index}`} key={clip.id}>
      <img src={clip.image} alt="" style={{ objectPosition: clip.image_position }} />
      <Play fill="currentColor" strokeWidth={0} />
    </span>)}
    <svg className="folder-front" viewBox="0 0 144 112" fill="none"><path d="M11 54h122a5 5 0 0 1 5 6l-6 37a8 8 0 0 1-8 7H20a8 8 0 0 1-8-7L6 60a5 5 0 0 1 5-6Z" fill="var(--yellow)" stroke="#cbb14f" /><path d="M15 58h114" stroke="#fff4bc" strokeLinecap="round" /></svg>
    <span className="folder-label"><Film size={11} />DATASET</span>
  </span>;
}

export default function DatasetPlan({ compilation, selection, count, durationSeconds = 0, previewClips = [], onChange, disabled, onSelectAll, onSaveBundle, bundleSaved, savingBundle, saveError }) {
  const [editing, setEditing] = useState(false);
  const [review, setReview] = useState(false);
  const plan = compilation.plan;
  const hasVariants = plan.modifications.some(id => id !== 'upscale');
  const generated = hasVariants ? count * plan.variants_per_clip : 0;
  const hours = durationSeconds / 3600;
  const formattedHours = formatHours(durationSeconds);
  function update(key, value) { onChange({ ...plan, [key]: value }); }
  function toggle(id) { update('modifications', plan.modifications.includes(id) ? plan.modifications.filter(value => value !== id) : [...plan.modifications, id]); }
  function openReview() {
    if (plan.modifications.includes('object_swap') && !plan.target_object.trim()) {
      setEditing(true);
      setTimeout(() => {
        const input = document.getElementById('dataset-target-object');
        input?.setCustomValidity('Enter a target object.');
        input?.reportValidity();
      }, 0);
      return;
    }
    setReview(true);
  }
  return <section className="compiled-dataset" aria-label="Compiled dataset">
    <div className="compiled-header">
      <DatasetFolder clips={previewClips} />
      <div>
        <h2>Your dataset, compiled.</h2>
        <p className="compiled-metrics" aria-live="polite" aria-atomic="true"><span><strong>{count.toLocaleString('en-US')}</strong> {count === 1 ? 'clip,' : 'clips,'}</span><span><strong>{formattedHours}</strong> {hours === 1 ? 'hour' : 'hours'} of content</span></p>
        <p>{plan.summary}</p>
      </div>
      <div className="compiled-actions">
        <button className="primary-button" disabled={disabled || !count} onClick={openReview}>{commerce.checkout_mode === 'purchase' ? 'Purchase dataset' : 'Review dataset'}<ArrowRight size={14} /></button>
        <button className="secondary-button" disabled={disabled || !count || savingBundle || bundleSaved} onClick={onSaveBundle}>{bundleSaved ? <Check size={15} /> : <FolderPlus size={15} />}{savingBundle ? 'Saving bundle…' : bundleSaved ? 'Saved to collections' : 'Save to My collections'}</button>
        <button className="secondary-button edit-changes" aria-expanded={editing} aria-controls="dataset-adjustments" onClick={() => setEditing(value => !value)}><SlidersHorizontal size={16} />Customize video variants<ChevronDown size={15} /></button>
        {bundleSaved && <a href="/collection">View collections<ArrowRight size={12} /></a>}
      </div>
    </div>
    <div className="compiled-tags compiled-bottom">{plan.modifications.includes('upscale') && <span>{plan.resolution} copies</span>}{generated > 0 && <span>{generated} planned variants</span>}{plan.modifications.includes('object_swap') && plan.target_object && <span>{plan.target_object}</span>}</div>
    {saveError && <p className="dataset-error" role="alert">{saveError}</p>}
    {editing && <div className="dataset-adjustments" id="dataset-adjustments">
      <div className="change-checklist">{changes.map(([id, label]) => <label key={id}><input type="checkbox" checked={plan.modifications.includes(id)} onChange={() => toggle(id)} />{label}</label>)}</div>
      <div className="change-fields">
        {plan.modifications.includes('object_swap') && <label>Target object<input id="dataset-target-object" required maxLength={120} value={plan.target_object} placeholder="e.g. bananas" onChange={event => { event.target.setCustomValidity(''); update('target_object', event.target.value); }} /></label>}
        {plan.modifications.includes('background') && <label>Target background<input value={plan.background} maxLength={200} placeholder="e.g. a banana plantation" onChange={event => update('background', event.target.value)} /></label>}
        {plan.modifications.includes('lighting') && <label>Lighting<select value={plan.lighting} onChange={event => update('lighting', event.target.value)}><option value="varied">Varied lighting</option><option value="daylight">Daylight</option><option value="low_light">Low light</option></select></label>}
        {plan.modifications.includes('upscale') && <label>Output resolution<select value={plan.resolution} onChange={event => update('resolution', event.target.value)}><option>1080p</option><option>4K</option></select></label>}
        {hasVariants && <label>Variants per source<input type="number" min="1" max={settings.max_variants_per_clip} value={plan.variants_per_clip} onChange={event => update('variants_per_clip', Math.max(1, Math.min(settings.max_variants_per_clip, Number(event.target.value) || 1)))} /></label>}
      </div><p>Original clips and annotations stay included. Changes apply to new copies.</p>
      <button className="select-all-sources" onClick={onSelectAll}>Include all matching sources</button>
    </div>}
    {review && <DatasetReview compilation={compilation} selection={selection} count={count} onClose={() => setReview(false)} />}
  </section>;
}
