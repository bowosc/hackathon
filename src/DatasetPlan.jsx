import { useEffect, useRef, useState } from 'react';
import { ArrowRight, Check, ChevronDown, Download, SlidersHorizontal, Sparkles, X } from 'lucide-react';
import { download, postJSON } from './api.js';
import settings from '../config/settings.json';
import './dataset.css';

const changes = [['upscale', 'Upscale resolution'], ['object_swap', 'Replace objects'], ['lighting', 'Vary lighting'], ['background', 'Change background']];

function Review({ compilation, selection, count, onClose }) {
  const ref = useRef(null);
  const [manifest, setManifest] = useState(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const plan = compilation.plan;
  useEffect(() => {
    ref.current.showModal();
    const previous = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = previous; };
  }, []);
  async function submit(event) {
    event.preventDefault(); setSaving(true); setError('');
    const form = new FormData(event.currentTarget);
    try { setManifest(await postJSON('/api/dataset-requests', { prompt: compilation.prompt, interpretation_mode: compilation.mode, selection, lab: form.get('lab'), email: form.get('email') })); }
    catch (error) { setError(error.message); }
    finally { setSaving(false); }
  }
  return <dialog ref={ref} className="dataset-review clip-dialog" aria-labelledby="dataset-review-title" onCancel={event => { if (saving) event.preventDefault(); else onClose(); }} onClick={event => { if (event.target === event.currentTarget && !saving) onClose(); }}>
    <button className="review-close icon-button" aria-label="Close dataset request" disabled={saving} onClick={onClose}><X size={18} /></button>
    {manifest ? <div className="request-success">
      <span className="success-mark"><Check size={24} /></span><h2 id="dataset-review-title">Your dataset request is saved.</h2>
      <p>{manifest.planned_outputs.source_clips} source clips and your full transformation instructions are included.</p>
      <p className="review-note">No payment has been taken. Video generation has not started.</p>
      <button className="primary-button" onClick={() => download(manifest, `${manifest.id}.json`)}><Download size={15} />Download dataset request</button><button className="secondary-button" onClick={onClose}>Keep refining</button>
    </div> : <form onSubmit={submit}>
      <h2 id="dataset-review-title">Review your dataset</h2>
      <div className="request-brief"><strong>{count} source clips</strong><p>{plan.summary}</p></div>
      <h3>Your instructions</h3><p className="request-instructions">{compilation.prompt}</p>
      <ul className="review-changes"><li><Check size={14} />Original footage and annotations included</li>{plan.modifications.map(id => <li key={id}><Check size={14} />{changes.find(change => change[0] === id)?.[1]}{id === 'object_swap' ? `: ${plan.target_object}` : id === 'background' && plan.background ? `: ${plan.background}` : id === 'upscale' ? `: ${plan.resolution}` : id === 'lighting' ? `: ${plan.lighting.replace('_', ' ')}` : ''}</li>)}</ul>
      {plan.modifications.some(id => id !== 'upscale') && <p className="review-note">{plan.variants_per_clip} generated variants per source clip.</p>}
      <label>Lab or organization<input name="lab" placeholder="Your research team" autoComplete="organization" maxLength={160} /></label>
      <label>Email<input name="email" type="email" placeholder="you@lab.com" autoComplete="email" maxLength={254} required /></label>
      {error && <p className="dataset-error" role="alert">{error}</p>}
      <p className="review-note">Save this dataset request. Pricing and generation are confirmed separately.</p>
      <button className="primary-button request-submit" disabled={saving}>{saving ? 'Saving request…' : 'Request this dataset'}<ArrowRight size={15} /></button>
    </form>}
  </dialog>;
}

export default function DatasetPlan({ compilation, selection, count, onChange, disabled, onSelectAll }) {
  const [editing, setEditing] = useState(false);
  const [review, setReview] = useState(false);
  const plan = compilation.plan;
  const hasVariants = plan.modifications.some(id => id !== 'upscale');
  const generated = hasVariants ? count * plan.variants_per_clip : 0;
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
    <div className="compiled-header"><span className="compiled-symbol"><Sparkles size={18} /></span><div><h2>Your dataset, compiled.</h2><p>{plan.summary}</p></div><button className="primary-button" disabled={disabled || !count} onClick={openReview}>Review dataset<ArrowRight size={14} /></button></div>
    <div className="compiled-bottom"><div className="compiled-tags"><span>{count} selected sources</span>{plan.modifications.includes('upscale') && <span>{plan.resolution} copies</span>}{generated > 0 && <span>{generated} planned variants</span>}{plan.modifications.includes('object_swap') && plan.target_object && <span>{plan.target_object}</span>}</div><button className="edit-changes" aria-expanded={editing} onClick={() => setEditing(value => !value)}><SlidersHorizontal size={12} />Adjust changes<ChevronDown size={12} /></button></div>
    <p className="selection-scope">All matching sources are included, across every page. Uncheck any you don’t need.</p>
    {compilation.mode === 'demo' && <p className="demo-interpretation">Keyword interpretation · The AI interpreter is not connected here, or did not answer.</p>}
    {editing && <div className="dataset-adjustments">
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
    {review && <Review compilation={compilation} selection={selection} count={count} onClose={() => setReview(false)} />}
  </section>;
}
