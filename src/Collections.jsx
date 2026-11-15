import { useEffect, useRef, useState } from 'react';
import { ArrowRight, FolderPlus, Trash2 } from 'lucide-react';
import { DatasetFolder } from './DatasetPlan.jsx';
import DatasetReview from './DatasetReview.jsx';
import { postJSON } from './api.js';
import { formatHours } from './format.js';
import './collections.css';

export default function Collections({ bundles, onRemove, storageError }) {
  const [selected, setSelected] = useState(() => bundles.map(bundle => bundle.id));
  const [checkout, setCheckout] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [review, setReview] = useState(false);
  const [retry, setRetry] = useState(0);
  const all = useRef(null);
  const chosen = bundles.filter(bundle => selected.includes(bundle.id));
  const selectionKey = chosen.map(bundle => bundle.id).join(',');
  const current = checkout?.bundles.map(bundle => bundle.id).join(',') === selectionKey;

  useEffect(() => {
    if (all.current) all.current.indeterminate = chosen.length > 0 && chosen.length < bundles.length;
  }, [chosen.length, bundles.length]);
  useEffect(() => {
    if (!selectionKey) { setCheckout(null); setError(''); setLoading(false); return; }
    const controller = new AbortController();
    setLoading(true); setError('');
    postJSON('/api/bundle-checkout-preview', { bundle_ids: selectionKey.split(',') }, controller.signal)
      .then(setCheckout)
      .catch(error => { if (error.name !== 'AbortError') setError(error.message); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [selectionKey, retry]);

  function toggle(id) { setSelected(ids => ids.includes(id) ? ids.filter(value => value !== id) : [...ids, id]); }

  if (!bundles.length) return <section className="state-message empty-state">
    <FolderPlus size={34} strokeWidth={1.3} /><h2>No bundles saved yet</h2>
    <p>Save a compiled dataset from the explorer to keep its clips and transformation settings together.</p>
    <a className="primary-button" href="/explore">Find your dataset<ArrowRight size={15} /></a>
  </section>;

  return <section className="bundle-collections" aria-label="Saved dataset bundles">
    <div className="collection-toolbar"><label><input ref={all} type="checkbox" aria-label="Select all bundles" checked={chosen.length === bundles.length} onChange={() => setSelected(chosen.length === bundles.length ? [] : bundles.map(bundle => bundle.id))} /><span>{bundles.length} saved {bundles.length === 1 ? 'bundle' : 'bundles'}</span></label><span>{chosen.length} selected</span></div>
    <div className="saved-bundle-grid">{bundles.map(saved => {
      const bundle = checkout?.bundles.find(bundle => bundle.id === saved.id) || saved;
      const checked = selected.includes(bundle.id);
      return <article key={bundle.id} className={`bundle-card ${checked ? 'selected' : ''}`}>
        <div className="bundle-card-controls"><label><input type="checkbox" checked={checked} onChange={() => toggle(bundle.id)} aria-label={`Select bundle: ${bundle.name}`} />{checked ? 'Selected' : 'Select bundle'}</label><button className="icon-button" aria-label={`Remove bundle: ${bundle.name}`} onClick={() => onRemove(bundle.id)}><Trash2 size={15} /></button></div>
        <div className="bundle-card-body"><DatasetFolder clips={bundle.previews || []} /><div><h2 title={bundle.name}>{bundle.name}</h2><p><strong>{bundle.clip_count.toLocaleString('en-US')}</strong> clips · <strong>{formatHours(bundle.duration_seconds)}</strong> hours</p></div></div>
        <div className="compiled-tags bundle-tags">{bundle.plan.modifications.length ? bundle.plan.modifications.map(id => <span key={id} title={id === 'object_swap' ? bundle.plan.target_object : undefined}>{({ upscale: 'Upscaled', object_swap: bundle.plan.target_object || 'Object replacement', lighting: 'Lighting variants', background: 'Background variants' })[id]}</span>) : <span>Original footage & annotations</span>}</div>
      </article>;
    })}</div>
    {(error || storageError) && <div className="collection-error" role="alert"><p>{error || storageError}</p>{error && <button className="secondary-button" onClick={() => setRetry(value => value + 1)}>Try again</button>}</div>}
    <div className="bundle-checkout-summary" aria-busy={loading}>
      <div><strong>{chosen.length} {chosen.length === 1 ? 'bundle' : 'bundles'} selected</strong><p>{loading || !current ? chosen.length ? 'Checking your saved bundles…' : 'Select bundles to check out together.' : `${checkout.totals.clip_count.toLocaleString('en-US')} clips · ${formatHours(checkout.totals.duration_seconds)} hours of content`}</p>{current && checkout.totals.unique_clip_count < checkout.totals.clip_count && <span>{checkout.totals.unique_clip_count.toLocaleString('en-US')} unique source clips across these bundles.</span>}</div>
      <button className="primary-button" disabled={!chosen.length || loading || !current || !!error} onClick={() => setReview(true)}>Checkout selected<ArrowRight size={16} /></button>
    </div>
    {review && current && <DatasetReview checkout={checkout} onClose={() => setReview(false)} />}
  </section>;
}
