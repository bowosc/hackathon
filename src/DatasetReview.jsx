import { useEffect, useRef, useState } from 'react';
import { ArrowRight, Check, Download, X } from 'lucide-react';
import { download, postJSON } from './api.js';
import { formatHours } from './format.js';
import commerce from '../config/commerce.json';
import { fileSize, money, rememberPurchase } from './purchases.js';
import './dataset.css';

function Changes({ plan, requested }) {
  const labels = { upscale: 'Upscale resolution', object_swap: 'Replace objects', lighting: 'Vary lighting', background: 'Change background' };
  return <>
    <ul className="review-changes"><li><Check size={14} />Original footage and annotations included</li>{plan.modifications.map(id => <li key={id}><Check size={14} />{labels[id]}{id === 'object_swap' ? `: ${plan.target_object}` : id === 'background' && plan.background ? `: ${plan.background}` : id === 'upscale' ? `: ${plan.resolution}` : id === 'lighting' ? `: ${plan.lighting.replace('_', ' ')}` : ''}</li>)}</ul>
    {plan.modifications.some(id => id !== 'upscale') && <p className="review-note">{plan.variants_per_clip} {requested ? 'variants requested' : 'generated variants'} per source clip.</p>}
  </>;
}

export default function DatasetReview({ compilation, selection, count, checkout, onClose }) {
  const ref = useRef(null);
  const [manifest, setManifest] = useState(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [quote, setQuote] = useState(null);
  const [quoting, setQuoting] = useState(false);
  const [accepted, setAccepted] = useState(false);
  const [retry, setRetry] = useState(0);
  const purchasing = commerce.checkout_mode === 'purchase';
  const cardEntry = purchasing && commerce.demo_card_entry && quote?.demo_available;
  const purchaseDraft = checkout ? { bundle_ids: checkout.bundles.map(bundle => bundle.id) }
    : { dataset: { prompt: compilation.prompt, interpretation_mode: compilation.mode, selection } };
  const draftKey = JSON.stringify(purchaseDraft);
  const multiple = checkout && checkout.bundles.length > 1;
  useEffect(() => {
    if (!purchasing) return;
    const controller = new AbortController();
    setQuoting(true); setError('');
    postJSON('/api/purchase-preview', JSON.parse(draftKey), controller.signal)
      .then(setQuote)
      .catch(error => { if (error.name !== 'AbortError') setError(error.message); })
      .finally(() => { if (!controller.signal.aborted) setQuoting(false); });
    return () => controller.abort();
  }, [draftKey, purchasing, retry]);
  useEffect(() => {
    ref.current.showModal();
    const previous = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = previous; };
  }, []);
  async function submit(event) {
    event.preventDefault(); setSaving(true); setError('');
    const demo = event.nativeEvent.submitter?.value === 'demo';
    const form = new FormData(event.currentTarget);
    const contact = { lab: form.get('lab'), email: form.get('email') };
    try {
      if (purchasing) {
        const purchase = await postJSON('/api/purchases', { ...purchaseDraft, ...contact, accept_available_versions: accepted, expected_amount_cents: quote.amount_cents, demo_checkout: demo, demo_card_entry: !!cardEntry && !demo });
        rememberPurchase({ ...purchase, created_at: Date.now() });
        window.location.assign(purchase.checkout_url);
        return;
      }
      setManifest(await postJSON(checkout ? '/api/dataset-checkouts' : '/api/dataset-requests', checkout
        ? { ...contact, bundle_ids: checkout.bundles.map(bundle => bundle.id) }
        : { ...contact, prompt: compilation.prompt, interpretation_mode: compilation.mode, selection }));
    } catch (error) { setError(error.code === 'checkout_unavailable' ? 'We couldn’t start checkout. Please try again.' : error.message); }
    finally { setSaving(false); }
  }
  return <dialog ref={ref} className="dataset-review clip-dialog" aria-labelledby="dataset-review-title" onCancel={event => { if (saving) event.preventDefault(); else onClose(); }} onClick={event => { if (event.target === event.currentTarget && !saving) onClose(); }}>
    <div className="review-header">
      <h2 id="dataset-review-title">{manifest ? checkout ? 'Your checkout request is saved.' : 'Your dataset request is saved.' : multiple ? 'Review your datasets' : 'Review your dataset'}</h2>
      <button className="review-close icon-button" aria-label="Close dataset review" disabled={saving} onClick={onClose}><X size={18} /></button>
    </div>
    {manifest ? <div className="request-success review-body">
      <span className="success-mark"><Check size={24} /></span>
      <p>{checkout ? `${manifest.totals.bundle_count} bundles and ${manifest.totals.clip_count.toLocaleString('en-US')} clips are included in one request, with each bundle’s transformation settings.` : `${manifest.planned_outputs.source_clips} source clips and your full transformation instructions are included.`}</p>
      <p className="review-note">No payment has been taken. Video generation has not started.</p>
      <button className="primary-button" onClick={() => download(manifest, `${manifest.id}.json`)}><Download size={15} />{checkout ? 'Download checkout request' : 'Download dataset request'}</button><button className="secondary-button" onClick={onClose}>{checkout ? 'Back to collections' : 'Keep refining'}</button>
    </div> : <form className="review-form" onSubmit={submit}>
      <div className="review-body">
        {checkout ? <>
          <div className="request-brief"><strong>{checkout.totals.bundle_count} bundles · {checkout.totals.clip_count.toLocaleString('en-US')} clips</strong><p>{formatHours(checkout.totals.duration_seconds)} hours of content across your selected bundles.</p>{checkout.totals.unique_clip_count < checkout.totals.clip_count && <p>{checkout.totals.unique_clip_count.toLocaleString('en-US')} unique source clips. Each bundle keeps its own transformations.</p>}</div>
          {checkout.bundles.map(bundle => <section className="review-bundle" key={bundle.id}>
            <h3>{bundle.name}</h3><p>{bundle.clip_count.toLocaleString('en-US')} clips · {formatHours(bundle.duration_seconds)} hours of content</p>
            {purchasing && bundle.plan.modifications.length > 0 && <h3>Transformation instructions</h3>}<Changes plan={bundle.plan} requested={purchasing} /><details><summary>View original instructions</summary><p className="request-instructions">{bundle.prompt}</p></details>
          </section>)}
        </> : <>
          <div className="request-brief"><strong>{count.toLocaleString('en-US')} source clips</strong><p>{compilation.plan.summary}</p></div>
          <h3>Your instructions</h3><p className="request-instructions">{compilation.prompt}</p>{purchasing && compilation.plan.modifications.length > 0 && <h3>Transformation instructions</h3>}<Changes plan={compilation.plan} requested={purchasing} />
        </>}
        {purchasing && <section className="purchase-quote" aria-label="Purchase total" aria-live="polite">
          {quote ? <><div><span>Total</span><strong>{money(quote.amount_cents, quote.currency)}</strong></div><p>{money(quote.cents_per_hour, quote.currency)} per source hour · {money(quote.minimum_order_cents, quote.currency)} minimum. Overlapping source clips are billed once.</p><h3>Included in your download</h3><p>{quote.video_count.toLocaleString('en-US')} available video files · {fileSize(quote.bytes)}<br />Original footage, available processed versions, annotations, and provenance in one ZIP.</p>
            {quote.requires_custom_ack && <label className="purchase-ack"><input type="checkbox" checked={accepted} onChange={event => setAccepted(event.target.checked)} required /><span>I’m purchasing the available versions. New diffusion edits are saved as instructions and are not generated with this purchase.</span></label>}
            {quote.payment_provider === 'demo' && <p className="demo-checkout-note">Local demo checkout. No card is required and no money is charged.</p>}
          </> : <p>{quoting ? 'Calculating your total…' : 'Your purchase total is unavailable.'}</p>}
        </section>}
        <label>Lab or organization<input name="lab" placeholder="Your research team" autoComplete="organization" maxLength={160} /></label>
        <label>Email<input name="email" type="email" placeholder="you@lab.com" autoComplete="email" maxLength={254} required /></label>
        {error && <div className="dataset-error" role="alert"><p>{error}</p>{purchasing && <button type="button" className="secondary-button" onClick={() => setRetry(value => value + 1)}>Try again</button>}</div>}
      </div>
      <div className="review-footer">
        <p className="review-note">{purchasing ? 'Download your dataset immediately after checkout.' : checkout ? 'Submit your selected bundles together. Pricing and generation are confirmed separately.' : 'Save this dataset request. Pricing and generation are confirmed separately.'}</p>
        <button className="primary-button request-submit" disabled={saving || (purchasing && (quoting || !quote || (quote.requires_custom_ack && !accepted)))}>{saving ? purchasing ? 'Preparing your download…' : 'Saving request…' : purchasing ? cardEntry ? 'Continue to payment' : quote?.payment_provider === 'demo' ? 'Continue to demo checkout' : quote ? `Pay ${money(quote.amount_cents, quote.currency)} & download` : 'Continue to checkout' : multiple ? 'Request these datasets' : 'Request this dataset'}<ArrowRight size={15} /></button>
        {purchasing && quote?.demo_available && !cardEntry && <button type="submit" name="checkout" value="demo" className="demo-skip-button" disabled={saving || quoting || (quote.requires_custom_ack && !accepted)}>Demo: skip payment</button>}
      </div>
    </form>}
  </dialog>;
}
