import { useEffect, useState } from 'react';
import { ArrowLeft, ArrowRight, Banana, Check, Copy, Download, Film, FolderOpen, RefreshCw, ShoppingBag } from 'lucide-react';
import { fileSize, money, rememberPurchase, savedPurchases } from './purchases.js';
import { formatHours } from './format.js';
import CardCheckout from './CardCheckout.jsx';
import './purchase.css';

export default function PurchasePage() {
  const id = window.location.pathname.split('/')[2];
  const stored = savedPurchases();
  const token = new URLSearchParams(window.location.hash.slice(1)).get('access') || stored.find(item => item.id === id)?.access_token;
  const [purchase, setPurchase] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [retry, setRetry] = useState(0);
  useEffect(() => { document.title = 'Peelset — Purchases & downloads'; }, []);
  async function load(signal) {
    const response = await fetch(`/api/purchases/${id}`, { headers: { Authorization: `Bearer ${token}` }, signal });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'We could not load this purchase.');
    setPurchase(data); setError('');
    rememberPurchase({ id, access_token: token, quote: data.quote, created_at: Date.now(), status: data.status });
    return data;
  }
  useEffect(() => {
    if (!id) return;
    if (!token) { setError('Open your private checkout link to access this purchase.'); return; }
    const controller = new AbortController();
    let timer;
    let attempts = 0;
    async function refresh() {
      setLoading(true);
      try {
        const data = await load(controller.signal);
        if (data.provider === 'stripe' && data.status === 'awaiting_payment' && ++attempts < 15) timer = setTimeout(refresh, 3000);
      } catch (error) { if (error.name !== 'AbortError') setError(error.message); }
      finally { if (!controller.signal.aborted) setLoading(false); }
    }
    refresh();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [id, token, retry]);

  async function demoPay() {
    setLoading(true); setError('');
    try {
      const response = await fetch(`/api/purchases/${id}/demo-pay`, { method: 'POST', headers: { Authorization: `Bearer ${token}` } });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Demo checkout could not be completed.');
      await load();
    } catch (error) { setError(error.message); }
    finally { setLoading(false); }
  }
  async function copyLink() {
    try {
      await navigator.clipboard.writeText(`${window.location.origin}/purchases/${id}#access=${token}`);
      setCopied(true);
    } catch { setError('Copy this page’s URL to keep your private download link.'); }
  }
  const paid = purchase?.status === 'paid' || purchase?.status === 'demo_paid';
  const cardEntry = purchase?.provider === 'demo' && purchase.status === 'awaiting_payment';
  return <div className={`purchase-shell ${cardEntry ? 'card-entry-page' : ''}`}>
    <header><a className="brand" href="/"><Banana className="brand-mark" strokeWidth={1.6} />peelset<span className="brand-period">.</span></a><a href="/collection"><ArrowLeft size={14} />My collections</a></header>
    <main>
      <p className="eyebrow"><span className="eyebrow-line" />YOUR RESEARCH, READY TO GO</p>
      <h1>{!id ? 'Purchases & downloads' : cardEntry ? 'Checkout' : paid ? purchase.ready ? 'Your dataset is ready.' : 'Your download has expired.' : 'Complete your purchase.'}</h1>
      {!id ? <>
        <p className="purchase-intro">Return to your datasets and pick up where you left off.</p>
        {stored.length ? <div className="purchase-history">{stored.map(item => <a href={`/purchases/${item.id}#access=${item.access_token}`} key={item.id}><span className="purchase-history-icon"><FolderOpen size={22} /></span><div><strong>{item.quote?.unique_clip_count?.toLocaleString('en-US') || 'Video'} clips · {item.quote ? money(item.quote.amount_cents, item.quote.currency) : 'Dataset purchase'}</strong><p>{item.status === 'paid' || item.status === 'demo_paid' ? 'View download' : 'View purchase'}{item.quote?.payment_provider === 'demo' ? ' · Local demo' : ''}</p></div><ArrowRight size={18} /></a>)}</div> : <div className="purchase-empty"><ShoppingBag size={32} /><h2>Your next dataset starts here.</h2><p>Purchases made in this browser will appear here.</p><a href="/explore" className="primary-button">Explore the data<ArrowRight size={15} /></a></div>}
      </> : <>
        {!purchase && loading && <p className="purchase-intro" role="status">Loading your purchase…</p>}
        {cardEntry && <div className="card-checkout-layout">
          <CardCheckout quote={purchase.quote} loading={loading} onComplete={demoPay} />
          <aside className="card-order-summary" aria-label="Order summary">
            <span className="purchase-status-icon"><FolderOpen size={25} /></span><h2>Your dataset</h2>
            <p className="purchase-stats">{purchase.quote.unique_clip_count.toLocaleString('en-US')} clips · {formatHours(purchase.quote.duration_seconds)} hours</p>
            <div className="purchase-bundles">{purchase.bundles.map((bundle, index) => <div key={index}><FolderOpen size={16} /><span>{bundle.name}</span><small>{bundle.clip_count.toLocaleString('en-US')} clips</small></div>)}</div>
            <p className="purchase-detail">Video files, annotations, and provenance. Ready to download after checkout.</p>
            <div className="purchase-total"><span>Total</span><strong>{money(purchase.quote.amount_cents, purchase.quote.currency)}</strong></div>
          </aside>
        </div>}
        {purchase && !cardEntry && <section className="purchase-card">
          <span className="purchase-status-icon">{paid ? <Check size={27} /> : <ShoppingBag size={27} />}</span>
          {purchase.provider === 'demo' && <p className="demo-checkout-note">Demo purchase · No money was charged.</p>}
          <div className="purchase-total"><span>{paid ? purchase.provider === 'demo' ? 'Demo total' : 'Amount paid' : 'Order total'}</span><strong>{money(purchase.quote.amount_cents, purchase.quote.currency)}</strong></div>
          <p className="purchase-stats"><Film size={17} />{purchase.quote.unique_clip_count.toLocaleString('en-US')} clips · {formatHours(purchase.quote.duration_seconds)} hours · {fileSize(purchase.quote.bytes)}</p>
          <div className="purchase-bundles">{purchase.bundles.map((bundle, index) => <div key={index}><FolderOpen size={17} /><span>{bundle.name}</span><small>{bundle.clip_count.toLocaleString('en-US')} clips</small></div>)}</div>
          <p className="purchase-intro">Your ZIP contains the available videos, annotations, and source provenance. Each bundle keeps its own selection and instructions.</p>
          {purchase.quote.requires_custom_ack && <p className="purchase-detail">New diffusion edits are included as instructions only; new video variants have not been generated.</p>}
          <div className="purchase-actions">
            {purchase.ready ? <><a className="primary-button" href={purchase.download_url}><Download size={18} />Download dataset ZIP</a><button className="secondary-button" onClick={copyLink}>{copied ? <Check size={16} /> : <Copy size={16} />}{copied ? 'Private link copied' : 'Copy private download link'}</button><p className="purchase-detail">Available until {new Date(purchase.expires_at * 1000).toLocaleDateString('en-US')}. Keep your private link to download again.</p></>
              : purchase.status === 'awaiting_payment' ? purchase.provider === 'demo' ? <button className="primary-button" onClick={demoPay} disabled={loading}>{loading ? 'Completing demo checkout…' : 'Complete demo purchase'}<ArrowRight size={17} /></button>
                : <><a className="primary-button" href={purchase.checkout_url}>Continue to secure payment<ArrowRight size={17} /></a><button className="secondary-button" disabled={loading} onClick={() => setRetry(value => value + 1)}><RefreshCw size={15} />{loading ? 'Checking payment…' : 'I’ve paid — check again'}</button></>
              : <a className="primary-button" href="/explore">Find another dataset<ArrowRight size={17} /></a>}
          </div>
          <p className="purchase-order-id">Purchase ID · {purchase.id}</p>
        </section>}
        {error && <div className="purchase-error" role="alert"><p>{error}</p><button className="secondary-button" onClick={() => setRetry(value => value + 1)}>Try again</button></div>}
      </>}
      <a className="purchase-back" href="/explore"><ArrowLeft size={14} />Back to the data explorer</a>
    </main>
  </div>;
}
