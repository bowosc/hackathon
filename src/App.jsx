import { useEffect, useRef, useState } from 'react';
import { ArrowLeft, ArrowRight, ArrowUpRight, Banana, Bookmark, Check, ChevronDown, Download, LayoutGrid, List, Plus, SlidersHorizontal, Sparkles, X } from 'lucide-react';
import site from '../config/site.json';
import settings from '../config/settings.json';
import DatasetPlan from './DatasetPlan.jsx';
import { download, getJSON, postJSON } from './api.js';
import HomePage from './HomePage.jsx';

const filterGroups = [
  {
    "key": "environment",
    "label": "Environment",
    "options": [
      "Grocery store",
      "Gas station"
    ]
  },
  {
    "key": "action",
    "label": "Action",
    "options": [
      "Pick & place",
      "Restocking",
      "Carrying",
      "Tool use",
      "Navigation",
      "Human-object interaction"
    ]
  },
  {
    "key": "annotation",
    "label": "Annotations",
    "options": [
      "Object bounding boxes",
      "Action segments",
      "Object tracking",
      "Scene descriptions",
      "Human pose"
    ]
  },
  {
    "key": "resolution",
    "label": "Resolution",
    "options": [
      "720p",
      "1080p",
      "4K"
    ]
  },
  {
    "key": "lighting",
    "label": "Lighting",
    "options": [
      "Daylight",
      "Indoor",
      "Low light"
    ]
  }
];
const initial = new URLSearchParams(window.location.search);
const currentPage = window.location.pathname === '/collection' ? 'collection' : 'explorer';
const storageKey = site.storage_key;
const duration = seconds => `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;

function loadSaved() {
  try {
    const value = JSON.parse(localStorage.getItem(storageKey) || '[]');
    return Array.isArray(value) ? [...new Set(value.filter(id => typeof id === 'string'))] : [];
  } catch { return []; }
}

function Mark() {
  return <Banana className="brand-mark" strokeWidth={1.6} aria-hidden="true" />;
}

function ClipCard({ clip, saved, onToggle, onOpen, selectionMode, included, onSelect }) {
  const checked = selectionMode ? included : saved;
  const label = selectionMode ? (included ? 'Exclude from dataset' : 'Include in dataset') : (saved ? 'Remove from collection' : 'Add to collection');
  return <article className={`clip-card ${selectionMode && !included ? 'is-excluded' : ''}`}>
    <button className="clip-preview" onClick={() => onOpen(clip)} aria-label={`View clip details: ${clip.title}`}>
      <img src={clip.image} alt={clip.image_alt} style={{ objectPosition: clip.image_position }} loading="lazy" />
      <span className="camera-code"><span />{clip.camera}</span>
      <span className="preview-open"><ArrowUpRight size={21} /></span>
      <span className="clip-duration">{duration(clip.duration_seconds)}</span>
    </button>
    <div className="clip-info">
      <div className="clip-category"><span>{clip.environment}</span><span>{clip.source_resolution} · {clip.fps} fps</span></div>
      <div className="clip-title-row"><button className="clip-title" onClick={() => onOpen(clip)}>{clip.title}</button>
        <button className={`save-button ${checked ? 'is-saved' : ''}`} onClick={() => selectionMode ? onSelect(clip.id) : onToggle(clip.id)} aria-label={`${label}: ${clip.title}`} aria-pressed={checked}>{checked ? <Check size={17} /> : <Plus size={17} />}</button>
      </div>
      <div className="clip-tags">{clip.actions.slice(0, 2).map(action => <span key={action}>{action}</span>)}{clip.variants.length > 1 && <span className="variant-count" title="Available versions"><Sparkles size={11} />{clip.variants.length}</span>}</div>
    </div>
  </article>;
}

function ClipDialog({ clip, saved, onToggle, onClose }) {
  const ref = useRef(null);
  const versions = {
    source: {
      title: "Original CCTV",
      description: "The source recording, clipped and labeled."
    },
    upscaled: {
      title: "Diffusion upscaled",
      description: "Higher-resolution reconstruction. Fine visual details may be synthesized."
    },
    augmented: {
      title: "Diffusion augmented",
      description: "Variants with modified lighting, objects, or backgrounds."
    }
  };
  useEffect(() => {
    ref.current.showModal();
    const previous = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = previous; };
  }, []);

  return <dialog ref={ref} className="clip-dialog" aria-labelledby="clip-dialog-title" onCancel={onClose} onClick={event => { if (event.target === event.currentTarget) onClose(); }}>
    <div className="detail-preview">
      {clip.video_url ? <video src={clip.video_url} poster={clip.image} controls /> : <img src={clip.image} alt={clip.image_alt} />}
      {!clip.video_url && <span className="detail-sample">Illustrative still · No video attached</span>}
      <button className="dialog-close" onClick={onClose} aria-label="Close clip details"><X size={19} /></button>
    </div>
    <div className="detail-body">
      <div className="eyebrow">{clip.id} <span> / </span> {clip.environment}</div>
      <h2 id="clip-dialog-title">{clip.title}</h2><p className="detail-description">{clip.description}</p>
      <dl className="detail-stats">{[["Duration", duration(clip.duration_seconds)], ["Resolution", clip.source_resolution], ["Frame rate", `${clip.fps} fps`]].map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
      <h3>Actions</h3><div className="detail-tags">{clip.actions.map(value => <span key={value}>{value}</span>)}</div>
      <h3>Included annotations</h3><div className="annotation-list">{clip.annotations.map(value => <span key={value}><Check size={14} />{value}</span>)}</div>
      <h3>Available versions</h3><div className="version-list">{clip.variants.map(variant => <div key={variant}><Check size={15} /><div><strong>{versions[variant].title}</strong><p>{versions[variant].description}</p></div></div>)}</div>
      <p className="lineage">Every generated version links back to its source clip.</p>
      <div className="detail-actions"><button className="primary-button" onClick={() => onToggle(clip.id)}>{saved ? <Check size={16} /> : <Plus size={16} />}{saved ? "Remove from collection" : "Add to collection"}</button><button className="secondary-button" onClick={() => download(clip, `${clip.id}.json`)}><Download size={16} />Download metadata</button></div>
    </div>
  </dialog>;
}

function loadCompilation() {
  try {
    const value = JSON.parse(sessionStorage.getItem('peelset:compiled') || 'null');
    return value && typeof value.prompt === 'string' && typeof value.plan?.summary === 'string' && Array.isArray(value.plan.modifications) ? value : null;
  } catch { return null; }
}

export default function App() {
  const [saved, setSaved] = useState(loadSaved);
  const [compiled, setCompiled] = useState(loadCompilation);
  const [query, setQuery] = useState(() => initial.get('q') || loadCompilation()?.prompt || '');
  const [compiling, setCompiling] = useState(false);
  const [compileError, setCompileError] = useState('');
  const [excluded, setExcluded] = useState([]);
  const [filters, setFilters] = useState(() => Object.fromEntries(filterGroups.map(group => [group.key, initial.getAll(group.key)])));
  const [variant, setVariant] = useState(initial.get('variant') || '');
  const [sort, setSort] = useState(initial.get('sort') || settings.default_sort);
  const [page, setPage] = useState(0);
  const [layout, setLayout] = useState('grid');
  const [mobileFilters, setMobileFilters] = useState(false);
  const [result, setResult] = useState({ clips: [], total: 0, catalog_total: 0, selected_count: 0 });
  const [collection, setCollection] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [retry, setRetry] = useState(0);
  const [activeClip, setActiveClip] = useState(null);
  const params = new URLSearchParams();
  if (variant) params.set('variant', variant);
  if (sort !== settings.default_sort) params.set('sort', sort);
  Object.entries(filters).forEach(([key, values]) => values.forEach(value => params.append(key, value)));
  const queryString = params.toString();
  const savedIds = saved.join(',');
  const activeFilters = Object.entries(filters).flatMap(([key, values]) => values.map(value => ({ key, value })));
  const criteria = compiled ? { summary: '', actions: compiled.plan.actions, objects: compiled.plan.objects,
    environments: compiled.plan.environments, annotations: compiled.plan.annotations, keywords: compiled.plan.keywords } : null;
  const previewKey = JSON.stringify({ plan: criteria, filters, variant: variant || null, sort, excluded_ids: excluded });
  const selection = { plan: compiled?.plan || null, filters, variant: variant || null, sort, excluded_ids: excluded };

  useEffect(() => { document.title = window.location.pathname === '/' ? 'Peelset — Real-world data for robotics' : 'Peelset — Data explorer'; }, []);
  useEffect(() => { try { localStorage.setItem(storageKey, JSON.stringify(saved)); } catch { /* Optional browser storage. */ } }, [saved]);
  useEffect(() => {
    try { if (compiled) sessionStorage.setItem('peelset:compiled', JSON.stringify(compiled)); else sessionStorage.removeItem('peelset:compiled'); } catch { /* Optional draft persistence. */ }
  }, [compiled]);
  useEffect(() => {
    if (window.location.pathname === '/' || currentPage !== 'explorer') return;
    const controller = new AbortController();
    setLoading(true); setError(false);
    window.history.replaceState(null, '', `/explore${queryString ? `?${queryString}` : ''}`);
    postJSON(`/api/dataset-preview?limit=${settings.page_size}&offset=${page * settings.page_size}`, JSON.parse(previewKey), controller.signal)
      .then(data => setResult(previous => ({ ...data, clips: page ? [...new Map([...previous.clips, ...data.clips].map(clip => [clip.id, clip])).values()] : data.clips })))
      .catch(error => { if (error.name !== 'AbortError') setError(true); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [previewKey, queryString, page, retry]);
  useEffect(() => {
    if (currentPage !== 'collection') return;
    const controller = new AbortController();
    setLoading(true); setError(false);
    const ids = new URLSearchParams();
    saved.forEach(id => ids.append('ids', id));
    getJSON(`/api/collection?${ids}`, controller.signal)
      .then(data => setCollection(data.clips))
      .catch(error => { if (error.name !== 'AbortError') setError(true); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [savedIds, retry]);

  function toggleSaved(id) { setSaved(values => values.includes(id) ? values.filter(value => value !== id) : [...values, id]); }
  function toggleExcluded(id) { setExcluded(values => values.includes(id) ? values.filter(value => value !== id) : [...values, id]); }
  function toggleFilter(key, value) {
    setPage(0);
    setFilters(previous => ({ ...previous, [key]: previous[key].includes(value) ? previous[key].filter(item => item !== value) : [...previous[key], value] }));
  }
  function clearInstructions() { setQuery(''); setCompiled(null); setExcluded([]); setCompileError(''); setPage(0); }
  function reset() {
    setFilters(Object.fromEntries(filterGroups.map(group => [group.key, []])));
    setVariant(''); clearInstructions();
  }
  async function submit(value) {
    if (!value.trim() || compiling) return;
    setQuery(value); setCompiling(true); setCompileError('');
    try {
      const data = await postJSON('/api/compile', { prompt: value.trim() });
      setCompiled(data); setExcluded([]); setPage(0);
    } catch (error) { setCompileError(error.message); }
    finally { setCompiling(false); }
  }
  if (window.location.pathname === '/') return <HomePage />;
  const isCollection = currentPage === 'collection';
  const clips = isCollection ? collection : result.clips;

  return <div className="app-shell">
    <aside className={`sidebar ${mobileFilters ? 'mobile-open' : ''}`}>
      <a className="brand" href="/"><Mark />{site.brand}<span className="brand-period">.</span></a>
      <button className="mobile-close icon-button" aria-label="Close filters" onClick={() => setMobileFilters(false)}><X size={19} /></button>
      <div className="workspace-label">RESEARCH WORKSPACE</div>
      <nav className="main-nav" aria-label="RESEARCH WORKSPACE">
        <a href="/explore" className={!isCollection ? 'active' : ''} aria-current={!isCollection ? 'page' : undefined}><LayoutGrid size={17} />Data explorer<ArrowUpRight className="nav-arrow" size={15} /></a>
        <a href="/collection" className={isCollection ? 'active' : ''} aria-current={isCollection ? 'page' : undefined}><Bookmark size={17} />My collection<span className="nav-count">{saved.length}</span></a>
      </nav>
      {!isCollection && <div className="filters">
        <div className="filter-heading"><h2>Filters</h2><button onClick={reset}>Reset all</button></div>
        {filterGroups.map(group => <fieldset key={group.key}><legend>{group.label}</legend>{group.options.map(option => <label className="checkbox-label" key={option}><input type="checkbox" checked={filters[group.key].includes(option)} onChange={() => toggleFilter(group.key, option)} /><span>{option}</span></label>)}</fieldset>)}
      </div>}
      <div className="sidebar-footer"><span className="small-mark"><Mark /></span><p>Built for embodied intelligence.</p><a href="/"><ArrowLeft size={13} />Back to homepage</a></div>
    </aside>
    {mobileFilters && <button className="sidebar-shade" aria-label="Close filters" onClick={() => setMobileFilters(false)} />}
    <div className="main-shell">
      <button className="mobile-menu workspace-menu icon-button" aria-label="Filters" onClick={() => setMobileFilters(true)}><SlidersHorizontal size={19} /></button>
      <main className="main-content">
        <section className="intro">
          {isCollection && <p className="eyebrow"><span className="eyebrow-line" />YOUR NEXT TRAINING SET</p>}
          <h1>{isCollection ? 'A little less searching. More building.' : 'The world is your training set.'}</h1>
          {isCollection ? <p className="intro-description">Keep interesting clips together, then export the metadata to share with your team.</p> : <>
            <form className="search-form ai-search-form" onSubmit={event => { event.preventDefault(); submit(query); }} role="search">
              <div className="prompt-input"><Sparkles size={20} /><textarea aria-label="Describe your dataset" placeholder="Tell us what you’re training. For example: I need hand-object interactions for a banana plantation robot. Adapt the manipulated objects to bananas." rows={3} maxLength={3000} required value={query} onChange={event => setQuery(event.target.value)} />{query && <button type="button" className="clear-search icon-button" aria-label="Clear instructions" disabled={compiling} onClick={clearInstructions}><X size={16} /></button>}</div>
              <div className="prompt-footer"><span>Describe the tasks, objects, and changes you need.</span><button className="primary-button" type="submit" disabled={compiling || !query.trim()}>{compiling ? 'Compiling…' : 'Compile dataset'}<ArrowRight size={15} /></button></div>
            </form>
            <div className="suggestions"><span>Try</span>{['All hand-object interactions', 'Move objects in a banana plantation', 'Tool use in low light'].map(value => <button key={value} disabled={compiling} onClick={() => submit(value)}>{value}<ArrowUpRight size={12} /></button>)}</div>
            {compileError && <p className="dataset-error" role="alert">{compileError}</p>}
            {compiled && query.trim() !== compiled.prompt.trim() && <p className="dataset-error">Your instructions have changed. Compile again to update the selection.</p>}
          </>}
        </section>
        {!isCollection && compiled && <DatasetPlan compilation={compiled} selection={selection} count={result.selected_count} disabled={loading || compiling || !!compileError || query.trim() !== compiled.prompt.trim() || error} onChange={plan => setCompiled(previous => ({ ...previous, plan }))} onSelectAll={() => setExcluded([])} />}
        {!isCollection && <div className="variant-tabs" role="group" aria-label="Available versions">{[['', 'All footage'], ['source', 'Source footage'], ['upscaled', 'Upscaled'], ['augmented', 'Augmented']].map(([value, label]) => <button key={value} className={variant === value ? 'selected' : ''} aria-pressed={variant === value} onClick={() => { setVariant(value); setPage(0); }}>{value === 'upscaled' || value === 'augmented' ? <Sparkles size={14} /> : null}{label}{!value && <span>{result.catalog_total}</span>}</button>)}</div>}
        <div className="results-toolbar">
          <p aria-live="polite"><strong>{isCollection ? collection.length : compiled ? result.selected_count : result.total}</strong> {isCollection ? 'clips in your collection' : compiled ? 'source clips selected' : <>clips<span className="result-suffix"> ready to explore</span></>}</p>
          <div className="toolbar-actions">
            {isCollection ? <button className="secondary-button export-button" disabled={!collection.length || loading || error} onClick={() => download(collection, site.collection_export_filename)}><Download size={14} />Export metadata</button> : <label className="sort-label"><span>Sort by</span><select value={sort} onChange={event => { setSort(event.target.value); setPage(0); }}><option value="relevance">Relevance</option><option value="newest">Newest first</option><option value="duration">Longest first</option></select><ChevronDown size={13} /></label>}
            <div className="layout-switch"><button aria-label="Grid view" aria-pressed={layout === 'grid'} className={layout === 'grid' ? 'selected' : ''} onClick={() => setLayout('grid')}><LayoutGrid size={16} /></button><button aria-label="List view" aria-pressed={layout === 'list'} className={layout === 'list' ? 'selected' : ''} onClick={() => setLayout('list')}><List size={17} /></button></div>
          </div>
        </div>
        {!!activeFilters.length && !isCollection && <div className="active-filters" aria-label="Active filters">{activeFilters.map(({ key, value }) => <button key={`${key}-${value}`} onClick={() => toggleFilter(key, value)}>{value}<X size={12} /></button>)}</div>}
        {loading && !clips.length ? <div className="state-message" role="status"><span className="loader" /><p>{isCollection ? 'Loading your collection…' : 'Finding your next dataset…'}</p></div> : error ? <div className="state-message" role="alert"><p>We couldn’t load this selection. Check that the Python backend is running.</p><button className="secondary-button" onClick={() => setRetry(value => value + 1)}>Try again</button></div> : clips.length ? <>
          <div className={`clip-grid ${layout === 'list' ? 'list-layout' : ''}`} aria-busy={loading}>{clips.map(clip => <ClipCard key={clip.id} clip={clip} saved={saved.includes(clip.id)} onToggle={toggleSaved} onOpen={setActiveClip} selectionMode={!isCollection && !!compiled} included={!excluded.includes(clip.id)} onSelect={toggleExcluded} />)}</div>
          {!isCollection && result.total > clips.length && <div className="load-more"><button className="secondary-button" disabled={loading} onClick={() => setPage(value => value + 1)}>Load more clips<ArrowRight size={15} /></button></div>}
          {isCollection && collection.length < saved.length && <p className="catalog-note">Some saved clips are no longer in the catalog.</p>}
        </> : <div className="state-message empty-state"><Bookmark size={29} strokeWidth={1.4} /><h2>{isCollection ? 'Your collection starts here' : 'No clips found'}</h2><p>{isCollection ? 'Save clips from the explorer to put together your next dataset.' : 'Try a broader request or clear a few filters.'}</p>{isCollection ? <a className="primary-button" href="/explore">Explore the catalog<ArrowRight size={15} /></a> : <button className="secondary-button" onClick={reset}>Clear filters</button>}</div>}
        <footer className="catalog-note"><span />Sample catalog · Illustrative stills and metadata</footer>
      </main>
    </div>
    {activeClip && <ClipDialog clip={activeClip} saved={saved.includes(activeClip.id)} onToggle={toggleSaved} onClose={() => setActiveClip(null)} />}
  </div>;
}
