import { ArrowRight, ArrowUpRight, Banana, Check, GitBranch } from 'lucide-react';
import EngineScene from './EngineScene.jsx';
import site from '../config/site.json';
import './home.css';

const stages = [
  {
    "id": "segment",
    "title": "Find the interaction",
    "description": "Sort the quiet hours away. Recursively crop the useful moments."
  },
  {
    "id": "refine",
    "title": "Refine the detail",
    "description": "Diffusion upscaling brings small details into focus, with the source alongside."
  },
  {
    "id": "label",
    "title": "Label what matters",
    "description": "Objects, poses, actions, and the relationships between them."
  },
  {
    "id": "multiply",
    "title": "Multiply the moment",
    "description": "New objects, lighting, and backgrounds. The same underlying interaction."
  }
];

export default function HomePage() {
  return <div className="landing">
    <header className="landing-header">
      <a className="brand" href="/"><Banana className="brand-mark" strokeWidth={1.6} aria-hidden="true" />{site.brand}<span className="brand-period">.</span></a>
      <nav aria-label="Browse datasets"><a className="primary-button" href="/explore">Find data<ArrowUpRight size={16} /></a></nav>
    </header>
    <main>
      <section className="landing-hero">
        <h1>Real-world footage.<br /><span>A richer training set.</span></h1>
        <p className="landing-description">We turn everyday video into richly labeled data for robotics, VLMs, and VLAs. One real interaction, many more ways to learn.</p>
        <div className="landing-actions"><a className="primary-button" href="/explore">Find your dataset<ArrowRight size={18} /></a><p className="catalog-hint">Describe what you need. We’ll compile the dataset.</p></div>
      </section>
      <section className="engine" id="engine" aria-labelledby="engine-title">
        <div className="engine-heading"><h2 id="engine-title"><Banana size={19} strokeWidth={1.6} />The banana engine</h2><span>A LITTLE REFINEMENT GOES A LONG WAY</span></div>
        <figure className="engine-figure" aria-label="The banana engine: segment real footage into useful interactions, upscale the detail, label objects and actions, then create variations in lighting, objects, and backgrounds.">
          <ol className="engine-stages">{stages.map((stage, index) => <li key={stage.id}>
            <div className="engine-visual"><EngineScene stage={stage.id} />{index < stages.length - 1 && <ArrowRight className="engine-connector" size={19} strokeWidth={1.3} aria-hidden="true" />}</div>
            <div className="stage-caption"><span className="stage-index">{String(index + 1).padStart(2, '0')}</span><h3>{stage.title}</h3><p>{stage.description}</p></div>
          </li>)}</ol>
          <figcaption><span className="engine-caption"><GitBranch size={15} />One real moment. A whole bunch of possibilities.</span><span className="source-note"><Check size={13} />Source footage stays linked to every generated version.</span></figcaption>
        </figure>
      </section>
    </main>
    <footer className="landing-footer"><span>Built for embodied intelligence.</span><span>{site.domain}</span></footer>
  </div>;
}
