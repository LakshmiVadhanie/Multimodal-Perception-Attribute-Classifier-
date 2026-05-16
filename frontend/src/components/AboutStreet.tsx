import { motion } from 'framer-motion';
import { Image as ImageIcon, ArrowRight, Wrench, Brain, Target, BarChart2, Car, Radio, Package, Bot, Tag, Zap, TrendingUp, Cloud, Code, Info } from 'lucide-react';

const ARCH_NODES = [
  { icon: ImageIcon, label: 'Image' },
  { icon: ArrowRight, label: '', isArrow: true },
  { icon: Wrench, label: 'Preprocess\n224×224' },
  { icon: ArrowRight, label: '', isArrow: true },
  { icon: Brain, label: 'ViT-B/16\nBackbone' },
  { icon: ArrowRight, label: '', isArrow: true },
  { icon: Target, label: 'Multi-Head\nClassifier' },
  { icon: ArrowRight, label: '', isArrow: true },
  { icon: BarChart2, label: '8 Attribute\nPredictions' },
];

const DATASETS = [
  { name: 'BDD100K', desc: '100K driving videos from Berkeley', icon: Car, link: 'https://bdd-data.berkeley.edu/' },
  { name: 'nuScenes', desc: 'Full sensor suite — 1000 scenes', icon: Radio, link: 'https://www.nuscenes.org/' },
  { name: 'COCO', desc: 'Common objects in context', icon: Package, link: 'https://cocodataset.org/' },
];

const MODEL_CARDS = [
  { icon: Bot, title: 'ViT-B/16', desc: 'Vision Transformer base, pre-trained on ImageNet-21K. 86M parameters.' },
  { icon: Tag, title: 'BLIP-2 / LLaVA', desc: 'VLM auto-labeler for ontology-guided annotation of unlabeled images.' },
  { icon: Zap, title: 'FastAPI', desc: 'Async REST API with image upload, validation, and JSON responses.' },
  { icon: TrendingUp, title: 'MLflow', desc: 'Experiment tracking — logs metrics, checkpoints, and artifact versions.' },
  { icon: Cloud, title: 'AWS S3', desc: 'Artifact storage for datasets, model weights, and evaluation outputs.' },
  { icon: Code, title: 'PyTorch', desc: 'Training framework with LoRA-enabled fine-tuning via PEFT.' },
];

const RESULTS_TABLE = [
  { model: 'ViT-B/16', dataset: 'BDD100K (50K)', acc: '91.2%', f1: '89.4%' },
  { model: 'ViT-L/16', dataset: 'BDD100K (50K)', acc: '92.8%', f1: '91.1%' },
];

export function AboutStreet() {
  return (
    <section id="about" className="about-section">
      {/* Header */}
      <motion.div
        className="text-center"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}
      >
        <h2 className="section-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Info size={40} className="accent-red" /> About <span className="accent-red">Street</span>
        </h2>
        <div className="divider" style={{ background: 'linear-gradient(90deg, var(--accent-red), transparent)' }} />
        <p className="section-subtitle">
          Architecture, datasets, training results, and how the
          auto-labeling pipeline works.
        </p>
      </motion.div>

      {/* Architecture diagram */}
      <motion.div
        className="arch-diagram"
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
      >
        <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 20 }}>
          Model Architecture
        </p>
        <div className="arch-flow">
          {ARCH_NODES.map((n, i) => {
            const Icon = n.icon;
            return n.isArrow ? (
              <span key={i} className="arch-arrow"><Icon size={20} /></span>
            ) : (
              <motion.div
                key={i}
                className="arch-node"
                initial={{ opacity: 0, scale: 0.8 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
              >
                <div className="arch-node-icon"><Icon size={24} /></div>
                <span className="arch-node-label">{n.label}</span>
              </motion.div>
            )
          })}
        </div>
      </motion.div>

      {/* Tech stack cards */}
      <div style={{ width: '100%', maxWidth: 1100 }}>
        <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 16 }}>
          Tech Stack
        </p>
        <div className="about-grid">
          {MODEL_CARDS.map((c, i) => {
            const Icon = c.icon;
            return (
              <motion.div
                key={c.title}
                className="about-card"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.07 }}
              >
                <div className="about-card-icon"><Icon size={32} /></div>
                <div className="about-card-title">{c.title}</div>
                <div className="about-card-desc">{c.desc}</div>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Results table */}
      <motion.div
        className="arch-diagram"
        style={{ maxWidth: 700 }}
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
      >
        <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 16 }}>
          Benchmark Results
        </p>
        <table className="results-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Dataset</th>
              <th>Accuracy</th>
              <th>F1 (macro)</th>
            </tr>
          </thead>
          <tbody>
            {RESULTS_TABLE.map(r => (
              <tr key={r.model}>
                <td>{r.model}</td>
                <td>{r.dataset}</td>
                <td className="highlight">{r.acc}</td>
                <td className="highlight">{r.f1}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p style={{ marginTop: 12, fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
          Auto-labeling pipeline reduces manual annotation by <strong style={{ color: 'var(--accent-green)' }}>~60%</strong> across 8 categories.
        </p>
      </motion.div>

      {/* Datasets */}
      <div style={{ width: '100%', maxWidth: 1100 }}>
        <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 16 }}>
          Supported Datasets
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 'var(--space-md)' }}>
          {DATASETS.map((d, i) => {
            const Icon = d.icon;
            return (
              <motion.a
                key={d.name}
                href={d.link}
                target="_blank"
                rel="noopener noreferrer"
                className="about-card"
                style={{ textDecoration: 'none', cursor: 'pointer' }}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                whileHover={{ y: -4 }}
              >
                <div className="about-card-icon"><Icon size={32} /></div>
                <div className="about-card-title" style={{ color: 'var(--accent-cyan)' }}>{d.name} ↗</div>
                <div className="about-card-desc">{d.desc}</div>
              </motion.a>
            )
          })}
        </div>
      </div>
    </section>
  );
}
