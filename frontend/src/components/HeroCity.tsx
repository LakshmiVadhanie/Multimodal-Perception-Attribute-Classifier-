import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Microscope, Map, CarFront, Bus, Bike } from 'lucide-react';

interface Props {
  onNavigate: (section: string) => void;
  backendOnline: boolean;
  demoMode: boolean;
}

const STATS = [
  { value: '1,200', label: 'Trained Intersections' },
  { value: '4', label: 'Risk Heuristics' },
  { value: '8', label: 'Monitored Attributes' },
  { value: '98%', label: 'Incident Recall' },
];

const VEHICLES = [CarFront, Bus, Bike];

export function HeroCity({ onNavigate, backendOnline, demoMode }: Props) {
  const [countedStats, setCountedStats] = useState(STATS.map(() => '0'));
  const hasAnimated = useRef(false);

  useEffect(() => {
    if (hasAnimated.current) return;
    hasAnimated.current = true;
    const targets = [1200, 4, 8, 98];
    const suffixes = ['', '', '', '%'];
    const prefixes = ['', '', '', ''];
    const durations = [1200, 600, 600, 900];

    targets.forEach((target, i) => {
      const start = performance.now();
      const dur = durations[i];
      function tick(now: number) {
        const progress = Math.min((now - start) / dur, 1);
        const val = progress * target;
        const str = i === 0
          ? `${prefixes[i]}${val.toFixed(1)}${suffixes[i]}`
          : `${prefixes[i]}${Math.round(val)}${suffixes[i]}`;
        setCountedStats(prev => { const n = [...prev]; n[i] = str; return n; });
        if (progress < 1) requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
    });
  }, []);

  return (
    <section id="hero" className="hero">
      <div className="hero-bg">
        <div className="hero-grid" />
        <div className="hero-glow-1" />
        <div className="hero-glow-2" />
      </div>

      <motion.div
        className="hero-content"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: 'easeOut' }}
        style={{ background: 'var(--bg-glass)', padding: '40px', borderRadius: '32px', border: '1px solid var(--border-card)', backdropFilter: 'blur(16px)' }}
      >
        <div className="hero-tag">
          <span style={{ width: 6, height: 6, borderRadius: '50%', background: backendOnline ? '#22d3a0' : '#ff5e5b', display: 'inline-block' }} />
          {backendOnline ? (demoMode ? 'Demo Mode Active' : 'Live Inference') : 'Backend Offline'}
        </div>

        <h1 className="hero-title">
          Discover Invisible
          <br />
          <span className="line2" style={{ color: 'var(--accent-red)' }}>Intersection Risks</span>
        </h1>

        <p className="hero-desc">
          Upload any traffic camera or intersection feed to our <strong>VisionZero Inference Engine</strong>. 
          We automatically track vulnerable road users and mathematically identify near-misses, distracted pedestrians, 
          and erratic vehicles before collisions occur.
        </p>

        <div className="hero-stats">
          {STATS.map((s, i) => (
            <motion.div
              key={s.label}
              className="hero-stat"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 + i * 0.1 }}
            >
              <span className="hero-stat-value">{countedStats[i]}</span>
              <span className="hero-stat-label">{s.label}</span>
            </motion.div>
          ))}
        </div>

        <div className="hero-cta">
          <button className="btn btn-primary btn-lg" onClick={() => onNavigate('upload')} style={{ background: 'var(--accent-red)', color: '#fff', borderColor: 'transparent' }}>
            <Microscope size={20} /> Input Traffic Feed
          </button>
          <button className="btn btn-secondary btn-lg" onClick={() => onNavigate('map')}>
            <Map size={20} /> View Risk Taxonomy
          </button>
        </div>
      </motion.div>

      <div className="road-marquee-wrap" style={{ background: 'rgba(26, 30, 42, 0.8)', backdropFilter: 'blur(8px)' }}>
        <div className="road-lanes">
          <div className="road-lane-line" />
          <div className="road-lane-line" />
        </div>
        {VEHICLES.map((IconClass, i) => (
          <motion.div
            key={i}
            className="road-vehicle"
            style={{ position: 'absolute', color: 'var(--accent-cyan)' }}
            animate={{ x: ['-60px', 'calc(100vw + 60px)'] }}
            transition={{
              duration: 6 + i * 2,
              delay: i * 2.5,
              repeat: Infinity,
              ease: 'linear',
            }}
          >
            <IconClass size={32} />
          </motion.div>
        ))}
      </div>
    </section>
  );
}
