import { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Building2, Camera, BarChart3, Map as MapIcon, Info, TrafficCone } from 'lucide-react';
import './index.css';

import { checkHealth } from './api/classifier';

import { HeroCity } from './components/HeroCity';
import { StreetMap } from './components/StreetMap';
import { UploadLab } from './components/UploadLab';
import { ResultsDisplay } from './components/ResultsDisplay';
import { OntologyView } from './components/OntologyView';
import { AboutStreet } from './components/AboutStreet';
import { MapBackground } from './components/MapBackground';

type Section = 'hero' | 'upload' | 'results' | 'map' | 'about';

const NAV_ITEMS: { id: Section; label: string; icon: any }[] = [
  { id: 'hero',   label: 'City', icon: Building2 },
  { id: 'upload', label: 'Upload', icon: Camera },
  { id: 'results',label: 'Results', icon: BarChart3 },
  { id: 'map',    label: 'Ontology', icon: MapIcon },
  { id: 'about',  label: 'About', icon: Info },
];

const PAGE_TRANSITIONS = {
  initial: { opacity: 0, x: 40 },
  animate: { opacity: 1, x: 0 },
  exit:    { opacity: 0, x: -40 },
};

export default function App() {
  const [section, setSection] = useState<Section>('hero');
  const [result, setResult] = useState<any | null>(null);
  const [backendOnline, setBackendOnline] = useState(false);
  const [demoMode, setDemoMode] = useState(false);
  const [checkingBackend, setCheckingBackend] = useState(true);

  // Poll backend health on mount
  useEffect(() => {
    async function probe() {
      try {
        const health = await checkHealth();
        setBackendOnline(true);
        setDemoMode(health.model_loaded === false);
      } catch {
        setBackendOnline(false);
      } finally {
        setCheckingBackend(false);
      }
    }
    probe();
    const interval = setInterval(probe, 30_000);
    return () => clearInterval(interval);
  }, []);

  const navigate = (target: string) => {
    setSection(target as Section);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleResult = (res: any) => {
    setResult(res);
  };

  return (
    <div className="app">
      {/* Global Background Map */}
      <MapBackground />

      {/* Navbar */}
      <nav className="navbar" style={{ background: 'rgba(6, 10, 20, 0.6)', backdropFilter: 'blur(20px)' }}>
        <div className="navbar-logo" onClick={() => navigate('hero')}>
          <div className="navbar-logo-icon" style={{ color: 'var(--accent-red)' }}><TrafficCone size={20} /></div>
          <span className="navbar-logo-text">
            VisionZero<span style={{ color: 'var(--accent-red)' }}>Predictor</span>
          </span>
        </div>

        <div className="navbar-nav">
          {NAV_ITEMS.map(item => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                id={`nav-${item.id}`}
                className={`nav-btn ${section === item.id ? 'active' : ''}`}
                onClick={() => navigate(item.id)}
                style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
              >
                <Icon size={16} />
                {item.label}
              </button>
            );
          })}
        </div>

        <div className={`navbar-badge ${demoMode || !backendOnline ? 'demo' : 'live'}`}>
          <span className="badge-dot" />
          {checkingBackend ? 'Connecting…' : !backendOnline ? 'Offline' : demoMode ? 'Demo Mode' : 'Live'}
        </div>
      </nav>

      {/* Page content with section transitions */}
      <AnimatePresence mode="wait">
        <motion.div
          key={section}
          {...PAGE_TRANSITIONS}
          transition={{ duration: 0.3, ease: 'easeInOut' }}
          style={{ minHeight: '100vh', width: '100%', display: 'flex', flexDirection: 'column' }}
        >
          {section === 'hero' && (
            <HeroCity
              onNavigate={navigate}
              backendOnline={backendOnline}
              demoMode={demoMode}
            />
          )}
          {section === 'upload' && (
            <UploadLab onResult={handleResult} onNavigate={navigate} />
          )}
          {section === 'results' && (
            <ResultsDisplay result={result} onNavigate={navigate} />
          )}
          {section === 'map' && (
            <OntologyView />
          )}
          {section === 'about' && (
            <AboutStreet />
          )}
        </motion.div>
      </AnimatePresence>

      {/* Street map nav (shown everywhere) */}
      <StreetMap onNavigate={navigate} active={section} />

      {/* Footer */}
      <footer className="footer" style={{ background: 'rgba(13, 20, 33, 0.7)', backdropFilter: 'blur(10px)' }}>
        <span className="footer-copy">
          © 2026 VisionZero Predictor — Urban Infrastructure & Shadow-Crash API
        </span>
        <div className="footer-links">
          <span className="footer-link" onClick={() => navigate('about')}>Docs</span>
          <span className="footer-link" onClick={() => navigate('map')}>Ontology</span>
          <a
            className="footer-link"
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noopener noreferrer"
          >
            API Docs
          </a>
        </div>
      </footer>
    </div>
  );
}
