import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Activity, Compass, EyeOff, Lightbulb, Ruler, UserRound, Users, Eye, Map, MapPin } from 'lucide-react';

const ONTOLOGY_ENTRIES = [
  {
    name: 'mobility',
    icon: Activity,
    title: 'Mobility State',
    values: ['moving', 'stationary', 'slow_moving'],
    desc: 'Whether the road user is in motion, stopped, or moving slowly.',
    color: '#00d4ff',
    lat: 37.7749, lng: -122.4194, city: 'San Francisco, CA',
  },
  {
    name: 'orientation',
    icon: Compass,
    title: 'Orientation',
    values: ['facing_toward', 'facing_away', 'lateral'],
    desc: 'Direction the road user faces relative to the camera.',
    color: '#a855f7',
    lat: 51.5074, lng: -0.1278, city: 'London, UK',
  },
  {
    name: 'occlusion',
    icon: EyeOff,
    title: 'Occlusion Level',
    values: ['none', 'partial', 'heavy'],
    desc: 'How much of the road user is hidden by other objects.',
    color: '#f7c948',
    lat: 35.6762, lng: 139.6503, city: 'Tokyo, Japan',
  },
  {
    name: 'lighting',
    icon: Lightbulb,
    title: 'Lighting Condition',
    values: ['well_lit', 'low_light', 'backlit'],
    desc: 'Illumination conditions affecting image quality.',
    color: '#22d3a0',
    lat: 48.8566, lng: 2.3522, city: 'Paris, France',
  },
  {
    name: 'size',
    icon: Ruler,
    title: 'Apparent Size',
    values: ['small', 'medium', 'large'],
    desc: 'How large the road user appears relative to the frame.',
    color: '#00d4ff',
    lat: 40.7128, lng: -74.006, city: 'New York, NY',
  },
  {
    name: 'posture',
    icon: UserRound,
    title: 'Posture',
    values: ['upright', 'leaning', 'crouched'],
    desc: 'Body posture — applies to pedestrians and cyclists only.',
    color: '#ff5e5b',
    lat: 52.52, lng: 13.405, city: 'Berlin, Germany',
  },
  {
    name: 'group',
    icon: Users,
    title: 'Group Size',
    values: ['solo', 'pair', 'group'],
    desc: 'Whether the road user is traveling alone or with others.',
    color: '#a855f7',
    lat: -33.8688, lng: 151.2093, city: 'Sydney, Australia',
  },
  {
    name: 'attention',
    icon: Eye,
    title: 'Attention State',
    values: ['attentive', 'distracted', 'phone_use'],
    desc: 'Pedestrian awareness of traffic — phone use detection.',
    color: '#f7c948',
    lat: 1.3521, lng: 103.8198, city: 'Singapore',
  },
];

export function OntologyView() {
  const [selected, setSelected] = useState<string>('mobility');
  
  // Note: the background map will handle moving its center based on CustomEvents
  useEffect(() => {
    const entry = ONTOLOGY_ENTRIES.find(e => e.name === selected);
    if (!entry) return;
    
    // In a real app we might use Context, but custom events are simple and effective here 
    // to command the MapBackground component to fly to the coordinates without re-rendering everything.
    const event = new CustomEvent('FlyToCity', { detail: { lat: entry.lat, lng: entry.lng } });
    window.dispatchEvent(event);
  }, [selected]);

  const selectedEntry = ONTOLOGY_ENTRIES.find(e => e.name === selected)!;
  const SelectedIcon = selectedEntry.icon;

  return (
    <section id="map" className="map-section">
      <motion.div
        className="text-center"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}
      >
        <h2 className="section-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Map size={40} className="accent-yellow" /> Ontology <span className="accent-yellow">Map</span>
        </h2>
        <div className="divider" style={{ background: 'linear-gradient(90deg, var(--accent-yellow), transparent)' }} />
        <p className="section-subtitle">
          Each of the 8 attribute categories is pinned to a city famous for dense road traffic.
          Select a category to explore and fly the background map to that city.
        </p>
      </motion.div>

      <motion.div
        className="map-layout"
        style={{ gridTemplateColumns: 'minmax(320px, 400px)', justifyContent: 'center' }}
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ delay: 0.1 }}
      >
        <div className="ontology-panel">
          {ONTOLOGY_ENTRIES.map(entry => {
            const Icon = entry.icon;
            return (
              <motion.div
                key={entry.name}
                className={`ontology-card ${selected === entry.name ? 'selected' : ''}`}
                onClick={() => setSelected(entry.name)}
                whileHover={{ x: 4 }}
              >
                <div className="ontology-card-header">
                  <div className="ontology-card-icon"><Icon size={20} /></div>
                  <span className="ontology-card-title" style={{ color: selected === entry.name ? entry.color : undefined }}>
                    {entry.title}
                  </span>
                </div>
                {selected === entry.name && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    transition={{ duration: 0.2 }}
                  >
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: 6, marginBottom: 8 }}>
                      {entry.desc}
                    </p>
                    <div className="ontology-values">
                      {entry.values.map(v => (
                        <span
                          key={v}
                          className="ontology-value-badge"
                          style={{ borderColor: `${entry.color}40`, color: entry.color }}
                        >
                          {v.replace(/_/g, ' ')}
                        </span>
                      ))}
                    </div>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 8, fontFamily: 'var(--font-mono)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <MapPin size={12} /> {entry.city}
                    </p>
                  </motion.div>
                )}
              </motion.div>
            )
          })}
        </div>
      </motion.div>

      {/* Selected attribute detail */}
      <motion.div
        key={selected}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        style={{
          background: 'var(--bg-card)',
          border: `1px solid ${selectedEntry.color}30`,
          borderRadius: 'var(--radius-xl)',
          padding: 'var(--space-xl)',
          maxWidth: 700,
          width: '100%',
          textAlign: 'center',
          backdropFilter: 'blur(16px)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 16, color: selectedEntry.color }}>
          <SelectedIcon size={48} />
        </div>
        <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1.5rem', color: selectedEntry.color, marginBottom: 8 }}>
          {selectedEntry.title}
        </h3>
        <p style={{ color: 'var(--text-secondary)', marginBottom: 16 }}>{selectedEntry.desc}</p>
        <div style={{ display: 'flex', gap: 10, justifyContent: 'center', flexWrap: 'wrap' }}>
          {selectedEntry.values.map(v => (
            <span key={v} style={{
              padding: '6px 18px',
              borderRadius: 9999,
              background: `${selectedEntry.color}15`,
              border: `1px solid ${selectedEntry.color}40`,
              color: selectedEntry.color,
              fontFamily: 'var(--font-mono)',
              fontSize: '0.875rem',
              fontWeight: 600,
            }}>
              {v.replace(/_/g, ' ')}
            </span>
          ))}
        </div>
      </motion.div>
    </section>
  );
}
