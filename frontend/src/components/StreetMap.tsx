import { motion } from 'framer-motion';
import { Building2, Camera, BarChart3, Map as MapIcon, Info, MapPin, CarFront, Compass } from 'lucide-react';

interface Props {
  onNavigate: (section: string) => void;
  active: string;
}

const blocks = [
  { id: 'hero',     label: 'City\nOverview', x: 40,  y: 40,  w: 180, h: 100, icon: Building2, color: '#00d4ff' },
  { id: 'upload',   label: 'Upload\nLab',     x: 290, y: 40,  w: 160, h: 100, icon: Camera, color: '#a855f7' },
  { id: 'results',  label: 'Results\nBlvd',   x: 520, y: 40,  w: 160, h: 100, icon: BarChart3, color: '#22d3a0' },
  { id: 'map',      label: 'Ontology\nMap',   x: 40,  y: 200, w: 210, h: 100, icon: MapIcon, color: '#f7c948' },
  { id: 'about',    label: 'About\nStreet',   x: 330, y: 200, w: 160, h: 100, icon: Info, color: '#ff5e5b' },
];

const roads = [
  // horizontal main road
  { x1: 40, y1: 160, x2: 700, y2: 160 },
  // vertical connector
  { x1: 240, y1: 40, x2: 240, y2: 310 },
  { x1: 460, y1: 40, x2: 460, y2: 160 },
  // bottom horizontal
  { x1: 40, y1: 310, x2: 520, y2: 310 },
];

export function StreetMap({ onNavigate, active }: Props) {
  return (
    <div className="street-map-section" style={{ background: 'rgba(13, 20, 33, 0.4)', backdropFilter: 'blur(10px)' }}>
      <p className="street-map-label" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <MapPin size={12} /> Navigate the city
      </p>
      <div className="street-map-container">
        <svg viewBox="0 0 740 330" className="street-map-svg" aria-label="Street navigation map">
          {/* Background */}
          <rect x="0" y="0" width="740" height="330" fill="rgba(13, 20, 33, 0.6)" rx="16" />

          {/* Grid lines */}
          {Array.from({ length: 12 }).map((_, i) => (
            <line key={`h${i}`} x1="0" y1={i * 30} x2="740" y2={i * 30}
              stroke="rgba(0,212,255,0.04)" strokeWidth="1" />
          ))}
          {Array.from({ length: 26 }).map((_, i) => (
            <line key={`v${i}`} x1={i * 30} y1="0" x2={i * 30} y2="330"
              stroke="rgba(0,212,255,0.04)" strokeWidth="1" />
          ))}

          {/* Roads */}
          {roads.map((r, i) => (
            <g key={i}>
              <line x1={r.x1} y1={r.y1} x2={r.x2} y2={r.y2}
                stroke="#1a1e2a" strokeWidth="18" />
              {/* Dashed center line */}
              <line x1={r.x1} y1={r.y1} x2={r.x2} y2={r.y2}
                stroke="rgba(247,201,72,0.4)" strokeWidth="1.5"
                strokeDasharray="12 10" />
            </g>
          ))}

          {/* Intersection dots */}
          {[
            [240, 160], [460, 160], [240, 310],
          ].map(([cx, cy], i) => (
            <circle key={i} cx={cx} cy={cy} r="5"
              fill="#1a1e2a" stroke="rgba(0,212,255,0.3)" strokeWidth="1.5" />
          ))}

          {/* Blocks */}
          {blocks.map((b) => {
            const Icon = b.icon;
            return (
              <motion.g
                key={b.id}
                className={`street-block ${active === b.id ? 'active' : ''}`}
                onClick={() => onNavigate(b.id)}
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
                style={{ cursor: 'pointer' }}
              >
                {/* Shadow */}
                <rect x={b.x + 3} y={b.y + 3} width={b.w} height={b.h}
                  rx="10" fill="rgba(0,0,0,0.4)" />
                {/* Block background */}
                <motion.rect
                  className="street-block-bg"
                  x={b.x} y={b.y} width={b.w} height={b.h} rx="10"
                  fill={active === b.id ? `${b.color}20` : 'rgba(17,24,39,0.9)'}
                  animate={{ fill: active === b.id ? `${b.color}20` : 'rgba(17,24,39,0.9)' }}
                />
                {/* Border */}
                <rect className="street-block-border"
                  x={b.x} y={b.y} width={b.w} height={b.h} rx="10"
                  fill="none"
                  stroke={active === b.id ? b.color : 'rgba(255,255,255,0.08)'} // active stroke doesn't seem to animate perfectly here 
                  strokeWidth={active === b.id ? 1.5 : 1}
                />
                {/* Active glow */}
                {active === b.id && (
                  <rect x={b.x} y={b.y} width={b.w} height={b.h} rx="10"
                    fill="none"
                    stroke={b.color}
                    strokeWidth="0.5"
                    filter={`drop-shadow(0 0 8px ${b.color})`}
                  />
                )}
                
                {/* Lucide SVG Icon */}
                <g transform={`translate(${b.x + 16}, ${b.y + 16})`} style={{ color: active === b.id ? b.color : '#8ca0bc' }}>
                  <Icon width={24} height={24} />
                </g>

                {/* Label */}
                {b.label.split('\n').map((line, li) => (
                  <text key={li}
                    x={b.x + 16} y={b.y + 60 + li * 16}
                    fontSize="11.5"
                    fontFamily="'Space Grotesk', sans-serif"
                    fontWeight="600"
                    fill={active === b.id ? b.color : '#8ca0bc'}
                  >
                    {line}
                  </text>
                ))}
                {/* Active indicator */}
                {active === b.id && (
                  <circle cx={b.x + b.w - 12} cy={b.y + 12} r="4" fill={b.color}>
                    <animate attributeName="opacity" values="1;0.3;1" dur="1.5s" repeatCount="indefinite" />
                  </circle>
                )}
              </motion.g>
            )
          })}

          {/* Animated tiny car */}
          <motion.g
            animate={{ x: [40, 680, 40] }}
            transition={{ duration: 12, repeat: Infinity, ease: 'linear' }}
            transform="translate(0, 146)"
          >
            <CarFront width={22} height={22} color="var(--accent-cyan)" />
          </motion.g>

          {/* Compass rose */}
          <g transform="translate(685, 275)" opacity="0.15" color="#fff">
            <Compass width={32} height={32} />
          </g>
        </svg>
      </div>
    </div>
  );
}
