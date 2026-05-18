import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Compass, EyeOff, Lightbulb, Ruler, UserRound, Users, Eye, Map as MapIcon, Camera, CarFront, PersonStanding, Bike, Clock, Target, ClipboardList, AlertTriangle, PlayCircle } from 'lucide-react';
import type { ImageAnalysisResponse, VideoAnalysisResult, TrackSummary, Detection } from '../api/classifier';

const ATTR_CONFIG: Record<string, { icon: any; color: string; accentVar: string }> = {
  mobility:    { icon: Activity, color: '#00d4ff', accentVar: '--accent-cyan' },
  orientation: { icon: Compass, color: '#a855f7', accentVar: '--accent-purple' },
  occlusion:   { icon: EyeOff, color: '#f7c948', accentVar: '--accent-yellow' },
  lighting:    { icon: Lightbulb, color: '#22d3a0', accentVar: '--accent-green' },
  size:        { icon: Ruler, color: '#00d4ff', accentVar: '--accent-cyan' },
  posture:     { icon: UserRound, color: '#ff5e5b', accentVar: '--accent-red' },
  group:       { icon: Users, color: '#a855f7', accentVar: '--accent-purple' },
  attention:   { icon: Eye, color: '#f7c948', accentVar: '--accent-yellow' },
};

const DISPLAY_NAMES: Record<string, string> = {
  mobility:    'Mobility State',
  orientation: 'Orientation',
  occlusion:   'Occlusion Level',
  lighting:    'Lighting Condition',
  size:        'Apparent Size',
  posture:     'Posture',
  group:       'Group Size',
  attention:   'Attention State',
};

const USER_ICONS: Record<string, any> = { vehicle: CarFront, pedestrian: PersonStanding, cyclist: Bike };

interface Props {
  result: { type: 'image' | 'video'; data: ImageAnalysisResponse | VideoAnalysisResult; previewUrl?: string } | null;
  onNavigate: (section: string) => void;
}

export function ResultsDisplay({ result, onNavigate }: Props) {
  if (!result) {
    return (
      <section id="results" className="results-section">
        <div className="results-empty">
          <div className="results-empty-icon" style={{ color: 'var(--accent-green)', opacity: 0.8 }}><MapIcon size={64} /></div>
          <h2 className="section-title" style={{ fontSize: '2rem' }}>Results <span className="accent-green">Boulevard</span></h2>
          <p style={{ color: 'var(--text-muted)' }}>
            No analysis yet. Upload media in the Vision Lab to see tracking and attribute predictions here.
          </p>
          <button className="btn btn-primary" onClick={() => onNavigate('upload')}>
            <Camera size={16} /> Go to Vision Lab
          </button>
        </div>
      </section>
    );
  }

  const isVideo = result.type === 'video';
  const data = result.data as any;

  // Derive subjects (detections for image, tracks for video)
  const subjects = isVideo ? (data.track_summaries || []) : (data.detections || []);
  const alerts = isVideo ? (data.alerts || []) : ((data.detections || []).flatMap((d: any) => d.alerts || []));

  return (
    <section id="results" className="results-section">
      {/* Header */}
      <motion.div
        className="results-header"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
      >
        <h2 className="section-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Target size={40} className="accent-green" /> Infrastructure Risk <span className="accent-green">Heatmap</span>
        </h2>
        <div className="divider" style={{ background: 'linear-gradient(90deg, var(--accent-green), transparent)' }} />

        <div className="results-meta">
          <div className="result-meta-chip">
            <span className="chip-dot" />
            <ClipboardList size={14} /> {subjects.length} Subjects Tracked
          </div>
          {isVideo && (
            <div className="result-meta-chip">
              <PlayCircle size={14} /> {data.total_frames_processed} Frames Processed
            </div>
          )}
          <div className="result-meta-chip" style={{ borderColor: alerts.length > 0 ? 'var(--accent-red)' : undefined }}>
            <AlertTriangle size={14} color={alerts.length > 0 ? 'var(--accent-red)' : undefined} /> {alerts.length} Alert Flags
          </div>
          {isVideo && (
            <div className="result-meta-chip">
              <Clock size={14} /> {data.duration_s}s Processing Time
            </div>
          )}
        </div>
      </motion.div>

      {/* Alert Feed Widget */}
      {alerts.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          style={{ width: '100%', maxWidth: '1000px', marginBottom: '32px' }}
        >
          <div className="control-card" style={{ borderColor: 'var(--accent-red)' }}>
            <h3 style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--accent-red)', marginBottom: 16 }}>
              <AlertTriangle size={18} /> Shadow-Crash Event Log
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {alerts.map((a: any, idx: number) => (
                <div key={idx} style={{ background: 'rgba(255,94,91,0.1)', padding: '12px 16px', borderRadius: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center', border: '1px solid rgba(255,94,91,0.2)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <span style={{ color: 'var(--accent-red)', fontWeight: 'bold' }}>[ALERT]</span>
                    <span>{a.rule_name || a.name}</span>
                  </div>
                  <span style={{ fontSize: '0.8rem', opacity: 0.7 }}>Rule ID: {a.rule_id || a.id}</span>
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      )}

      {/* Subject Mapping (The original UI adapted for multiple objects) */}
      <div style={{ width: '100%', maxWidth: '1000px', display: 'flex', flexDirection: 'column', gap: 32 }}>
        {subjects.map((subj: any, i: number) => {
          const UserIcon = USER_ICONS[subj.road_user_type] || CarFront;
          const attrs = Object.entries(subj.attributes || subj.last_attributes || {});

          return (
            <motion.div
              key={subj.track_id || i}
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
              className="control-card"
            >
              <h3 style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16, borderBottom: '1px solid var(--border-card)', paddingBottom: 12 }}>
                <UserIcon size={20} /> 
                <span style={{ textTransform: 'capitalize' }}>{subj.road_user_type}</span>
                <span style={{ fontSize: '0.8rem', opacity: 0.5, marginLeft: 'auto' }}>
                  {isVideo ? `Track ID: ${subj.track_id} (Tracked for ${subj.age_frames} frames)` : `Detection Confidence: ${Math.round(subj.confidence * 100)}%`}
                </span>
              </h3>

              <div className="results-grid" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))' }}>
                {attrs.map(([attrName, labelValue]) => {
                  const cfg = ATTR_CONFIG[attrName] ?? { icon: Target, color: '#00d4ff', accentVar: '--accent-cyan' };
                  const displayName = DISPLAY_NAMES[attrName] ?? attrName;
                  const Icon = cfg.icon;

                  return (
                    <div key={attrName} className="attr-card" style={{ '--card-accent': cfg.color, padding: '16px' } as React.CSSProperties}>
                      <div className="attr-card-header" style={{ marginBottom: 8 }}>
                        <div className="attr-name">
                          <div className="attr-icon" style={{ background: `${cfg.color}18`, color: cfg.color, padding: 6 }}>
                            <Icon size={16} />
                          </div>
                          <span className="attr-title" style={{ fontSize: '0.85rem' }}>{displayName}</span>
                        </div>
                      </div>
                      <div className="attr-prediction" style={{ marginTop: 0 }}>
                        <span className="attr-label-badge" style={{ background: `${cfg.color}15`, color: cfg.color, borderColor: `${cfg.color}40`, fontSize: '0.8rem' }}>
                          {(labelValue as string).replace(/_/g, ' ')}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </motion.div>
          );
        })}
      </div>

      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.8 }} style={{ marginTop: 40 }}>
        <button className="btn btn-secondary" onClick={() => onNavigate('upload')}>
          ← Analyze Another Scene
        </button>
      </motion.div>
    </section>
  );
}
