import { useCallback, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { analyzeImage, analyzeVideo, pollVideoJob } from '../api/classifier';
import { Camera, Zap, FolderSearch, Loader, Microscope, Target, Info, Search, Brain, Tag, BarChart as BarChartIcon, AlertTriangle, RefreshCcw, UploadCloud, Video } from 'lucide-react';

interface Props {
  onResult: (res: any) => void;
  onNavigate: (section: string) => void;
}

export function UploadLab({ onResult, onNavigate }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [statusText, setStatusText] = useState('Analyzing...');
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const loadFile = (f: File) => {
    setFile(f);
    setError(null);
    const url = URL.createObjectURL(f);
    setPreview(url);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f && (f.type.startsWith('image/') || f.type.startsWith('video/'))) loadFile(f);
    else setError('Please drop a valid image or video file.');
  }, []);

  const onFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) loadFile(f);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setStatusText('Uploading data to engine...');
    try {
      if (file.type.startsWith('image/')) {
        setStatusText('Detecting and classifying objects...');
        const result = await analyzeImage(file);
        onResult({ type: 'image', data: result, previewUrl: preview });
        onNavigate('results');
      } else if (file.type.startsWith('video/')) {
        setStatusText('Uploading video stream...');
        const { job_id } = await analyzeVideo(file);
        
        const res = await pollVideoJob(job_id, (status) => {
          if (status === 'queued') setStatusText('Job queued in pipeline...');
          else if (status === 'running') setStatusText('Tracking subjects and evaluating rule engine...');
        });
        onResult({ type: 'video', data: res });
        onNavigate('results');
      }
    } catch (err: any) {
      setError(err.message ?? 'Classification failed. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  const isVideo = file?.type.startsWith('video/');

  return (
    <section id="upload" className="upload-section">
      {/* Header */}
      <motion.div
        className="text-center"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}
      >
        <h2 className="section-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Camera size={40} className="accent-purple" /> Infrastructure <span className="accent-purple">Ingestor</span>
        </h2>
        <div className="divider" style={{ background: 'linear-gradient(90deg, var(--accent-purple), transparent)' }} />
        <p className="section-subtitle">
          Upload traffic camera feeds or intersection footage. Our VisionZero engine will
          process the paths, extract behavioral attributes, and map shadow-crashes automatically.
        </p>
      </motion.div>

      <motion.div
        className="upload-layout"
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ delay: 0.1 }}
      >
        {/* Drop zone */}
        <div
          className={`dropzone ${dragging ? 'dragging' : ''} ${preview ? 'has-image' : ''}`}
          onClick={() => !preview && inputRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          {preview ? (
            <>
              {isVideo ? (
                <video src={preview} controls className="dropzone-preview" style={{ maxHeight: '300px' }} />
              ) : (
                <img src={preview} alt="Upload preview" className="dropzone-preview" />
              )}
              <div className="dropzone-overlay">
                <button className="btn btn-secondary btn-sm" onClick={e => { e.stopPropagation(); setFile(null); setPreview(null); }}>
                  <RefreshCcw size={14} /> Change Media
                </button>
              </div>
            </>
          ) : (
            <>
              <div className="dropzone-icon">
                <Video size={28} />
              </div>
              <p className="dropzone-text">Drop video or image here</p>
              <p className="dropzone-hint">MP4, MOV, JPEG, PNG supported</p>
              <button className="btn btn-secondary btn-sm" onClick={() => inputRef.current?.click()}>
                <FolderSearch size={14} /> Browse Files
              </button>
            </>
          )}
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp,video/mp4,video/quicktime,video/x-msvideo"
            style={{ display: 'none' }}
            onChange={onFileInput}
            id="image-upload-input"
          />
        </div>

        {/* Controls panel */}
        <div className="upload-controls">
          {/* Analyze */}
          <div className="control-card analyze-btn-wrap">
            <button
              id="analyze-button"
              className="btn btn-primary"
              style={{ width: '100%', marginBottom: 16 }}
              onClick={handleAnalyze}
              disabled={!file || loading}
            >
              {loading ? (
                <><Loader size={16} className="spinner" style={{ animation: 'spin 1s linear infinite' }} /> {statusText}</>
              ) : (
                <><Zap size={16} /> Execute Pipeline</>
              )}
            </button>

            <AnimatePresence>
              {loading && (
                <motion.div
                  className="analyzing-bar"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <div className="analyzing-bar-fill" />
                </motion.div>
              )}
            </AnimatePresence>

            <AnimatePresence>
              {error && (
                <motion.p
                  style={{ color: 'var(--accent-red)', fontSize: '0.875rem', display: 'flex', alignItems: 'center', gap: 6 }}
                  initial={{ opacity: 0, y: -8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                >
                  <AlertTriangle size={14} /> {error}
                </motion.p>
              )}
            </AnimatePresence>

            {!file && (
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textAlign: 'center' }}>
                Upload traffic media to execute the vulnerability pipeline
              </p>
            )}
          </div>

          {/* Description */}
          <div className="control-card">
            <p className="control-label" style={{ display: 'flex', alignItems: 'center', gap: 6 }}><Info size={14} /> Industrial Pipeline Execution</p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {[
                { icon: Target, text: 'YOLOv8 detects & crops road users' },
                { icon: Search, text: 'IoU Tracker maps paths across frames' },
                { icon: Brain, text: 'ViT-B/16 maps 8 multi-attributes' },
                { icon: AlertTriangle, text: 'Rule Engine evaluates risk conditions' },
              ].map((item, i) => {
                const Icon = item.icon;
                return (
                  <div key={i} style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                    <span><Icon size={14} /></span>
                    <span>{item.text}</span>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      </motion.div>
    </section>
  );
}
