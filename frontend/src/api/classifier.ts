// API client for the FastAPI v2 Production Integration

export interface Alert {
  id: string;
  name: string;
  severity: string;
}

export interface Detection {
  road_user_type: string;
  confidence: number;
  bbox: number[];
  attributes: Record<string, string>;
  alerts: Alert[];
}

export interface ImageAnalysisResponse {
  source: string;
  detections: Detection[];
  total_alerts: number;
}

export interface TrackSummary {
  track_id: number;
  road_user_type: string;
  age_frames: number;
  last_attributes: Record<string, string>;
  last_bbox: number[];
}

export interface VideoAnalysisResult {
  source: string;
  total_frames_processed: number;
  total_detections: number;
  total_classifications: number;
  total_alerts: number;
  alerts: Alert[];
  track_summaries: TrackSummary[];
  duration_s: number;
}

export interface JobResponse {
  job_id: string;
  status: 'queued' | 'running' | 'complete' | 'failed';
  result?: VideoAnalysisResult;
  error?: string;
}

const BASE = '/api';

export async function checkHealth() {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error('Backend not reachable');
  return res.json();
}

/** Analyzes a single image returning all bounding box detections and alerts. */
export async function analyzeImage(file: File): Promise<ImageAnalysisResponse> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE}/analyze/image`, { method: 'POST', body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(err.detail ?? 'Image analysis failed');
  }
  return res.json();
}

/** Starts an async video analysis job and returns the job queue locator. */
export async function analyzeVideo(file: File): Promise<{ job_id: string }> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE}/analyze/video`, { method: 'POST', body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(err.detail ?? 'Video upload failed');
  }
  return res.json();
}

/** Polls the async video job until completion. */
export async function pollVideoJob(jobId: string, onProgress?: (status: string) => void): Promise<VideoAnalysisResult> {
  return new Promise((resolve, reject) => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${BASE}/jobs/${jobId}`);
        if (!res.ok) throw new Error('Failed to query job status');
        const data: JobResponse = await res.json();
        
        if (onProgress) onProgress(data.status);

        if (data.status === 'complete') {
          clearInterval(interval);
          resolve(data.result!);
        } else if (data.status === 'failed') {
          clearInterval(interval);
          reject(new Error(data.error ?? 'Video Processing Failed'));
        }
      } catch (err: any) {
        clearInterval(interval);
        reject(err);
      }
    }, 1500); // poll every 1.5s
  });
}
