const API_BASE = '/api';

export interface DatasetItem {
  id: string;
  name: string;
  version: string;
  modality: string;
  label_mode: string;
  source: string;
  total_samples: number;
  train_count: number;
  val_count: number;
  test_count: number;
  created_at: string;
}

export interface SampleItem {
  id: string;
  dataset_id: string;
  external_sample_id: string;
  text: string;
  label: string | null;
  label_status: string;
  split: string;
  state: 'ACTIVE' | 'QUARANTINED' | 'RESTORED';
  risk_score?: number;
  risk_level?: 'LOW' | 'MEDIUM' | 'HIGH';
}

export interface DatasetDetail extends DatasetItem {
  samples: SampleItem[];
}

export interface ScanItem {
  id: string;
  dataset_id: string;
  name: string;
  detector: string;
  model_version: string;
  status: 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED';
  error_message?: string | null;
  threshold?: number | null;
  started_at: string;
  completed_at?: string | null;
  metrics: Record<string, number>;
}

export interface ScanSampleItem {
  sample_id: string;
  external_sample_id: string;
  text: string;
  label: string | null;
  label_status: string;
  split: string;
  state: 'ACTIVE' | 'QUARANTINED' | 'RESTORED';
  raw_score: number;
  normalized_score: number;
  risk_score: number;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
  dominant_evidence: string;
  evidence: {
    layer_attributions?: Record<string, number>;
    dominant_layer?: number;
    trajectory?: string;
    token_attributions?: Array<{
      token: string;
      start_char: number;
      end_char: number;
      saliency_score: number;
      is_suspicious_span: boolean;
    }>;
    evidence_summary?: string;
  };
}

export interface QuarantineEventItem {
  id: string;
  sample_id: string;
  action: string;
  previous_state: string;
  new_state: string;
  reason: string;
  timestamp: string;
}

export interface SampleInvestigation {
  id: string;
  dataset_id: string;
  external_sample_id: string;
  text: string;
  label: string | null;
  label_status: string;
  split: string;
  state: 'ACTIVE' | 'QUARANTINED' | 'RESTORED';
  risk_score: number;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
  dominant_evidence: string;
  token_attributions: Array<{
    token: string;
    start_char: number;
    end_char: number;
    saliency_score: number;
    is_suspicious_span: boolean;
  }>;
  layer_scores: Record<string, number>;
  layer_attributions: Record<string, number>;
  dominant_layer: number;
  trajectory: string;
  evidence_summary: string;
  quarantine_history: QuarantineEventItem[];
}

export interface PurificationPreview {
  dataset_id: string;
  total_original_samples: number;
  projected_active_count: number;
  projected_quarantined_count: number;
  projected_restored_count: number;
  risk_threshold: number;
}

export interface PurificationResponse {
  original_dataset_id: string;
  original_dataset_version: string;
  purified_dataset_id: string;
  purified_dataset_version: string;
  total_original_samples: number;
  active_count: number;
  quarantined_count: number;
  restored_count: number;
  artifact_uri: string;
  created_at: string;
}

export interface RetrainingResponse {
  id: string;
  raw_dataset_id: string;
  purified_dataset_id: string;
  status: string;
  raw_clean_accuracy: number;
  raw_attack_success_rate: number;
  purified_clean_accuracy: number;
  purified_attack_success_rate: number;
  ca_delta: number;
  asr_reduction: number;
  quarantined_samples_count: number;
  completed_at: string;
}

export interface OverviewStats {
  total_datasets: number;
  total_samples: number;
  total_quarantined: number;
  total_restored: number;
  total_scans: number;
  completed_scans: number;
  running_scans: number;
  avg_auroc: number | null;
  avg_precision: number | null;
  avg_recall: number | null;
  avg_f1: number | null;
  recent_scans: Array<{
    id: string;
    dataset_id: string;
    name: string;
    detector: string;
    status: string;
    started_at: string | null;
    completed_at: string | null;
    metrics: Record<string, number>;
  }>;
  recent_datasets: Array<{
    id: string;
    name: string;
    version: string;
    modality: string;
    total_samples: number;
    train_count: number;
    val_count: number;
    test_count: number;
    created_at: string | null;
  }>;
}

// API Functions

export async function fetchHealth(): Promise<{ status: string }> {
  const res = await fetch('/health');
  if (!res.ok) throw new Error('Healthcheck failed');
  return res.json();
}

export async function fetchOverviewStats(): Promise<OverviewStats> {
  const res = await fetch(`${API_BASE}/stats/overview`);
  if (!res.ok) throw new Error('Failed to fetch overview stats');
  return res.json();
}

export async function fetchDatasets(): Promise<DatasetItem[]> {
  const res = await fetch(`${API_BASE}/datasets`);
  if (!res.ok) throw new Error('Failed to fetch datasets');
  return res.json();
}

export async function fetchDatasetDetail(id: string): Promise<DatasetDetail> {
  const res = await fetch(`${API_BASE}/datasets/${id}`);
  if (!res.ok) throw new Error(`Failed to fetch dataset ${id}`);
  return res.json();
}

export async function uploadDatasetFile(file: File, name?: string): Promise<DatasetItem> {
  const formData = new FormData();
  formData.append('file', file);
  if (name) formData.append('name', name);

  const res = await fetch(`${API_BASE}/datasets`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(err.detail || 'Upload failed');
  }
  return res.json();
}

export async function fetchScans(): Promise<ScanItem[]> {
  const res = await fetch(`${API_BASE}/scans`);
  if (!res.ok) throw new Error('Failed to fetch scans');
  return res.json();
}

export async function fetchScan(id: string): Promise<ScanItem> {
  const res = await fetch(`${API_BASE}/scans/${id}`);
  if (!res.ok) throw new Error(`Failed to fetch scan ${id}`);
  return res.json();
}

export async function createScan(payload: {
  dataset_id: string;
  name?: string;
  detector: 'FLARE' | 'ISOLATION_FOREST' | 'KMEANS';
  layers?: number[];
  threshold?: number;
  seed?: number;
}): Promise<ScanItem> {
  const res = await fetch(`${API_BASE}/scans`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to create scan' }));
    throw new Error(err.detail || 'Failed to create scan');
  }
  return res.json();
}

export async function fetchScanSamples(scanId: string, riskLevel?: string): Promise<ScanSampleItem[]> {
  const url = riskLevel
    ? `${API_BASE}/scans/${scanId}/samples?risk_level=${riskLevel}`
    : `${API_BASE}/scans/${scanId}/samples`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch scan samples');
  return res.json();
}

export async function fetchSampleInvestigation(sampleId: string): Promise<SampleInvestigation> {
  const res = await fetch(`${API_BASE}/samples/${sampleId}`);
  if (!res.ok) throw new Error(`Failed to fetch sample ${sampleId}`);
  return res.json();
}

export async function quarantineSample(sampleId: string, reason: string): Promise<SampleInvestigation> {
  const res = await fetch(`${API_BASE}/samples/${sampleId}/quarantine`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ reason }),
  });
  if (!res.ok) throw new Error('Failed to quarantine sample');
  return res.json();
}

export async function restoreSample(sampleId: string, reason: string): Promise<SampleInvestigation> {
  const res = await fetch(`${API_BASE}/samples/${sampleId}/restore`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ reason }),
  });
  if (!res.ok) throw new Error('Failed to restore sample');
  return res.json();
}

export async function fetchPurificationPreview(
  datasetId: string,
  riskThreshold: number,
  experimentId?: string
): Promise<PurificationPreview> {
  const params = new URLSearchParams({
    dataset_id: datasetId,
    risk_threshold: String(riskThreshold),
  });
  if (experimentId) params.append('experiment_id', experimentId);

  const res = await fetch(`${API_BASE}/purification/preview?${params.toString()}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to fetch preview' }));
    throw new Error(err.detail || 'Failed to fetch preview');
  }
  return res.json();
}

export async function triggerPurification(payload: {
  dataset_id: string;
  experiment_id?: string;
  risk_threshold?: number;
  version_suffix?: string;
}): Promise<PurificationResponse> {
  const res = await fetch(`${API_BASE}/purification`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Purification failed' }));
    throw new Error(err.detail || 'Purification failed');
  }
  return res.json();
}

export async function triggerRetraining(payload: {
  raw_dataset_id: string;
  purified_dataset_id: string;
  target_label?: string;
}): Promise<RetrainingResponse> {
  const res = await fetch(`${API_BASE}/retraining`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Retraining benchmark failed' }));
    throw new Error(err.detail || 'Retraining benchmark failed');
  }
  return res.json();
}
