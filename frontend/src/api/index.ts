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

// -------------------------------------------------------------
// Live Real-Time Investigation Console Interfaces & API
// -------------------------------------------------------------

export interface LiveInvestigationRequest {
  dataset_id: string;
  attack_type: 'none' | 'rare_word' | 'common_word' | 'sentence_trigger' | 'syntactic_style' | 'semantic_trigger' | 'character_perturbation' | 'text_backdoor_v1';
  poison_rate: number;
  target_label: string;
  seed: number;
  enabled_signals: Array<'semantic' | 'neighborhood' | 'stability' | 'density'>;
  weighting_strategy: 'learned_validation' | 'equal';
  calibration_method: 'youden_j' | 'f1_optimal' | 'target_fpr_0.01' | 'target_fpr_0.05';
  run_baselines: boolean;
  baseline_methods: Array<'flare' | 'onion'>;
  epochs?: number;
  learning_rate?: number;
}

export interface LiveSSEEvent {
  event_id: number;
  event_type: string;
  stage: string;
  status: 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED';
  timestamp: string;
  message: string;
  progress_info?: {
    processed_samples?: number;
    total_samples?: number;
    model_a?: string;
    model_b?: string;
  } | null;
  data?: Record<string, any> | null;
}

export interface SignalContribution {
  signal_name: string;
  weight: number;
  raw_value?: number | null;
  normalized_value: number;
  linear_contribution: number;
  contribution_percentage: number;
}

export interface LiveSampleInspection {
  sample_id: string;
  text: string;
  label: string | null;
  split: string;
  ground_truth_poisoned?: boolean | null;
  trust_score: number;
  suspicion_score: number;
  threshold: number;
  decision: 'ISOLATE' | 'RETAIN';
  signals: {
    semantic?: number | null;
    neighborhood?: number | null;
    stability?: number | null;
    density?: number | null;
  };
  contributions: SignalContribution[];
}

export interface LiveRetrainingReport {
  baseline_clean_accuracy: number;
  purified_clean_accuracy: number;
  clean_accuracy_delta: number;
  baseline_attack_success_rate: number;
  purified_attack_success_rate: number;
  attack_success_rate_reduction: number;
  isolated_count: number;
  retained_count: number;
  total_train_samples: number;
  retention_rate: number;
  evaluation_precision?: number | null;
  evaluation_recall?: number | null;
  evaluation_f1?: number | null;
  evaluation_auroc?: number | null;
  confusion_matrix?: {
    true_positives: number;
    false_positives: number;
    true_negatives: number;
    false_negatives: number;
  } | null;
}

export interface LiveBaselineResult {
  method: string;
  threshold: number;
  precision: number;
  recall?: number | null;
  f1: number;
  auroc?: number | null;
  auprc?: number | null;
  retention_rate: number;
  downstream_clean_accuracy: number;
  downstream_attack_success_rate: number;
  runtime_seconds: number;
  status: 'SUCCESS' | 'ERROR';
  error_message?: string | null;
}

export interface LiveJobSummary {
  job_id: string;
  status: 'CREATED' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'CANCELLED';
  dataset_id: string;
  attack_type: string;
  poison_rate: number;
  created_at: string;
  completed_at?: string | null;
  current_stage: string;
  error_message?: string | null;
}

export interface LiveJobResponse {
  job_id: string;
  status: 'CREATED' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'CANCELLED';
  request: LiveInvestigationRequest;
  created_at: string;
  completed_at?: string | null;
  current_stage: string;
  error_message?: string | null;
  dataset_fingerprint?: string | null;
  calibrated_threshold?: number | null;
  learned_weights?: Record<string, number> | null;
  sample_inspections: LiveSampleInspection[];
  retraining_report?: LiveRetrainingReport | null;
  baseline_results: LiveBaselineResult[];
  events: LiveSSEEvent[];
}

export async function startLiveInvestigation(payload: LiveInvestigationRequest): Promise<{ job_id: string; status: string; created_at: string }> {
  const res = await fetch(`${API_BASE}/live/investigate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to start live investigation' }));
    throw new Error(err.detail || 'Failed to start live investigation');
  }
  return res.json();
}

export async function fetchLiveJob(jobId: string): Promise<LiveJobResponse> {
  const res = await fetch(`${API_BASE}/live/jobs/${jobId}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: `Job ${jobId} not found` }));
    throw new Error(err.detail || `Job ${jobId} not found`);
  }
  return res.json();
}

export async function fetchLiveJobs(): Promise<LiveJobSummary[]> {
  const res = await fetch(`${API_BASE}/live/jobs`);
  if (!res.ok) throw new Error('Failed to fetch live jobs list');
  return res.json();
}

export function createLiveEventSource(
  jobId: string,
  onEvent: (event: LiveSSEEvent) => void,
  onError?: (err: any) => void
): EventSource {
  const es = new EventSource(`${API_BASE}/live/stream/${jobId}`);
  
  es.onmessage = (msg) => {
    try {
      const data = JSON.parse(msg.data);
      onEvent(data);
    } catch (e) {
      console.error('Failed to parse SSE event data', e);
    }
  };

  // Listen to custom event types as well
  const eventTypes = [
    'JOB_CREATED', 'DATASET_VALIDATING', 'DATASET_VALIDATED',
    'POISONING_STARTED', 'POISONING_COMPLETED',
    'SPLITTING_STARTED', 'SPLITTING_COMPLETED',
    'REPRESENTATIONS_STARTED', 'REPRESENTATIONS_PROGRESS', 'REPRESENTATIONS_COMPLETED',
    'SEMANTIC_ANALYSIS_STARTED', 'SEMANTIC_ANALYSIS_COMPLETED',
    'NEIGHBORHOOD_ANALYSIS_STARTED', 'NEIGHBORHOOD_ANALYSIS_COMPLETED',
    'STABILITY_ANALYSIS_STARTED', 'STABILITY_ANALYSIS_COMPLETED',
    'DENSITY_ANALYSIS_STARTED', 'DENSITY_ANALYSIS_COMPLETED',
    'TRUST_SCORING_STARTED', 'TRUST_SCORING_COMPLETED',
    'THRESHOLD_CALIBRATION_STARTED', 'THRESHOLD_CALIBRATION_COMPLETED',
    'ISOLATION_STARTED', 'ISOLATION_COMPLETED',
    'RETRAINING_STARTED', 'RETRAINING_PROGRESS', 'RETRAINING_COMPLETED',
    'EVALUATION_STARTED', 'EVALUATION_COMPLETED',
    'BASELINE_STARTED', 'BASELINE_COMPLETED',
    'JOB_COMPLETED', 'JOB_FAILED'
  ];

  eventTypes.forEach((evtType) => {
    es.addEventListener(evtType, (e: any) => {
      try {
        const data = JSON.parse(e.data);
        onEvent(data);
      } catch (err) {
        console.error(`Failed to parse SSE event ${evtType}`, err);
      }
    });
  });

  es.onerror = (err) => {
    if (onError) onError(err);
  };

  return es;
}
