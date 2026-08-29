export interface ExperimentListItem {
    experiment_name: string;
    experiment_fingerprint: string;
    dataset_id: string;
    dataset_version: string;
    poisoning_enabled: boolean;
    status: string;
}

export interface PoisoningMetadata {
    attack_type: string;
    poison_rate: number;
    trigger: string;
    target_label: string;
    seed: number;
}

export interface RepresentationConfig {
    model_name: string;
    max_length: number;
    batch_size: number;
    device: string;
    layers: number[];
    use_cache: boolean;
    cache_dir: string;
}

export interface DetectorConfig {
    layers: number[];
    threshold: number;
}

export interface EvaluationReport {
    total_evaluated: number;
    poisoned_samples: number;
    clean_samples: number;
    true_positive: number;
    false_positive: number;
    true_negative: number;
    false_negative: number;
    precision: number;
    recall: number;
    f1: number;
    accuracy: number;
    detector_name: string;
}

export interface DetectionResult {
    sample_ids: string[];
    scores: number[];
    is_anomalous: boolean[];
    detector_name: string;
}

export interface DetectionPipelineResult {
    dataset_id: string;
    dataset_version: string;
    pipeline_fingerprint: string;
    poisoning_metadata: PoisoningMetadata | null;
    representation_config: RepresentationConfig;
    detector_config: DetectorConfig;
    threshold: number;
    detection_result: DetectionResult;
    evaluation_report: EvaluationReport;
}

export interface ExperimentResult {
    experiment_name: string;
    experiment_fingerprint: string;
    dataset_id: string;
    dataset_version: string;
    pipeline_result: DetectionPipelineResult;
}

export async function fetchExperiments(): Promise<ExperimentListItem[]> {
    const res = await fetch('/api/demo/experiments');
    if (!res.ok) throw new Error('Failed to fetch experiments');
    return res.json();
}

export async function fetchExperiment(fingerprint: string): Promise<ExperimentResult> {
    const res = await fetch(`/api/demo/experiments/${fingerprint}`);
    if (!res.ok) throw new Error('Failed to fetch experiment details');
    return res.json();
}

export interface RunExperimentResponse {
    status: string;
    error?: string;
    fingerprint: string;
    result: ExperimentResult | null;
}

export interface DatasetUploadResponse {
    dataset_id: string;
    dataset_version: string;
    filename: string;
    total_samples: number;
    train_samples: number;
    validation_samples: number;
    test_samples: number;
    label_mode: string;
}

export async function uploadDataset(file: File): Promise<DatasetUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    
    const res = await fetch('/api/demo/datasets/upload', {
        method: 'POST',
        body: formData,
    });
    
    if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Failed to upload dataset');
    }
    return res.json();
}

export async function runExperiment(configName: string, customDatasetId?: string): Promise<RunExperimentResponse> {
    const body: any = { config_name: configName };
    if (customDatasetId) {
        body.custom_dataset_id = customDatasetId;
    }
    
    const res = await fetch('/api/demo/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config_name: configName })
    });
    
    if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Failed to run experiment');
    }
    const data = await res.json();
    if (data.status === 'failed' && data.error) {
        throw new Error(data.error);
    }
    return data;
}
