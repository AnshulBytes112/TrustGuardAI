import React, { useState, useEffect, useRef } from 'react';
import {
  LiveInvestigationRequest,
  LiveJobResponse,
  LiveJobSummary,
  LiveSampleInspection,
  LiveSSEEvent,
  startLiveInvestigation,
  fetchLiveJob,
  fetchLiveJobs,
  createLiveEventSource,
} from '../api';

const PIPELINE_STAGES = [
  { id: 'INITIALIZATION', label: 'Initialization', icon: '⚡' },
  { id: 'VALIDATION', label: 'Dataset Validation', icon: '🔍' },
  { id: 'POISONING', label: 'Backdoor Injection', icon: '🧪' },
  { id: 'SPLITTING', label: 'Dataset Partition', icon: '✂️' },
  { id: 'REPRESENTATIONS', label: 'DistilBERT Reps', icon: '🧬' },
  { id: 'SEMANTIC_CONSISTENCY', label: 'Semantic Signal', icon: '🎯' },
  { id: 'NEIGHBORHOOD_CONSISTENCY', label: 'Neighborhood Signal', icon: '🌐' },
  { id: 'PREDICTION_STABILITY', label: 'Stability Signal', icon: '🛡️' },
  { id: 'DENSITY_ANALYSIS', label: 'Density Signal', icon: '📊' },
  { id: 'CALIBRATION', label: 'Threshold Calibration', icon: '⚙️' },
  { id: 'TRUST_SCORING', label: 'Trust Aggregation', icon: '💎' },
  { id: 'ISOLATION', label: 'Purification Isolation', icon: '🧹' },
  { id: 'RETRAINING', label: 'Downstream Retraining', icon: '🔄' },
  { id: 'EVALUATION', label: 'Test Evaluation', icon: '📈' },
];

export const LiveInvestigationView: React.FC = () => {
  // Configuration state
  const [datasetId, setDatasetId] = useState<string>('demo_sst2');
  const [attackType, setAttackType] = useState<LiveInvestigationRequest['attack_type']>('rare_word');
  const [poisonRate, setPoisonRate] = useState<number>(0.10);
  const [targetLabel, setTargetLabel] = useState<string>('POSITIVE');
  const [weightingStrategy, setWeightingStrategy] = useState<'learned_validation' | 'equal'>('learned_validation');
  const [calibrationMethod, setCalibrationMethod] = useState<'youden_j' | 'f1_optimal' | 'target_fpr_0.01' | 'target_fpr_0.05'>('youden_j');
  const [runBaselines, setRunBaselines] = useState<boolean>(true);
  const [selectedBaselines, setSelectedBaselines] = useState<Array<'flare' | 'onion'>>(['flare', 'onion']);
  const [seed, setSeed] = useState<number>(42);

  // Active execution state
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [jobData, setJobData] = useState<LiveJobResponse | null>(null);
  const [events, setEvents] = useState<LiveSSEEvent[]>([]);
  const [isExecuting, setIsExecuting] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Inspection modal
  const [selectedSample, setSelectedSample] = useState<LiveSampleInspection | null>(null);
  const [sampleFilter, setSampleFilter] = useState<'ALL' | 'ISOLATED' | 'RETAINED'>('ALL');
  const [searchQuery, setSearchQuery] = useState<string>('');

  // Past jobs list
  const [pastJobs, setPastJobs] = useState<LiveJobSummary[]>([]);

  const eventSourceRef = useRef<EventSource | null>(null);
  const logTerminalRef = useRef<HTMLDivElement | null>(null);

  // Fetch past jobs on mount
  useEffect(() => {
    loadPastJobs();
  }, []);

  const loadPastJobs = async () => {
    try {
      const list = await fetchLiveJobs();
      setPastJobs(list);
    } catch (e) {
      console.warn('Failed to load past live jobs', e);
    }
  };

  // Auto-scroll terminal log
  useEffect(() => {
    if (logTerminalRef.current) {
      logTerminalRef.current.scrollTop = logTerminalRef.current.scrollHeight;
    }
  }, [events]);

  // Clean up EventSource on unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  const handleStartInvestigation = async () => {
    setErrorMsg(null);
    setIsExecuting(true);
    setEvents([]);
    setJobData(null);
    setSelectedSample(null);

    const req: LiveInvestigationRequest = {
      dataset_id: datasetId,
      attack_type: attackType,
      poison_rate: poisonRate,
      target_label: targetLabel,
      seed: seed,
      enabled_signals: ['semantic', 'neighborhood', 'stability', 'density'],
      weighting_strategy: weightingStrategy,
      calibration_method: calibrationMethod,
      run_baselines: runBaselines,
      baseline_methods: selectedBaselines,
      epochs: 15,
      learning_rate: 0.01,
    };

    try {
      const res = await startLiveInvestigation(req);
      setActiveJobId(res.job_id);
      connectSSE(res.job_id);
      loadPastJobs();
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to start live investigation.');
      setIsExecuting(false);
    }
  };

  const connectSSE = (jobId: string) => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    const es = createLiveEventSource(
      jobId,
      (event: LiveSSEEvent) => {
        setEvents((prev) => {
          // Deduplicate by event_id
          if (prev.some((e) => e.event_id === event.event_id)) return prev;
          return [...prev, event];
        });

        if (event.event_type === 'JOB_COMPLETED' || event.event_type === 'JOB_FAILED') {
          setIsExecuting(false);
          // Fetch final authoritative job data
          fetchLiveJob(jobId)
            .then((data) => {
              setJobData(data);
              loadPastJobs();
            })
            .catch((e) => console.error('Failed to fetch completed job', e));
        }
      },
      (err) => {
        console.warn('SSE connection warning', err);
      }
    );

    eventSourceRef.current = es;
  };

  const handleSelectPastJob = async (jobId: string) => {
    setActiveJobId(jobId);
    setErrorMsg(null);
    try {
      const full = await fetchLiveJob(jobId);
      setJobData(full);
      setEvents(full.events || []);
      if (full.status === 'RUNNING') {
        setIsExecuting(true);
        connectSSE(jobId);
      } else {
        setIsExecuting(false);
      }
    } catch (e: any) {
      setErrorMsg(e.message || 'Failed to load past job.');
    }
  };

  // Determine stage status from events
  const getStageStatus = (stageId: string): 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED' => {
    if (!jobData && events.length === 0) return 'PENDING';
    
    // Check if error in this stage
    const failEvent = events.find((e) => e.stage === stageId && e.status === 'FAILED');
    if (failEvent) return 'FAILED';

    const compEvent = events.find((e) => e.stage === stageId && e.status === 'COMPLETED');
    if (compEvent) return 'COMPLETED';

    const runEvent = events.find((e) => e.stage === stageId && e.status === 'RUNNING');
    if (runEvent) return 'RUNNING';

    // Check if subsequent stage has completed
    const currentIdx = PIPELINE_STAGES.findIndex((s) => s.id === stageId);
    const hasLaterCompleted = events.some((e) => {
      const eIdx = PIPELINE_STAGES.findIndex((s) => s.id === e.stage);
      return eIdx > currentIdx && e.status === 'COMPLETED';
    });
    if (hasLaterCompleted) return 'COMPLETED';

    return 'PENDING';
  };

  const getStageProgressText = (stageId: string): string | null => {
    const lastEvent = [...events].reverse().find((e) => e.stage === stageId);
    if (!lastEvent) return null;
    if (lastEvent.progress_info) {
      if (lastEvent.progress_info.processed_samples !== undefined && lastEvent.progress_info.total_samples !== undefined) {
        return `${lastEvent.progress_info.processed_samples} / ${lastEvent.progress_info.total_samples} samples`;
      }
      if (lastEvent.progress_info.model_a) {
        return `Model A: ${lastEvent.progress_info.model_a} | Model B: ${lastEvent.progress_info.model_b}`;
      }
    }
    if (lastEvent.status === 'RUNNING') {
      return 'Running...';
    }
    return null;
  };

  // Filtered samples
  const displaySamples = (jobData?.sample_inspections || []).filter((s) => {
    if (sampleFilter === 'ISOLATED' && s.decision !== 'ISOLATE') return false;
    if (sampleFilter === 'RETAINED' && s.decision !== 'RETAIN') return false;
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      return s.sample_id.toLowerCase().includes(q) || s.text.toLowerCase().includes(q);
    }
    return true;
  });

  return (
    <div className="space-y-6 animate-fadeIn pb-12">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-border pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center justify-center p-2 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse mr-2" />
              LIVE CONSOLE
            </span>
            <h1 className="text-2xl font-bold text-white tracking-tight">Real-Time Research Investigation</h1>
          </div>
          <p className="text-sm text-text-secondary mt-1">
            Zero simulation backend telemetry: live representation extraction, multi-signal TrustGuard scoring, purification, downstream retraining, and baseline comparisons.
          </p>
        </div>

        {/* Action button */}
        <div className="flex items-center gap-3">
          {pastJobs.length > 0 && (
            <select
              className="bg-card border border-border text-sm rounded-lg px-3 py-2 text-text-secondary focus:outline-none focus:border-primary"
              value={activeJobId || ''}
              onChange={(e) => e.target.value && handleSelectPastJob(e.target.value)}
            >
              <option value="">-- Recent Live Runs ({pastJobs.length}) --</option>
              {pastJobs.map((j) => (
                <option key={j.job_id} value={j.job_id}>
                  {j.job_id.slice(0, 10)}... [{j.attack_type.toUpperCase()}] ({j.status})
                </option>
              ))}
            </select>
          )}

          <button
            onClick={handleStartInvestigation}
            disabled={isExecuting}
            className={`px-5 py-2.5 rounded-lg font-medium text-sm flex items-center gap-2 shadow-lg transition-all ${
              isExecuting
                ? 'bg-primary/50 text-white cursor-not-allowed'
                : 'bg-primary hover:bg-primary-hover text-white shadow-primary/20 hover:shadow-primary/30 active:scale-95'
            }`}
          >
            {isExecuting ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Pipeline Active...
              </>
            ) : (
              <>
                <span>▶</span> Launch Live Investigation
              </>
            )}
          </button>
        </div>
      </div>

      {errorMsg && (
        <div className="bg-red-500/10 border border-red-500/30 text-red-400 px-4 py-3 rounded-lg text-sm flex items-center justify-between">
          <span>⚠️ {errorMsg}</span>
          <button onClick={() => setErrorMsg(null)} className="text-red-300 hover:text-white">✕</button>
        </div>
      )}

      {/* Configuration Panel */}
      <div className="bg-card border border-border rounded-xl p-5 shadow-sm space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-white uppercase tracking-wider flex items-center gap-2">
            <span>⚙️</span> Investigation Parameters
          </h2>
          <span className="text-xs text-text-tertiary">All values passed directly to ML engine</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4">
          {/* Dataset */}
          <div>
            <label className="block text-xs font-medium text-text-secondary mb-1">Dataset</label>
            <select
              value={datasetId}
              onChange={(e) => setDatasetId(e.target.value)}
              disabled={isExecuting}
              className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-primary disabled:opacity-50"
            >
              <option value="demo_sst2">SST-2 Canonical Benchmark</option>
              <option value="synthetic">Synthetic Clean/Poison Set</option>
              <option value="custom">Custom Uploaded JSONL</option>
            </select>
          </div>

          {/* Attack Type */}
          <div>
            <label className="block text-xs font-medium text-text-secondary mb-1">Attack Mechanism</label>
            <select
              value={attackType}
              onChange={(e) => setAttackType(e.target.value as any)}
              disabled={isExecuting}
              className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-primary disabled:opacity-50"
            >
              <option value="rare_word">Rare-Word Trigger</option>
              <option value="common_word">Common-Word Trigger</option>
              <option value="sentence_trigger">Sentence Trigger</option>
              <option value="syntactic_style">Syntactic Style Trigger</option>
              <option value="semantic_trigger">Semantic Domain Trigger</option>
              <option value="character_perturbation">Character Perturbation</option>
              <option value="text_backdoor_v1">Text Backdoor v1</option>
              <option value="none">Clean (No Poisoning)</option>
            </select>
          </div>

          {/* Poison Rate */}
          <div>
            <label className="block text-xs font-medium text-text-secondary mb-1">
              Poison Rate ({Math.round(poisonRate * 100)}%)
            </label>
            <select
              value={poisonRate}
              onChange={(e) => setPoisonRate(parseFloat(e.target.value))}
              disabled={isExecuting || attackType === 'none'}
              className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-primary disabled:opacity-50"
            >
              <option value={0.05}>5%</option>
              <option value={0.10}>10%</option>
              <option value={0.20}>20%</option>
              <option value={0.30}>30%</option>
            </select>
          </div>

          {/* Target Label */}
          <div>
            <label className="block text-xs font-medium text-text-secondary mb-1">Target Label</label>
            <select
              value={targetLabel}
              onChange={(e) => setTargetLabel(e.target.value)}
              disabled={isExecuting || attackType === 'none'}
              className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-primary disabled:opacity-50"
            >
              <option value="POSITIVE">POSITIVE</option>
              <option value="NEGATIVE">NEGATIVE</option>
            </select>
          </div>

          {/* Weighting Strategy */}
          <div>
            <label className="block text-xs font-medium text-text-secondary mb-1">Weighting Scheme</label>
            <select
              value={weightingStrategy}
              onChange={(e) => setWeightingStrategy(e.target.value as any)}
              disabled={isExecuting}
              className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-primary disabled:opacity-50"
            >
              <option value="learned_validation">Validation-Optimized</option>
              <option value="equal">Equal Weights (0.25 ea)</option>
            </select>
          </div>

          {/* Threshold Calibration */}
          <div>
            <label className="block text-xs font-medium text-text-secondary mb-1">Calibration Objective</label>
            <select
              value={calibrationMethod}
              onChange={(e) => setCalibrationMethod(e.target.value as any)}
              disabled={isExecuting}
              className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-primary disabled:opacity-50"
            >
              <option value="youden_j">Youden's J Index</option>
              <option value="f1_optimal">F1 Optimal</option>
              <option value="target_fpr_0.05">Target FPR ≤ 5%</option>
              <option value="target_fpr_0.01">Target FPR ≤ 1%</option>
            </select>
          </div>
        </div>

        {/* Baselines bar */}
        <div className="pt-2 border-t border-border/50 flex flex-wrap items-center justify-between text-xs text-text-secondary">
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={runBaselines}
                onChange={(e) => setRunBaselines(e.target.checked)}
                disabled={isExecuting}
                className="rounded border-border text-primary focus:ring-primary"
              />
              <span className="text-white font-medium">Compare with Baselines</span>
            </label>

            {runBaselines && (
              <div className="flex items-center gap-3 pl-2 border-l border-border">
                <label className="flex items-center gap-1.5 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={selectedBaselines.includes('flare')}
                    onChange={(e) => {
                      if (e.target.checked) setSelectedBaselines((p) => [...p, 'flare']);
                      else setSelectedBaselines((p) => p.filter((b) => b !== 'flare'));
                    }}
                    disabled={isExecuting}
                    className="rounded border-border text-primary"
                  />
                  <span>FLARE</span>
                </label>
                <label className="flex items-center gap-1.5 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={selectedBaselines.includes('onion')}
                    onChange={(e) => {
                      if (e.target.checked) setSelectedBaselines((p) => [...p, 'onion']);
                      else setSelectedBaselines((p) => p.filter((b) => b !== 'onion'));
                    }}
                    disabled={isExecuting}
                    className="rounded border-border text-primary"
                  />
                  <span>ONION</span>
                </label>
              </div>
            )}
          </div>

          <div className="flex items-center gap-2">
            <span>Seed:</span>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(parseInt(e.target.value) || 42)}
              disabled={isExecuting}
              className="w-16 bg-background border border-border rounded px-2 py-1 text-white text-xs"
            />
          </div>
        </div>
      </div>

      {/* Live Pipeline Stepper */}
      <div className="bg-card border border-border rounded-xl p-5 shadow-sm space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-white uppercase tracking-wider flex items-center gap-2">
            <span>🔄</span> End-to-End Pipeline Telemetry
          </h2>
          <span className="text-xs text-text-tertiary">
            Status: {isExecuting ? '⚡ Running computation...' : jobData ? '✅ Completed' : 'Idle'}
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2.5">
          {PIPELINE_STAGES.map((st) => {
            const status = getStageStatus(st.id);
            const progress = getStageProgressText(st.id);

            return (
              <div
                key={st.id}
                className={`p-3 rounded-lg border transition-all flex flex-col justify-between min-h-[84px] ${
                  status === 'COMPLETED'
                    ? 'bg-emerald-500/5 border-emerald-500/30 text-emerald-400'
                    : status === 'RUNNING'
                    ? 'bg-primary/10 border-primary shadow-sm shadow-primary/20 text-white animate-pulse'
                    : status === 'FAILED'
                    ? 'bg-red-500/10 border-red-500/40 text-red-400'
                    : 'bg-background/40 border-border/60 text-text-tertiary opacity-60'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-lg">{st.icon}</span>
                  <span
                    className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                      status === 'COMPLETED'
                        ? 'bg-emerald-500/20 text-emerald-300'
                        : status === 'RUNNING'
                        ? 'bg-primary/30 text-primary-light'
                        : status === 'FAILED'
                        ? 'bg-red-500/20 text-red-300'
                        : 'bg-border/30 text-text-tertiary'
                    }`}
                  >
                    {status}
                  </span>
                </div>
                <div>
                  <div className="text-xs font-semibold leading-tight text-white/90 mt-1 truncate">
                    {st.label}
                  </div>
                  {progress && (
                    <div className="text-[10px] text-text-secondary mt-0.5 font-mono truncate">
                      {progress}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Live Event Stream Terminal */}
      <div className="bg-[#0D1117] border border-border rounded-xl p-4 font-mono text-xs shadow-inner">
        <div className="flex items-center justify-between border-b border-border/40 pb-2 mb-2 text-text-tertiary">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-red-500/80 inline-block" />
            <span className="w-2.5 h-2.5 rounded-full bg-yellow-500/80 inline-block" />
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500/80 inline-block" />
            <span className="ml-2 text-text-secondary font-sans font-medium text-xs">SSE Server Event Stream (One-Way Authoritative)</span>
          </div>
          <span>{events.length} Events Received</span>
        </div>

        <div
          ref={logTerminalRef}
          className="h-44 overflow-y-auto space-y-1.5 pr-2 select-text"
        >
          {events.length === 0 ? (
            <div className="text-text-tertiary italic">Waiting for pipeline trigger...</div>
          ) : (
            events.map((ev) => (
              <div key={ev.event_id} className="flex items-start gap-2 leading-relaxed">
                <span className="text-text-tertiary shrink-0">
                  {new Date(ev.timestamp).toLocaleTimeString()}
                </span>
                <span
                  className={`font-semibold px-1 rounded text-[10px] shrink-0 ${
                    ev.event_type === 'JOB_COMPLETED'
                      ? 'bg-emerald-500/20 text-emerald-400'
                      : ev.event_type.includes('ERROR') || ev.event_type === 'JOB_FAILED'
                      ? 'bg-red-500/20 text-red-400'
                      : ev.event_type.includes('COMPLETED')
                      ? 'bg-blue-500/20 text-blue-400'
                      : 'bg-primary/20 text-primary-light'
                  }`}
                >
                  {ev.event_type}
                </span>
                <span className="text-gray-300">{ev.message}</span>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Retraining Robustness Results & Isolation Overview */}
      {jobData?.retraining_report && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          {/* Retraining Benchmark Card */}
          <div className="lg:col-span-2 bg-card border border-border rounded-xl p-5 shadow-sm space-y-4">
            <div className="flex items-center justify-between border-b border-border pb-3">
              <h3 className="text-sm font-semibold text-white uppercase tracking-wider flex items-center gap-2">
                <span>🔄</span> Retraining Robustness (Model A vs Model B)
              </h3>
              <span className="text-xs text-emerald-400 font-mono">
                Clean Acc Δ: {(jobData.retraining_report.clean_accuracy_delta * 100).toFixed(1)}% | ASR Reduction: {(jobData.retraining_report.attack_success_rate_reduction * 100).toFixed(1)}%
              </span>
            </div>

            <div className="grid grid-cols-2 gap-4">
              {/* Model A: Raw */}
              <div className="p-4 rounded-xl bg-background/80 border border-border/80 space-y-2">
                <div className="flex items-center justify-between text-xs font-semibold text-red-400">
                  <span>MODEL A (Raw Unpurified)</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-red-500/10 border border-red-500/20">POISONED TRAIN</span>
                </div>
                <div className="grid grid-cols-2 gap-2 pt-2 text-center">
                  <div className="p-2 rounded bg-card border border-border/40">
                    <div className="text-[10px] text-text-secondary">Clean Accuracy</div>
                    <div className="text-lg font-bold text-white font-mono">
                      {(jobData.retraining_report.baseline_clean_accuracy * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="p-2 rounded bg-card border border-border/40">
                    <div className="text-[10px] text-text-secondary">Attack Success Rate (ASR)</div>
                    <div className="text-lg font-bold text-red-400 font-mono">
                      {(jobData.retraining_report.baseline_attack_success_rate * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>

              {/* Model B: Purified */}
              <div className="p-4 rounded-xl bg-background/80 border border-emerald-500/30 space-y-2">
                <div className="flex items-center justify-between text-xs font-semibold text-emerald-400">
                  <span>MODEL B (TrustGuard Purified)</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20">PURIFIED TRAIN</span>
                </div>
                <div className="grid grid-cols-2 gap-2 pt-2 text-center">
                  <div className="p-2 rounded bg-card border border-border/40">
                    <div className="text-[10px] text-text-secondary">Clean Accuracy</div>
                    <div className="text-lg font-bold text-emerald-400 font-mono">
                      {(jobData.retraining_report.purified_clean_accuracy * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="p-2 rounded bg-card border border-border/40">
                    <div className="text-[10px] text-text-secondary">Attack Success Rate (ASR)</div>
                    <div className="text-lg font-bold text-emerald-400 font-mono">
                      {(jobData.retraining_report.purified_attack_success_rate * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Test Split Detection Metrics */}
            <div className="grid grid-cols-4 gap-2 pt-2 border-t border-border/40 text-center">
              <div className="p-2 rounded bg-background/50 border border-border/40">
                <div className="text-[10px] text-text-secondary">Precision</div>
                <div className="text-sm font-bold text-white font-mono">
                  {jobData.retraining_report.evaluation_precision !== null ? (jobData.retraining_report.evaluation_precision! * 100).toFixed(1) + '%' : 'N/A'}
                </div>
              </div>
              <div className="p-2 rounded bg-background/50 border border-border/40">
                <div className="text-[10px] text-text-secondary">Recall</div>
                <div className="text-sm font-bold text-white font-mono">
                  {jobData.retraining_report.evaluation_recall !== null ? (jobData.retraining_report.evaluation_recall! * 100).toFixed(1) + '%' : 'N/A'}
                </div>
              </div>
              <div className="p-2 rounded bg-background/50 border border-border/40">
                <div className="text-[10px] text-text-secondary">F1 Score</div>
                <div className="text-sm font-bold text-white font-mono">
                  {jobData.retraining_report.evaluation_f1 !== null ? (jobData.retraining_report.evaluation_f1! * 100).toFixed(1) + '%' : 'N/A'}
                </div>
              </div>
              <div className="p-2 rounded bg-background/50 border border-border/40">
                <div className="text-[10px] text-text-secondary">AUROC</div>
                <div className="text-sm font-bold text-white font-mono">
                  {jobData.retraining_report.evaluation_auroc !== null ? (jobData.retraining_report.evaluation_auroc!).toFixed(3) : 'N/A'}
                </div>
              </div>
            </div>
          </div>

          {/* Purification & Confusion Matrix Card */}
          <div className="bg-card border border-border rounded-xl p-5 shadow-sm space-y-4">
            <div className="flex items-center justify-between border-b border-border pb-3">
              <h3 className="text-sm font-semibold text-white uppercase tracking-wider flex items-center gap-2">
                <span>🧹</span> Purification & Confusion
              </h3>
              <span className="text-xs text-text-tertiary">
                Threshold: {jobData.calibrated_threshold?.toFixed(4) || 'N/A'}
              </span>
            </div>

            {/* Isolation counts */}
            <div className="grid grid-cols-2 gap-2 text-center">
              <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20">
                <div className="text-xs text-red-400 font-medium">Isolated / Quarantined</div>
                <div className="text-xl font-bold text-white font-mono mt-1">
                  {jobData.retraining_report.isolated_count}
                </div>
                <div className="text-[10px] text-text-tertiary mt-0.5">
                  of {jobData.retraining_report.total_train_samples} TRAIN samples
                </div>
              </div>

              <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
                <div className="text-xs text-emerald-400 font-medium">Retained for Model B</div>
                <div className="text-xl font-bold text-white font-mono mt-1">
                  {jobData.retraining_report.retained_count}
                </div>
                <div className="text-[10px] text-text-tertiary mt-0.5">
                  {(jobData.retraining_report.retention_rate * 100).toFixed(1)}% Retention Rate
                </div>
              </div>
            </div>

            {/* Confusion Matrix */}
            {jobData.retraining_report.confusion_matrix && (
              <div className="pt-2 border-t border-border/40 space-y-1.5">
                <div className="text-[11px] font-semibold text-text-secondary uppercase">TEST Split Confusion Matrix</div>
                <div className="grid grid-cols-2 gap-1.5 text-center text-xs font-mono">
                  <div className="p-2 rounded bg-background border border-emerald-500/30">
                    <div className="text-[10px] text-text-secondary">True Positives (TP)</div>
                    <div className="font-bold text-emerald-400">{jobData.retraining_report.confusion_matrix.true_positives}</div>
                  </div>
                  <div className="p-2 rounded bg-background border border-red-500/30">
                    <div className="text-[10px] text-text-secondary">False Positives (FP)</div>
                    <div className="font-bold text-red-400">{jobData.retraining_report.confusion_matrix.false_positives}</div>
                  </div>
                  <div className="p-2 rounded bg-background border border-red-500/30">
                    <div className="text-[10px] text-text-secondary">False Negatives (FN)</div>
                    <div className="font-bold text-red-400">{jobData.retraining_report.confusion_matrix.false_negatives}</div>
                  </div>
                  <div className="p-2 rounded bg-background border border-emerald-500/30">
                    <div className="text-[10px] text-text-secondary">True Negatives (TN)</div>
                    <div className="font-bold text-emerald-400">{jobData.retraining_report.confusion_matrix.true_negatives}</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Baseline Comparison Matrix */}
      {jobData?.baseline_results && jobData.baseline_results.length > 0 && (
        <div className="bg-card border border-border rounded-xl p-5 shadow-sm space-y-3">
          <div className="flex items-center justify-between border-b border-border pb-3">
            <h3 className="text-sm font-semibold text-white uppercase tracking-wider flex items-center gap-2">
              <span>⚖️</span> Baseline Scientific Comparison (Identical Split & Test Set)
            </h3>
            <span className="text-xs text-text-tertiary">Direct execution output</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-background/60 text-text-secondary uppercase tracking-wider border-b border-border">
                <tr>
                  <th className="py-2.5 px-3">Method</th>
                  <th className="py-2.5 px-3">Threshold</th>
                  <th className="py-2.5 px-3">Precision</th>
                  <th className="py-2.5 px-3">Recall</th>
                  <th className="py-2.5 px-3">F1</th>
                  <th className="py-2.5 px-3">AUROC</th>
                  <th className="py-2.5 px-3">Retention</th>
                  <th className="py-2.5 px-3">Clean Acc (CA)</th>
                  <th className="py-2.5 px-3">Attack Succ (ASR)</th>
                  <th className="py-2.5 px-3">Runtime</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/40 font-mono">
                {jobData.baseline_results.map((row, idx) => (
                  <tr
                    key={idx}
                    className={`hover:bg-background/40 transition-colors ${
                      row.method === 'TrustGuard' ? 'bg-primary/5 font-semibold text-white' : 'text-text-secondary'
                    }`}
                  >
                    <td className="py-2.5 px-3 flex items-center gap-2">
                      <span className={`w-2 h-2 rounded-full ${row.method === 'TrustGuard' ? 'bg-primary' : 'bg-gray-500'}`} />
                      <span className={row.method === 'TrustGuard' ? 'text-primary-light font-bold' : ''}>{row.method}</span>
                    </td>
                    <td className="py-2.5 px-3">{row.threshold.toFixed(4)}</td>
                    <td className="py-2.5 px-3">{(row.precision * 100).toFixed(1)}%</td>
                    <td className="py-2.5 px-3">{row.recall !== null && row.recall !== undefined ? (row.recall * 100).toFixed(1) + '%' : 'N/A'}</td>
                    <td className="py-2.5 px-3">{(row.f1 * 100).toFixed(1)}%</td>
                    <td className="py-2.5 px-3">{row.auroc !== null && row.auroc !== undefined ? row.auroc.toFixed(3) : 'N/A'}</td>
                    <td className="py-2.5 px-3">{(row.retention_rate * 100).toFixed(1)}%</td>
                    <td className="py-2.5 px-3 text-emerald-400">{(row.downstream_clean_accuracy * 100).toFixed(1)}%</td>
                    <td className="py-2.5 px-3 text-red-400">{(row.downstream_attack_success_rate * 100).toFixed(1)}%</td>
                    <td className="py-2.5 px-3 text-text-tertiary">{row.runtime_seconds.toFixed(2)}s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Granular Sample Inspection Table */}
      <div className="bg-card border border-border rounded-xl p-5 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-border pb-3">
          <div>
            <h3 className="text-sm font-semibold text-white uppercase tracking-wider flex items-center gap-2">
              <span>🔬</span> Sample Telemetry & Signal Contributions
            </h3>
            <p className="text-xs text-text-tertiary mt-0.5">
              Click any sample row to inspect its exact 4-signal breakdown and linear attribution contributions.
            </p>
          </div>

          <div className="flex items-center gap-3">
            {/* Filter buttons */}
            <div className="flex rounded-lg border border-border bg-background p-0.5 text-xs">
              <button
                onClick={() => setSampleFilter('ALL')}
                className={`px-3 py-1 rounded-md transition-colors ${
                  sampleFilter === 'ALL' ? 'bg-primary text-white font-medium' : 'text-text-secondary hover:text-white'
                }`}
              >
                All ({jobData?.sample_inspections.length || 0})
              </button>
              <button
                onClick={() => setSampleFilter('ISOLATED')}
                className={`px-3 py-1 rounded-md transition-colors ${
                  sampleFilter === 'ISOLATED' ? 'bg-red-500/20 text-red-400 font-medium' : 'text-text-secondary hover:text-white'
                }`}
              >
                Isolated ({(jobData?.sample_inspections || []).filter((s) => s.decision === 'ISOLATE').length})
              </button>
              <button
                onClick={() => setSampleFilter('RETAINED')}
                className={`px-3 py-1 rounded-md transition-colors ${
                  sampleFilter === 'RETAINED' ? 'bg-emerald-500/20 text-emerald-400 font-medium' : 'text-text-secondary hover:text-white'
                }`}
              >
                Retained ({(jobData?.sample_inspections || []).filter((s) => s.decision === 'RETAIN').length})
              </button>
            </div>

            {/* Search */}
            <input
              type="text"
              placeholder="Search sample ID / text..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="bg-background border border-border rounded-lg px-3 py-1.5 text-xs text-white placeholder-text-tertiary focus:outline-none focus:border-primary"
            />
          </div>
        </div>

        {/* Table */}
        <div className="overflow-x-auto max-h-96">
          <table className="w-full text-left text-xs">
            <thead className="bg-background/80 text-text-secondary uppercase tracking-wider border-b border-border sticky top-0 backdrop-blur">
              <tr>
                <th className="py-2.5 px-3">Sample ID</th>
                <th className="py-2.5 px-3">Split</th>
                <th className="py-2.5 px-3">Label</th>
                <th className="py-2.5 px-3">Text Excerpt</th>
                <th className="py-2.5 px-3 text-right">Trust Score</th>
                <th className="py-2.5 px-3 text-right">Suspicion Score</th>
                <th className="py-2.5 px-3 text-center">Decision</th>
                <th className="py-2.5 px-3 text-center">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/40">
              {displaySamples.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-8 text-center text-text-tertiary italic">
                    {jobData ? 'No matching samples found.' : 'Run a live investigation to populate sample telemetry.'}
                  </td>
                </tr>
              ) : (
                displaySamples.map((sample) => (
                  <tr
                    key={sample.sample_id}
                    onClick={() => setSelectedSample(sample)}
                    className="hover:bg-primary/5 cursor-pointer transition-colors"
                  >
                    <td className="py-2.5 px-3 font-mono font-medium text-white flex items-center gap-1.5">
                      {sample.ground_truth_poisoned === true && (
                        <span className="text-[10px] text-red-400" title="Ground Truth Poisoned">🧪</span>
                      )}
                      {sample.sample_id}
                    </td>
                    <td className="py-2.5 px-3">
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-background border border-border font-mono text-text-secondary">
                        {sample.split}
                      </span>
                    </td>
                    <td className="py-2.5 px-3">
                      <span className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${
                        sample.label === 'POSITIVE' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'
                      }`}>
                        {sample.label || 'UNLABELLED'}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-text-secondary max-w-xs truncate" title={sample.text}>
                      {sample.text}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono font-semibold text-emerald-400">
                      {(sample.trust_score * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono font-semibold text-red-400">
                      {sample.suspicion_score.toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3 text-center">
                      <span
                        className={`text-[10px] px-2 py-0.5 rounded font-bold ${
                          sample.decision === 'ISOLATE'
                            ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                            : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                        }`}
                      >
                        {sample.decision}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-center">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedSample(sample);
                        }}
                        className="px-2 py-1 rounded bg-primary/20 text-primary-light hover:bg-primary hover:text-white transition-colors text-[10px]"
                      >
                        Inspect
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Interactive Sample Detail Inspector Modal */}
      {selectedSample && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-card border border-border rounded-2xl max-w-2xl w-full p-6 shadow-2xl space-y-5 animate-scaleIn">
            {/* Modal Header */}
            <div className="flex items-center justify-between border-b border-border pb-3">
              <div className="flex items-center gap-2">
                <span className="text-xl">🔬</span>
                <div>
                  <h3 className="text-base font-bold text-white flex items-center gap-2">
                    Sample Inspection: <span className="font-mono text-primary-light">{selectedSample.sample_id}</span>
                  </h3>
                  <div className="text-xs text-text-tertiary flex items-center gap-2 mt-0.5">
                    <span>Split: <strong>{selectedSample.split}</strong></span>
                    <span>•</span>
                    <span>Label: <strong>{selectedSample.label || 'None'}</strong></span>
                    {selectedSample.ground_truth_poisoned !== null && (
                      <>
                        <span>•</span>
                        <span>Ground Truth: <strong className={selectedSample.ground_truth_poisoned ? 'text-red-400' : 'text-emerald-400'}>
                          {selectedSample.ground_truth_poisoned ? 'POISONED' : 'CLEAN'}
                        </strong></span>
                      </>
                    )}
                  </div>
                </div>
              </div>
              <button
                onClick={() => setSelectedSample(null)}
                className="text-text-tertiary hover:text-white p-1 rounded-lg text-lg"
              >
                ✕
              </button>
            </div>

            {/* Sample Text Box */}
            <div className="p-3 rounded-lg bg-background border border-border/80 text-sm text-gray-200 font-serif leading-relaxed">
              "{selectedSample.text}"
            </div>

            {/* Anomaly & Trust Gauges */}
            <div className="grid grid-cols-3 gap-3 text-center">
              <div className="p-3 rounded-xl bg-background border border-border">
                <div className="text-[10px] text-text-secondary uppercase">TrustScore</div>
                <div className="text-xl font-bold text-emerald-400 font-mono mt-1">
                  {(selectedSample.trust_score * 100).toFixed(1)}%
                </div>
              </div>

              <div className="p-3 rounded-xl bg-background border border-border">
                <div className="text-[10px] text-text-secondary uppercase">SuspicionScore</div>
                <div className="text-xl font-bold text-red-400 font-mono mt-1">
                  {selectedSample.suspicion_score.toFixed(4)}
                </div>
                <div className="text-[10px] text-text-tertiary mt-0.5">
                  Threshold: {selectedSample.threshold.toFixed(4)}
                </div>
              </div>

              <div className="p-3 rounded-xl bg-background border border-border flex flex-col justify-center items-center">
                <div className="text-[10px] text-text-secondary uppercase">Final Decision</div>
                <span
                  className={`mt-1 text-xs px-2.5 py-1 rounded font-bold ${
                    selectedSample.decision === 'ISOLATE'
                      ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                      : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                  }`}
                >
                  {selectedSample.decision}
                </span>
              </div>
            </div>

            {/* 4 Signals & Linear Contribution Breakdown */}
            <div className="space-y-2.5">
              <h4 className="text-xs font-semibold text-white uppercase tracking-wider">
                Attribution & Linear Contribution Breakdown (w_j × s_j)
              </h4>

              <div className="space-y-2">
                {selectedSample.contributions.map((c) => (
                  <div key={c.signal_name} className="p-2.5 rounded-lg bg-background border border-border/60 text-xs">
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-semibold text-white capitalize">{c.signal_name} Consistency</span>
                      <span className="text-text-secondary font-mono">
                        Weight: {c.weight.toFixed(2)} | Score: {c.normalized_value.toFixed(3)} | Contrib: <strong>{c.linear_contribution.toFixed(4)}</strong> ({c.contribution_percentage}%)
                      </span>
                    </div>

                    {/* Progress visual bar */}
                    <div className="w-full bg-border/40 rounded-full h-2 overflow-hidden">
                      <div
                        className="bg-primary h-full rounded-full transition-all duration-500"
                        style={{ width: `${Math.min(100, c.contribution_percentage)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Close button */}
            <div className="pt-2 flex justify-end">
              <button
                onClick={() => setSelectedSample(null)}
                className="px-4 py-2 rounded-lg bg-background hover:bg-border text-white text-xs font-medium transition-colors"
              >
                Close Inspector
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
