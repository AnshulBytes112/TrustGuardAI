import React, { useState, useEffect } from 'react';
import { fetchDatasets, startLiveInvestigation } from '../../api';
import type { DatasetItem, LiveInvestigationRequest } from '../../api';
import { LoadingSkeleton } from '../common/LoadingSkeleton';
import { ErrorState } from '../common/ErrorState';

interface NewInvestigationViewProps {
  initialDatasetId?: string | null;
  onInvestigationStarted: (jobId: string) => void;
  onCancel: () => void;
}

export const NewInvestigationView: React.FC<NewInvestigationViewProps> = ({
  initialDatasetId,
  onInvestigationStarted,
  onCancel,
}) => {
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Form State
  const [currentStep, setCurrentStep] = useState(1);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>(initialDatasetId || '');
  const [attackType, setAttackType] = useState<LiveInvestigationRequest['attack_type']>('rare_word');
  const [poisonRate, setPoisonRate] = useState<number>(0.05);
  const [targetLabel, setTargetLabel] = useState<string>('POSITIVE');
  const [seed, setSeed] = useState<number>(42);

  // Signals
  const [semanticEnabled, setSemanticEnabled] = useState(true);
  const [neighborhoodEnabled, setNeighborhoodEnabled] = useState(true);
  const [stabilityEnabled, setStabilityEnabled] = useState(true);
  const [densityEnabled, setDensityEnabled] = useState(true);

  // Calibration & Baselines
  const [weightingStrategy, setWeightingStrategy] = useState<'learned_validation' | 'equal'>('learned_validation');
  const [calibrationMethod, setCalibrationMethod] = useState<'youden_j' | 'f1_optimal' | 'target_fpr_0.01' | 'target_fpr_0.05'>('youden_j');
  const [runBaselines, setRunBaselines] = useState(true);
  const [runFlare, setRunFlare] = useState(true);
  const [runOnion, setRunOnion] = useState(true);

  useEffect(() => {
    const loadDatasets = async () => {
      try {
        setLoading(true);
        const list = await fetchDatasets();
        setDatasets(list);
        setSelectedDatasetId((prev) => prev || (list.length > 0 ? list[0].id : ''));
      } catch (err: any) {
        setError(err.message || 'Failed to load datasets for configuration');
      } finally {
        setLoading(false);
      }
    };
    loadDatasets();
  }, []);

  const handleLaunch = async () => {
    if (!selectedDatasetId) {
      setError('Please select a dataset to investigate');
      return;
    }

    const enabledSignals: Array<'semantic' | 'neighborhood' | 'stability' | 'density'> = [];
    if (semanticEnabled) enabledSignals.push('semantic');
    if (neighborhoodEnabled) enabledSignals.push('neighborhood');
    if (stabilityEnabled) enabledSignals.push('stability');
    if (densityEnabled) enabledSignals.push('density');

    if (enabledSignals.length === 0) {
      setError('Please enable at least one TrustGuard defense signal');
      return;
    }

    const baselineMethods: Array<'flare' | 'onion'> = [];
    if (runFlare) baselineMethods.push('flare');
    if (runOnion) baselineMethods.push('onion');

    const request: LiveInvestigationRequest = {
      dataset_id: selectedDatasetId,
      attack_type: attackType,
      poison_rate: poisonRate,
      target_label: targetLabel,
      seed,
      enabled_signals: enabledSignals,
      weighting_strategy: weightingStrategy,
      calibration_method: calibrationMethod,
      run_baselines: runBaselines && baselineMethods.length > 0,
      baseline_methods: baselineMethods,
      epochs: 5,
      learning_rate: 0.01,
    };

    try {
      setSubmitting(true);
      setError(null);
      const res = await startLiveInvestigation(request);
      onInvestigationStarted(res.job_id);
    } catch (err: any) {
      setError(err.message || 'Failed to launch live investigation');
      setSubmitting(false);
    }
  };

  if (loading) {
    return <LoadingSkeleton type="card" height="350px" />;
  }

  const selectedDataset = datasets.find((d) => d.id === selectedDatasetId);

  return (
    <div style={{ maxWidth: '880px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      {/* Wizard Step Progression Bar */}
      <div className="neuro-card" style={{ padding: '1.25rem 1.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          {[
            { step: 1, label: '1. Dataset' },
            { step: 2, label: '2. Attack Config' },
            { step: 3, label: '3. Defense Signals' },
            { step: 4, label: '4. Review & Launch' },
          ].map((s) => {
            const isCompleted = currentStep > s.step;
            const isCurrent = currentStep === s.step;
            return (
              <div
                key={s.step}
                onClick={() => setCurrentStep(s.step)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.6rem',
                  cursor: 'pointer',
                  opacity: isCurrent || isCompleted ? 1 : 0.45,
                }}
              >
                <div
                  style={{
                    width: '28px',
                    height: '28px',
                    borderRadius: '50%',
                    backgroundColor: isCurrent ? 'var(--cyan)' : isCompleted ? 'var(--emerald)' : 'var(--bg-sunken)',
                    border: '1px solid var(--border-default)',
                    color: isCurrent || isCompleted ? '#ffffff' : 'var(--text-muted)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '0.78rem',
                    fontWeight: 700,
                  }}
                >
                  {isCompleted ? '✓' : s.step}
                </div>
                <span style={{ fontSize: '0.82rem', fontWeight: isCurrent ? 700 : 500, color: isCurrent ? '#ffffff' : 'var(--text-secondary)' }}>
                  {s.label}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {error && <ErrorState message={error} />}

      {/* Step 1: Select Dataset */}
      {currentStep === 1 && (
        <div className="neuro-card" style={{ padding: '1.75rem' }}>
          <h3 style={{ fontFamily: 'var(--font-brand)', fontSize: '1.15rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.25rem' }}>
            Select Ingested Corpus
          </h3>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-dim)', marginBottom: '1.25rem' }}>
            Choose a training dataset to inject synthetic backdoor triggers and execute TrustGuardAI defense analysis.
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginBottom: '1.5rem' }}>
            {datasets.map((d) => {
              const isSelected = selectedDatasetId === d.id;
              return (
                <div
                  key={d.id}
                  onClick={() => setSelectedDatasetId(d.id)}
                  className={isSelected ? 'neuro-card' : 'neuro-sunken'}
                  style={{
                    padding: '1rem 1.25rem',
                    cursor: 'pointer',
                    border: isSelected ? '1px solid var(--cyan-border)' : '1px solid var(--border-subtle)',
                    backgroundColor: isSelected ? 'var(--bg-surface-elevated)' : 'var(--bg-sunken)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                  }}
                >
                  <div>
                    <div style={{ fontWeight: 700, fontSize: '0.92rem', color: isSelected ? '#ffffff' : 'var(--text-primary)' }}>
                      {d.name}
                    </div>
                    <div style={{ fontSize: '0.76rem', color: 'var(--text-dim)', marginTop: '0.15rem' }}>
                      {d.total_samples} samples &bull; {d.train_count} train / {d.val_count} val / {d.test_count} test
                    </div>
                  </div>
                  <span className="neuro-badge cyan">{d.modality}</span>
                </div>
              );
            })}
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem' }}>
            <button onClick={onCancel} className="neuro-btn">
              Cancel
            </button>
            <button
              onClick={() => setCurrentStep(2)}
              disabled={!selectedDatasetId}
              className="neuro-btn neuro-btn-primary"
            >
              Continue to Attack Config →
            </button>
          </div>
        </div>
      )}

      {/* Step 2: Attack Configuration */}
      {currentStep === 2 && (
        <div className="neuro-card" style={{ padding: '1.75rem' }}>
          <h3 style={{ fontFamily: 'var(--font-brand)', fontSize: '1.15rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.25rem' }}>
            Attack Configuration
          </h3>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-dim)', marginBottom: '1.25rem' }}>
            Specify synthetic backdoor trigger mechanism, injection poison rate, and target adversarial label.
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.25rem', marginBottom: '1.5rem' }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
                ATTACK MECHANISM
              </label>
              <select
                value={attackType}
                onChange={(e) => setAttackType(e.target.value as any)}
                className="neuro-select"
              >
                <option value="none">None (Clean Baseline Validation)</option>
                <option value="rare_word">Rare Word Trigger (e.g. 'cf', 'mn')</option>
                <option value="sentence_trigger">Sentence Trigger ('I watched this 3D movie')</option>
                <option value="syntactic_style">Syntactic Style Backdoor</option>
                <option value="semantic_trigger">Semantic Trigger</option>
                <option value="character_perturbation">Character Perturbation / Typo</option>
                <option value="text_backdoor_v1">Composite Backdoor v1</option>
              </select>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
                TARGET ADVERSARIAL LABEL
              </label>
              <select
                value={targetLabel}
                onChange={(e) => setTargetLabel(e.target.value)}
                className="neuro-select"
              >
                <option value="POSITIVE">POSITIVE</option>
                <option value="NEGATIVE">NEGATIVE</option>
              </select>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
                POISON RATE: {(poisonRate * 100).toFixed(1)}%
              </label>
              <input
                type="range"
                min="0.0"
                max="0.20"
                step="0.01"
                value={poisonRate}
                onChange={(e) => setPoisonRate(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--cyan)' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-dim)', marginTop: '0.2rem' }}>
                <span>0.0% (Clean)</span>
                <span>5.0%</span>
                <span>10.0%</span>
                <span>20.0% (Severe)</span>
              </div>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
                RANDOM SEED (REPRODUCIBILITY)
              </label>
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(parseInt(e.target.value) || 42)}
                className="neuro-input"
              />
            </div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <button onClick={() => setCurrentStep(1)} className="neuro-btn">
              ← Back
            </button>
            <button onClick={() => setCurrentStep(3)} className="neuro-btn neuro-btn-primary">
              Continue to Defense Signals →
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Defense Signals & Calibration */}
      {currentStep === 3 && (
        <div className="neuro-card" style={{ padding: '1.75rem' }}>
          <h3 style={{ fontFamily: 'var(--font-brand)', fontSize: '1.15rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.25rem' }}>
            TrustGuard Defense Signals & Calibration
          </h3>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-dim)', marginBottom: '1.25rem' }}>
            Select multi-signal analyzers to combine into the composite suspicion scoring engine.
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.5rem' }}>
            <label className="neuro-sunken" style={{ padding: '1rem', display: 'flex', alignItems: 'flex-start', gap: '0.75rem', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={semanticEnabled}
                onChange={(e) => setSemanticEnabled(e.target.checked)}
                style={{ marginTop: '0.2rem', accentColor: 'var(--cyan)' }}
              />
              <div>
                <div style={{ fontWeight: 700, fontSize: '0.86rem', color: 'var(--text-primary)' }}>Semantic Consistency</div>
                <div style={{ fontSize: '0.74rem', color: 'var(--text-dim)' }}>Cosine distance between sample text and class prototype representation</div>
              </div>
            </label>

            <label className="neuro-sunken" style={{ padding: '1rem', display: 'flex', alignItems: 'flex-start', gap: '0.75rem', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={neighborhoodEnabled}
                onChange={(e) => setNeighborhoodEnabled(e.target.checked)}
                style={{ marginTop: '0.2rem', accentColor: 'var(--cyan)' }}
              />
              <div>
                <div style={{ fontWeight: 700, fontSize: '0.86rem', color: 'var(--text-primary)' }}>Neighborhood Consistency</div>
                <div style={{ fontSize: '0.74rem', color: 'var(--text-dim)' }}>K-nearest neighbor class purity in DistilBERT representation space</div>
              </div>
            </label>

            <label className="neuro-sunken" style={{ padding: '1rem', display: 'flex', alignItems: 'flex-start', gap: '0.75rem', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={stabilityEnabled}
                onChange={(e) => setStabilityEnabled(e.target.checked)}
                style={{ marginTop: '0.2rem', accentColor: 'var(--cyan)' }}
              />
              <div>
                <div style={{ fontWeight: 700, fontSize: '0.86rem', color: 'var(--text-primary)' }}>Prediction Stability</div>
                <div style={{ fontSize: '0.74rem', color: 'var(--text-dim)' }}>Confidence shift under localized token masking perturbations</div>
              </div>
            </label>

            <label className="neuro-sunken" style={{ padding: '1rem', display: 'flex', alignItems: 'flex-start', gap: '0.75rem', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={densityEnabled}
                onChange={(e) => setDensityEnabled(e.target.checked)}
                style={{ marginTop: '0.2rem', accentColor: 'var(--cyan)' }}
              />
              <div>
                <div style={{ fontWeight: 700, fontSize: '0.86rem', color: 'var(--text-primary)' }}>Representation Density</div>
                <div style={{ fontSize: '0.74rem', color: 'var(--text-dim)' }}>Kernel density estimation and distance to centroid manifold</div>
              </div>
            </label>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.25rem', marginBottom: '1.5rem' }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
                SIGNAL WEIGHTING STRATEGY
              </label>
              <select
                value={weightingStrategy}
                onChange={(e) => setWeightingStrategy(e.target.value as any)}
                className="neuro-select"
              >
                <option value="learned_validation">Validation-Guided Optimization (Learned)</option>
                <option value="equal">Equal Uniform Weighting (1/4 each)</option>
              </select>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
                THRESHOLD CALIBRATION METHOD
              </label>
              <select
                value={calibrationMethod}
                onChange={(e) => setCalibrationMethod(e.target.value as any)}
                className="neuro-select"
              >
                <option value="youden_j">Youden's J-Statistic (Balanced Precision/Recall)</option>
                <option value="f1_optimal">F1-Optimal Threshold Calibration</option>
                <option value="target_fpr_0.01">Constrained FPR &le; 1.0% (High Utility)</option>
                <option value="target_fpr_0.05">Constrained FPR &le; 5.0%</option>
              </select>
            </div>
          </div>

          {/* Baselines Checkboxes */}
          <div className="neuro-sunken" style={{ padding: '1rem 1.25rem', marginBottom: '1.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
              <span style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                Compare Against Research Baselines
              </span>
              <input
                type="checkbox"
                checked={runBaselines}
                onChange={(e) => setRunBaselines(e.target.checked)}
                style={{ accentColor: 'var(--cyan)' }}
              />
            </div>
            {runBaselines && (
              <div style={{ display: 'flex', gap: '1.5rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={runFlare}
                    onChange={(e) => setRunFlare(e.target.checked)}
                    style={{ accentColor: 'var(--cyan)' }}
                  />
                  <span>FLARE (Representation Anomaly)</span>
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={runOnion}
                    onChange={(e) => setRunOnion(e.target.checked)}
                    style={{ accentColor: 'var(--cyan)' }}
                  />
                  <span>ONION (Language Model Perplexity)</span>
                </label>
              </div>
            )}
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <button onClick={() => setCurrentStep(2)} className="neuro-btn">
              ← Back
            </button>
            <button onClick={() => setCurrentStep(4)} className="neuro-btn neuro-btn-primary">
              Review Configuration →
            </button>
          </div>
        </div>
      )}

      {/* Step 4: Review & Launch */}
      {currentStep === 4 && (
        <div className="neuro-card" style={{ padding: '1.75rem' }}>
          <h3 style={{ fontFamily: 'var(--font-brand)', fontSize: '1.15rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.25rem' }}>
            Investigation Configuration Review
          </h3>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-dim)', marginBottom: '1.25rem' }}>
            Verify parameters before initiating the live asynchronous execution pipeline.
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginBottom: '1.75rem' }}>
            <div className="neuro-sunken" style={{ padding: '0.85rem 1.1rem', display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Dataset:</span>
              <strong style={{ color: '#ffffff' }}>{selectedDataset?.name || selectedDatasetId}</strong>
            </div>

            <div className="neuro-sunken" style={{ padding: '0.85rem 1.1rem', display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Attack Type:</span>
              <strong style={{ color: 'var(--rose-light)', textTransform: 'capitalize' }}>{attackType.replace(/_/g, ' ')}</strong>
            </div>

            <div className="neuro-sunken" style={{ padding: '0.85rem 1.1rem', display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Poison Injection Rate:</span>
              <strong style={{ color: 'var(--amber-light)', fontFamily: 'var(--font-mono)' }}>{(poisonRate * 100).toFixed(1)}%</strong>
            </div>

            <div className="neuro-sunken" style={{ padding: '0.85rem 1.1rem', display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Target Label & Seed:</span>
              <strong style={{ color: '#ffffff' }}>{targetLabel} (Seed: {seed})</strong>
            </div>

            <div className="neuro-sunken" style={{ padding: '0.85rem 1.1rem', display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Calibration & Weighting:</span>
              <strong style={{ color: 'var(--cyan-light)' }}>{calibrationMethod} &bull; {weightingStrategy}</strong>
            </div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <button onClick={() => setCurrentStep(3)} className="neuro-btn" disabled={submitting}>
              ← Back
            </button>
            <button
              onClick={handleLaunch}
              disabled={submitting}
              className="neuro-btn neuro-btn-emerald neuro-btn-lg"
              style={{ fontWeight: 800 }}
            >
              {submitting ? (
                <>
                  <span className="status-pip cyan" />
                  <span>Starting Pipeline...</span>
                </>
              ) : (
                <>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <polygon points="5 3 19 12 5 21 5 3" />
                  </svg>
                  <span>Start Live Investigation</span>
                </>
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
