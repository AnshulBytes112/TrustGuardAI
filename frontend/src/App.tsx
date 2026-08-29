import { useEffect, useState } from 'react';
import { fetchExperiment, fetchExperiments, runExperiment, uploadDataset } from './api';
import type { ExperimentResult, ExperimentListItem, DatasetUploadResponse } from './api';
import './index.css';

function App() {
  const [experiments, setExperiments] = useState<ExperimentListItem[]>([]);
  const [selectedConfig, setSelectedConfig] = useState<string>('poisoned_baseline');
  const [activeExperiment, setActiveExperiment] = useState<ExperimentResult | null>(null);
  
  const [uploadedDataset, setUploadedDataset] = useState<DatasetUploadResponse | null>(null);
  const [useCustomDataset, setUseCustomDataset] = useState<boolean>(false);
  
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadExperimentsList();
  }, []);

  const loadExperimentsList = async () => {
    try {
      const list = await fetchExperiments();
      setExperiments(list);
    } catch (err) {
      console.error(err);
      setError('Failed to load experiments. Ensure the backend is running.');
    }
  };

  const handleRunOrLoad = async () => {
    setLoading(true);
    setError(null);
    setActiveExperiment(null);
    try {
      const customId = (useCustomDataset && uploadedDataset) ? uploadedDataset.dataset_id : undefined;
      const response = await runExperiment(selectedConfig, customId);
      setActiveExperiment(response.result);
      await loadExperimentsList(); // refresh list
    } catch (err: any) {
      setError(err.message || 'An error occurred while running the experiment.');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    
    setLoading(true);
    setError(null);
    setActiveExperiment(null); // Clear the old dashboard when a new file is uploaded
    
    try {
      const result = await uploadDataset(file);
      setUploadedDataset(result);
      setUseCustomDataset(true);
    } catch (err: any) {
      setError(err.message || 'Failed to upload dataset.');
      setUploadedDataset(null);
      setUseCustomDataset(false);
    } finally {
      setLoading(false);
      // Reset input so the same file can be uploaded again if needed
      e.target.value = '';
    }
  };

  const handleLoadArtifact = async (fingerprint: string) => {
    setLoading(true);
    setError(null);
    setActiveExperiment(null);
    try {
      const result = await fetchExperiment(fingerprint);
      setActiveExperiment(result);
    } catch (err: any) {
      setError(err.message || 'Failed to load artifact.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <header className="header">
        <div className="header-content">
          <div className="logo">TrustGuardAI</div>
          <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
            <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>Demo Showcase</span>
          </div>
        </div>
      </header>

      <main className="container">
        <section className="hero">
          <h1>Detect poisoned training data before it compromises your model.</h1>
          <p>
            TrustGuardAI analyzes text representations to identify anomalous samples associated with data poisoning.
          </p>
          
          <div className="controls" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
              <select 
                value={selectedConfig} 
                onChange={(e) => setSelectedConfig(e.target.value)}
                disabled={loading}
              >
                <option value="poisoned_baseline">Poisoned Baseline</option>
                <option value="clean_baseline">Clean Baseline</option>
              </select>
              <button onClick={handleRunOrLoad} disabled={loading}>
                {loading ? 'Running...' : 'Run Experiment'}
              </button>
            </div>
            
            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', alignItems: 'center', backgroundColor: 'var(--panel-bg)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
               <div>
                 <input 
                   type="file" 
                   id="dataset-upload"
                   accept=".jsonl" 
                   style={{ display: 'none' }}
                   onChange={handleFileUpload}
                   disabled={loading}
                 />
                 <label 
                   htmlFor="dataset-upload" 
                   style={{ 
                     cursor: loading ? 'not-allowed' : 'pointer', 
                     padding: '0.4rem 0.8rem', 
                     backgroundColor: 'var(--card-bg)', 
                     border: '1px solid var(--border-color)', 
                     borderRadius: '4px',
                     fontSize: '0.9rem'
                   }}
                 >
                   Upload Custom Dataset (.jsonl)
                 </label>
               </div>
               
               {uploadedDataset && (
                 <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontSize: '0.9rem' }}>
                   <input 
                     type="checkbox" 
                     checked={useCustomDataset} 
                     onChange={(e) => setUseCustomDataset(e.target.checked)}
                     disabled={loading}
                   />
                   Run on Uploaded Dataset
                 </label>
               )}
            </div>
          </div>
          
          {uploadedDataset && (
            <div style={{ marginTop: '1rem', backgroundColor: 'var(--panel-bg)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border-color)', textAlign: 'left', fontSize: '0.9rem' }}>
              <h4 style={{ margin: '0 0 0.5rem 0', color: 'var(--text-main)' }}>Custom Dataset Validated</h4>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', color: 'var(--text-muted)' }}>
                <div><strong>ID:</strong> {uploadedDataset.dataset_id}</div>
                <div><strong>File:</strong> {uploadedDataset.filename}</div>
                <div><strong>Total Samples:</strong> {uploadedDataset.total_samples}</div>
                <div><strong>Label Mode:</strong> {uploadedDataset.label_mode}</div>
                <div style={{ gridColumn: '1 / -1' }}>
                   <strong>Splits:</strong> {uploadedDataset.train_samples} Train / {uploadedDataset.validation_samples} Val / {uploadedDataset.test_samples} Test
                </div>
              </div>
            </div>
          )}

          {experiments.length > 0 && (
            <div style={{ marginTop: '1rem', display: 'flex', gap: '0.5rem', justifyContent: 'center', alignItems: 'center', flexWrap: 'wrap' }}>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Existing Artifacts:</span>
              {experiments.map(exp => (
                <button 
                  key={exp.experiment_fingerprint}
                  onClick={() => handleLoadArtifact(exp.experiment_fingerprint)}
                  disabled={loading}
                  style={{ 
                    padding: '0.2rem 0.5rem', 
                    fontSize: '0.8rem', 
                    backgroundColor: 'var(--card-bg)',
                    border: '1px solid var(--border-color)',
                    color: 'var(--text-main)'
                  }}
                >
                  {exp.experiment_name}
                </button>
              ))}
            </div>
          )}
        </section>

        {error && (
          <div className={`message-box ${error.includes('Validation split must contain') ? 'warning-box' : ''}`}>
            {error.includes('Validation split must contain') ? (
              <>
                <h3 style={{ color: 'var(--accent-color)', marginBottom: '1rem' }}>Limitation Acknowledged: Calibration Skipped</h3>
                <p>
                  As designed, the <strong>Clean Baseline</strong> mathematically halts at the threshold calibration step.
                </p>
                <p style={{ marginTop: '0.5rem', color: 'var(--text-muted)' }}>
                  Youden's J statistic requires both clean and poisoned samples in the Validation split to calculate an optimal ROC curve. Because the clean baseline has 0 poisoned samples, the pipeline intentionally aborts to avoid fabricating artificial metrics.
                </p>
              </>
            ) : (
              <>
                <h3 className="error-text">Execution Failed</h3>
                <p>{error}</p>
              </>
            )}
          </div>
        )}

        {!activeExperiment && !error && !loading && (
          <div className="message-box">
            <h3>No completed experiment available.</h3>
            <p style={{ color: 'var(--text-muted)' }}>Select a configuration and run the experiment to view results.</p>
          </div>
        )}

        {loading && (
          <div className="message-box">
            <h3>Pipeline is running...</h3>
            <p style={{ color: 'var(--text-muted)' }}>
              Preparing dataset... Running poisoning... Extracting representations... Fitting detector... Calibrating threshold... Evaluating...
            </p>
          </div>
        )}

        {activeExperiment && (
          <div className="dashboard-content animate-in">
            <div className="mb-2 text-center" style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              Experiment Fingerprint: {activeExperiment.experiment_fingerprint} | Status: Completed
            </div>

            <div className="pipeline-container">
              <h3 className="card-title">Pipeline Execution</h3>
              <div className="pipeline-steps">
                <div className="pipeline-line"></div>
                <div className="pipeline-step">
                  <div className="step-circle completed">✓</div>
                  <div className="step-label">Dataset Loaded</div>
                </div>
                <div className="pipeline-step">
                  <div className="step-circle completed">✓</div>
                  <div className="step-label">Poisoning Applied</div>
                </div>
                <div className="pipeline-step">
                  <div className="step-circle completed">✓</div>
                  <div className="step-label">Representations Extracted</div>
                </div>
                <div className="pipeline-step">
                  <div className="step-circle completed">✓</div>
                  <div className="step-label">FLARE Fitted</div>
                </div>
                <div className="pipeline-step">
                  <div className="step-circle completed">✓</div>
                  <div className="step-label">Threshold Calibrated</div>
                </div>
                <div className="pipeline-step">
                  <div className="step-circle completed">✓</div>
                  <div className="step-label">Test Evaluated</div>
                </div>
              </div>
            </div>

            <div className="dashboard-grid">
              <div className="card">
                <h3 className="card-title">Evaluation Metrics</h3>
                <div className="metric-grid">
                  <div className="metric-item">
                    <div className="metric-label">Precision</div>
                    <div className="metric-value">{(activeExperiment.pipeline_result.evaluation_report.precision * 100).toFixed(1)}%</div>
                  </div>
                  <div className="metric-item">
                    <div className="metric-label">Recall</div>
                    <div className="metric-value">{(activeExperiment.pipeline_result.evaluation_report.recall * 100).toFixed(1)}%</div>
                  </div>
                  <div className="metric-item">
                    <div className="metric-label">F1 Score</div>
                    <div className="metric-value">{(activeExperiment.pipeline_result.evaluation_report.f1 * 100).toFixed(1)}%</div>
                  </div>
                  <div className="metric-item">
                    <div className="metric-label">AUROC</div>
                    <div className="metric-value">-</div>
                  </div>
                </div>
                <div style={{ marginTop: '1.5rem', display: 'flex', gap: '1.5rem', justifyContent: 'center' }}>
                   <div style={{ textAlign: 'center' }}>
                      <div className="metric-label">Accuracy</div>
                      <div style={{ fontWeight: 600 }}>{(activeExperiment.pipeline_result.evaluation_report.accuracy * 100).toFixed(1)}%</div>
                   </div>
                   <div style={{ textAlign: 'center' }}>
                      <div className="metric-label">False Positives</div>
                      <div style={{ fontWeight: 600 }}>{activeExperiment.pipeline_result.evaluation_report.false_positive}</div>
                   </div>
                   <div style={{ textAlign: 'center' }}>
                      <div className="metric-label">False Negatives</div>
                      <div style={{ fontWeight: 600 }}>{activeExperiment.pipeline_result.evaluation_report.false_negative}</div>
                   </div>
                </div>
              </div>

              <div className="card">
                <h3 className="card-title">Poisoning Configuration</h3>
                <div className="key-value-list">
                  <div className="kv-pair">
                    <span className="kv-key">Attack Type</span>
                    <span className="kv-value">{activeExperiment.pipeline_result.poisoning_metadata?.attack_type || 'Disabled'}</span>
                  </div>
                  <div className="kv-pair">
                    <span className="kv-key">Poison Rate</span>
                    <span className="kv-value">{activeExperiment.pipeline_result.poisoning_metadata ? `${(activeExperiment.pipeline_result.poisoning_metadata.poison_rate * 100).toFixed(1)}%` : '0%'}</span>
                  </div>
                  <div className="kv-pair">
                    <span className="kv-key">Target Label</span>
                    <span className="kv-value">{activeExperiment.pipeline_result.poisoning_metadata?.target_label || 'N/A'}</span>
                  </div>
                  <div className="kv-pair">
                    <span className="kv-key">Seed</span>
                    <span className="kv-value">{activeExperiment.pipeline_result.poisoning_metadata?.seed || 'N/A'}</span>
                  </div>
                  <div className="kv-pair">
                    <span className="kv-key">Poisoned Samples</span>
                    <span className="kv-value">{activeExperiment.pipeline_result.evaluation_report.poisoned_samples}</span>
                  </div>
                  <div className="kv-pair">
                    <span className="kv-key">Clean Samples</span>
                    <span className="kv-value">{activeExperiment.pipeline_result.evaluation_report.clean_samples}</span>
                  </div>
                </div>
              </div>

              <div className="card">
                <h3 className="card-title">Dataset Metadata</h3>
                <div className="key-value-list">
                  <div className="kv-pair">
                    <span className="kv-key">Dataset ID</span>
                    <span className="kv-value">{activeExperiment.dataset_id}</span>
                  </div>
                  <div className="kv-pair">
                    <span className="kv-key">Version</span>
                    <span className="kv-value">{activeExperiment.dataset_version}</span>
                  </div>
                  <div className="kv-pair">
                    <span className="kv-key">Total Evaluated (TEST)</span>
                    <span className="kv-value">{activeExperiment.pipeline_result.evaluation_report.total_evaluated}</span>
                  </div>
                  <div className="kv-pair">
                    <span className="kv-key">Status</span>
                    <span className="kv-value" style={{ color: 'var(--text-muted)' }}>
                      {activeExperiment.dataset_id.startsWith("custom_") ? "Custom User Upload" : "Synthetic Demo Fixture"}
                    </span>
                  </div>
                </div>
              </div>

              <div className="card">
                <h3 className="card-title">Detection Details</h3>
                <div className="key-value-list">
                  <div className="kv-pair">
                    <span className="kv-key">Representation Model</span>
                    <span className="kv-value">{activeExperiment.pipeline_result.representation_config.model_name}</span>
                  </div>
                  <div className="kv-pair">
                    <span className="kv-key">Detector</span>
                    <span className="kv-value">FLARE</span>
                  </div>
                  <div className="kv-pair">
                    <span className="kv-key">Layers Used</span>
                    <span className="kv-value">{activeExperiment.pipeline_result.detector_config.layers.join(', ')}</span>
                  </div>
                  <div className="kv-pair">
                    <span className="kv-key">Calibrated Threshold</span>
                    <span className="kv-value">{activeExperiment.pipeline_result.threshold.toFixed(4)}</span>
                  </div>
                </div>
              </div>
            </div>

            {activeExperiment.pipeline_result.detection_result.sample_ids.length > 0 && (
              <div className="card mb-2">
                <h3 className="card-title">Sample Investigation</h3>
                <div className="table-container">
                  <table>
                    <thead>
                      <tr>
                        <th>Sample ID</th>
                        <th>Anomaly Score</th>
                        <th>Prediction</th>
                      </tr>
                    </thead>
                    <tbody>
                      {activeExperiment.pipeline_result.detection_result.sample_ids.slice(0, 10).map((id, index) => (
                        <tr key={id}>
                          <td>{id}</td>
                          <td>{activeExperiment.pipeline_result.detection_result.scores[index].toFixed(4)}</td>
                          <td>
                            <span className={`badge ${activeExperiment.pipeline_result.detection_result.is_anomalous[index] ? 'anomalous' : 'clean'}`}>
                              {activeExperiment.pipeline_result.detection_result.is_anomalous[index] ? 'Suspicious' : 'Clean'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div style={{ marginTop: '1rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  * Showing top 10 evaluated test samples.
                </div>
              </div>
            )}

            <div className="card mb-2">
              <h3 className="card-title">How It Works</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', color: 'var(--text-muted)', fontSize: '0.95rem' }}>
                <p><strong style={{ color: '#fff' }}>1. Poisoning:</strong> Controlled trigger-based poisoning creates known ground truth for evaluation.</p>
                <p><strong style={{ color: '#fff' }}>2. Representation:</strong> DistilBERT converts text into contextual multi-layer representations.</p>
                <p><strong style={{ color: '#fff' }}>3. Detection:</strong> FLARE identifies samples whose representations behave anomalously against the clean <strong>TRAIN</strong> split.</p>
                <p><strong style={{ color: '#fff' }}>4. Calibration:</strong> The threshold is calibrated using Youden's J on the <strong>VALIDATION</strong> data.</p>
                <p><strong style={{ color: '#fff' }}>5. Evaluation:</strong> The frozen threshold is applied to the <strong>TEST</strong> set. Test labels are not used during detector fitting or threshold selection.</p>
              </div>
            </div>
            
            <div className="mb-2 text-center" style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Note: This dashboard demonstrates the core research pipeline. Features such as Risk/XAI, Purification, Retraining, GNNs, Image Support, and Production Databases remain future work.
            </div>
          </div>
        )}
      </main>
    </>
  );
}

export default App;
