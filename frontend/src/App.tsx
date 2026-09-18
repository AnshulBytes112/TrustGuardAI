import { useState, useEffect, useCallback } from 'react';
import {
  fetchHealth,
  fetchSystemInfo,
  fetchLiveJobs,
} from './api';
import type {
  SystemInfo,
  LiveJobSummary,
} from './api';
import { AppShell } from './components/layout/AppShell';
import { DashboardView } from './components/operations/DashboardView';
import { DatasetsView } from './components/operations/DatasetsView';
import { NewInvestigationView } from './components/operations/NewInvestigationView';
import { JobsHistoryView } from './components/operations/JobsHistoryView';
import { LiveInvestigationView } from './components/research/LiveInvestigationView';
import { SignalDiagnosticsView } from './components/research/SignalDiagnosticsView';
import { ThresholdParetoView } from './components/research/ThresholdParetoView';
import { AttackVariantsView } from './components/research/AttackVariantsView';
import { CrossDatasetView } from './components/research/CrossDatasetView';
import { GpuComputeView } from './components/system/GpuComputeView';
import { ArtifactExplorerView } from './components/system/ArtifactExplorerView';
import { SystemHealthView } from './components/system/SystemHealthView';
import './index.css';

export function App() {
  const [activeTab, setActiveTab] = useState<string>('dashboard');
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [investigationDatasetId, setInvestigationDatasetId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>('');

  // System & Health State
  const [backendOnline, setBackendOnline] = useState<boolean>(true);
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [liveJobs, setLiveJobs] = useState<LiveJobSummary[]>([]);

  const loadGlobalSystemState = useCallback(async () => {
    try {
      const [healthRes, sysRes, jobsRes] = await Promise.allSettled([
        fetchHealth(),
        fetchSystemInfo(),
        fetchLiveJobs(),
      ]);

      setBackendOnline(healthRes.status === 'fulfilled' && healthRes.value.status === 'ok');
      if (sysRes.status === 'fulfilled') setSystemInfo(sysRes.value);
      if (jobsRes.status === 'fulfilled') setLiveJobs(jobsRes.value);
    } catch (err) {
      console.error('System state polling error', err);
      setBackendOnline(false);
    }
  }, []);

  useEffect(() => {
    loadGlobalSystemState();
    const interval = setInterval(loadGlobalSystemState, 5000);
    return () => clearInterval(interval);
  }, [loadGlobalSystemState]);

  const handleNavigate = (tabId: string, jobId?: string) => {
    if (jobId) {
      setSelectedJobId(jobId);
      setActiveTab('live_pipeline');
    } else {
      setActiveTab(tabId);
    }
  };

  const handleLaunchInvestigationFromDataset = (datasetId: string) => {
    setInvestigationDatasetId(datasetId);
    setActiveTab('investigate_new');
  };

  const handleInvestigationStarted = (newJobId: string) => {
    setSelectedJobId(newJobId);
    loadGlobalSystemState();
    setActiveTab('live_pipeline');
  };

  const activeJobCount = liveJobs.filter((j) => j.status === 'RUNNING' || j.status === 'CREATED').length;

  // Determine which job to inspect in Live Pipeline view if none explicitly selected
  const activeJobIdToInspect = selectedJobId || (liveJobs.length > 0 ? liveJobs[0].job_id : null);

  return (
    <AppShell
      activeTab={activeTab}
      onSelectTab={setActiveTab}
      systemInfo={systemInfo}
      backendOnline={backendOnline}
      activeJobCount={activeJobCount}
      searchQuery={searchQuery}
      onSearchChange={setSearchQuery}
    >
      {/* 1. Dashboard View */}
      {activeTab === 'dashboard' && (
        <DashboardView
          onNavigate={handleNavigate}
          systemInfo={systemInfo}
        />
      )}

      {/* 2. Dataset Studio */}
      {activeTab === 'datasets' && (
        <DatasetsView
          onLaunchInvestigation={handleLaunchInvestigationFromDataset}
          searchQuery={searchQuery}
        />
      )}

      {/* 3. New Investigation Wizard */}
      {activeTab === 'investigate_new' && (
        <NewInvestigationView
          initialDatasetId={investigationDatasetId}
          onInvestigationStarted={handleInvestigationStarted}
          onCancel={() => setActiveTab('dashboard')}
        />
      )}

      {/* 4. Live Pipeline Console */}
      {activeTab === 'live_pipeline' && (
        activeJobIdToInspect ? (
          <LiveInvestigationView
            key={activeJobIdToInspect}
            jobId={activeJobIdToInspect}
            onBackToJobs={() => setActiveTab('jobs')}
            onNewJob={() => setActiveTab('investigate_new')}
          />
        ) : (
          <NewInvestigationView
            initialDatasetId={investigationDatasetId}
            onInvestigationStarted={handleInvestigationStarted}
            onCancel={() => setActiveTab('dashboard')}
          />
        )
      )}

      {/* 5. Investigation Jobs History */}
      {activeTab === 'jobs' && (
        <JobsHistoryView
          onInspectJob={(jId) => handleNavigate('live_pipeline', jId)}
          onNewJob={() => setActiveTab('investigate_new')}
          searchQuery={searchQuery}
        />
      )}

      {/* 6. Signal Diagnostics */}
      {activeTab === 'signal_diagnostics' && (
        <SignalDiagnosticsView />
      )}

      {/* 7. Threshold & Pareto */}
      {activeTab === 'threshold_pareto' && (
        <ThresholdParetoView />
      )}

      {/* 8. Attack Variants */}
      {activeTab === 'attack_variants' && (
        <AttackVariantsView />
      )}

      {/* 9. Cross-Dataset Generalization */}
      {activeTab === 'cross_dataset' && (
        <CrossDatasetView />
      )}

      {/* 10. GPU & Compute */}
      {activeTab === 'gpu_compute' && (
        <GpuComputeView systemInfo={systemInfo} />
      )}

      {/* 11. Artifact Explorer */}
      {activeTab === 'artifacts' && (
        <ArtifactExplorerView />
      )}

      {/* 12. System Health */}
      {activeTab === 'system_health' && (
        <SystemHealthView />
      )}
    </AppShell>
  );
}

export default App;
