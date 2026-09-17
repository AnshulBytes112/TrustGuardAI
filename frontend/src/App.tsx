import { useState, useEffect, useCallback } from 'react';
import type {
  DatasetItem,
  ScanItem,
  OverviewStats,
} from './api';
import {
  fetchHealth,
  fetchDatasets,
  fetchScans,
  fetchOverviewStats,
} from './api';
import { Sidebar } from './components/Sidebar';
import { LiveInvestigationView } from './components/LiveInvestigationView';
import { OverviewView } from './components/OverviewView';
import { DatasetsView } from './components/DatasetsView';
import { ScanView } from './components/ScanView';
import { SuspiciousSamplesView } from './components/SuspiciousSamplesView';
import { PurificationView } from './components/PurificationView';
import { BenchmarkView } from './components/BenchmarkView';
import { SampleInspectorModal } from './components/SampleInspectorModal';
import './index.css';

export function App() {
  const [activeView, setActiveView] = useState<string>('live');
  const [, setSystemHealthy] = useState<boolean>(false);
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [scans, setScans] = useState<ScanItem[]>([]);
  const [, setStats] = useState<OverviewStats | null>(null);
  const [inspectedSampleId, setInspectedSampleId] = useState<string | null>(null);

  // Cross-view state handoffs
  const [preselectedDatasetId, setPreselectedDatasetId] = useState<string | null>(null);
  const [preselectedScanId, setPreselectedScanId] = useState<string | null>(null);
  const [benchmarkRawId, setBenchmarkRawId] = useState<string | null>(null);
  const [benchmarkPurifiedId, setBenchmarkPurifiedId] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    try {
      const [healthRes, datasetsRes, scansRes, statsRes] = await Promise.all([
        fetchHealth().catch(() => ({ status: 'down' })),
        fetchDatasets().catch(() => []),
        fetchScans().catch(() => []),
        fetchOverviewStats().catch(() => null),
      ]);
      setSystemHealthy(healthRes.status === 'ok' || healthRes.status === 'healthy');
      setDatasets(datasetsRes);
      setScans(scansRes);
      if (statsRes) setStats(statsRes);
    } catch (err) {
      console.error('Failed loading system state:', err);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Active scan polling: if any scan is RUNNING or PENDING, poll every 2.5s
  useEffect(() => {
    const hasRunningScan = scans.some((s) => s.status === 'RUNNING' || s.status === 'PENDING');
    if (!hasRunningScan) return;

    const interval = setInterval(async () => {
      try {
        const [updatedScans, updatedStats] = await Promise.all([
          fetchScans(),
          fetchOverviewStats().catch(() => null),
        ]);
        setScans(updatedScans);
        if (updatedStats) setStats(updatedStats);
      } catch (e) {
        console.error('Polling error:', e);
      }
    }, 2500);

    return () => clearInterval(interval);
  }, [scans]);

  const handleLaunchScanFromDataset = (datasetId: string) => {
    setPreselectedDatasetId(datasetId);
    setActiveView('scans');
  };

  const handleViewScanSamples = (scanId: string) => {
    setPreselectedScanId(scanId);
    setActiveView('samples');
  };

  const handleNavigateToBenchmark = (rawDatasetId: string, purifiedDatasetId: string) => {
    setBenchmarkRawId(rawDatasetId);
    setBenchmarkPurifiedId(purifiedDatasetId);
    setActiveView('benchmark');
  };

  const handleScanCreated = (newScan: ScanItem) => {
    setScans((prev) => [newScan, ...prev]);
    setActiveView('scans');
  };

  return (
    <div className="app-layout">
      {/* Navigation Sidebar */}
      <Sidebar
        currentView={activeView}
        onSelectView={setActiveView}
      />

      {/* Main Content Area */}
      <div className="main-content">
        <main className="view-wrapper">
          {activeView === 'live' && (
            <LiveInvestigationView />
          )}

          {activeView === 'overview' && (
            <OverviewView
              datasets={datasets}
              scans={scans}
              onNavigate={setActiveView}
            />
          )}

          {activeView === 'datasets' && (
            <DatasetsView
              datasets={datasets}
              onRefresh={loadData}
              onLaunchScan={handleLaunchScanFromDataset}
              onInspectSample={setInspectedSampleId}
            />
          )}

          {activeView === 'scans' && (
            <ScanView
              datasets={datasets}
              scans={scans}
              initialDatasetId={preselectedDatasetId}
              onScanCreated={handleScanCreated}
              onViewScanSamples={handleViewScanSamples}
            />
          )}

          {activeView === 'samples' && (
            <SuspiciousSamplesView
              scans={scans}
              initialScanId={preselectedScanId}
              onInspectSample={setInspectedSampleId}
              onSampleStateChanged={loadData}
            />
          )}

          {activeView === 'purification' && (
            <PurificationView
              datasets={datasets}
              onPurificationComplete={() => loadData()}
              onNavigateToBenchmark={handleNavigateToBenchmark}
            />
          )}

          {activeView === 'benchmark' && (
            <BenchmarkView
              datasets={datasets}
              initialRawDatasetId={benchmarkRawId}
              initialPurifiedDatasetId={benchmarkPurifiedId}
            />
          )}
        </main>
      </div>

      {/* Deep Investigation XAI Modal */}
      {inspectedSampleId && (
        <SampleInspectorModal
          sampleId={inspectedSampleId}
          onClose={() => setInspectedSampleId(null)}
          onSampleUpdated={loadData}
        />
      )}
    </div>
  );
}

export default App;
