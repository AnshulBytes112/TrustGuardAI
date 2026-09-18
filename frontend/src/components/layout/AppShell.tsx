import React, { useState } from 'react';
import { Sidebar } from './Sidebar';
import { TopBar } from './TopBar';
import type { SystemInfo } from '../../api';

interface AppShellProps {
  activeTab: string;
  onSelectTab: (tabId: string) => void;
  systemInfo: SystemInfo | null;
  backendOnline: boolean;
  activeJobCount: number;
  searchQuery: string;
  onSearchChange: (q: string) => void;
  children: React.ReactNode;
}

const TAB_TITLES: Record<string, { title: string; category: string }> = {
  dashboard: { title: 'Security Command Center', category: 'Operations' },
  datasets: { title: 'Dataset Studio & Ingestion', category: 'Operations' },
  investigate_new: { title: 'New Investigation Wizard', category: 'Operations' },
  live_pipeline: { title: 'Live Investigation Console', category: 'Operations' },
  jobs: { title: 'Investigation Jobs History', category: 'Operations' },
  signal_diagnostics: { title: 'Signal Diagnostics & Distributions', category: 'Research' },
  threshold_pareto: { title: 'Continuous Threshold & Pareto Analysis', category: 'Research' },
  attack_variants: { title: 'Attack Mechanism Benchmark Matrix', category: 'Research' },
  cross_dataset: { title: 'Cross-Dataset Generalization Suite', category: 'Research' },
  gpu_compute: { title: 'GPU Hardware & Compute Telemetry', category: 'System' },
  artifacts: { title: 'Research Artifacts & Data Browser', category: 'System' },
  system_health: { title: 'System Health & Node Connectivity', category: 'System' },
};

export const AppShell: React.FC<AppShellProps> = ({
  activeTab,
  onSelectTab,
  systemInfo,
  backendOnline,
  activeJobCount,
  searchQuery,
  onSearchChange,
  children,
}) => {
  const [isMobileNavOpen, setIsMobileNavOpen] = useState(false);

  const currentMeta = TAB_TITLES[activeTab] || {
    title: 'Security Console',
    category: 'TrustGuardAI',
  };

  return (
    <div
      style={{
        display: 'flex',
        height: '100vh',
        width: '100vw',
        backgroundColor: 'var(--bg-canvas)',
        overflow: 'hidden',
      }}
    >
      {/* Sidebar Navigation */}
      <Sidebar
        activeTab={activeTab}
        onSelectTab={onSelectTab}
        isOpenMobile={isMobileNavOpen}
        onCloseMobile={() => setIsMobileNavOpen(false)}
        activeJobCount={activeJobCount}
      />

      {/* Main Content Area */}
      <div
        style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          height: '100vh',
          overflow: 'hidden',
          backgroundColor: 'var(--bg-canvas)',
        }}
      >
        {/* Top Bar */}
        <TopBar
          activeTabTitle={currentMeta.title}
          categoryTitle={currentMeta.category}
          systemInfo={systemInfo}
          backendOnline={backendOnline}
          onOpenMobileMenu={() => setIsMobileNavOpen(true)}
          onQuickLaunch={() => onSelectTab('investigate_new')}
          searchQuery={searchQuery}
          onSearchChange={onSearchChange}
        />

        {/* Scrollable View Canvas */}
        <main
          style={{
            flex: 1,
            overflowY: 'auto',
            padding: '1.75rem 2rem 3rem',
            backgroundColor: 'var(--bg-canvas)',
          }}
        >
          <div style={{ maxWidth: '1600px', margin: '0 auto', width: '100%' }}>
            {children}
          </div>
        </main>
      </div>
    </div>
  );
};
