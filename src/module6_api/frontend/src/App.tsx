import React, { useState } from 'react';
import { Header } from './components/common/Header';
import { Sidebar, TabType } from './components/common/Sidebar';
import { OverviewDashboard } from './components/overview/OverviewDashboard';
import { TransactionExplorer } from './components/explorer/TransactionExplorer';
import { InteractiveGraphView } from './components/graph/InteractiveGraphView';
import { ExplainabilityPanel } from './components/explainability/ExplainabilityPanel';
import { TriageQueueView } from './components/hitl/TriageQueueView';
import { AnalyticsView } from './components/analytics/AnalyticsView';
import { ReviewModal } from './components/hitl/ReviewModal';
import { FeedbackResponse, TransactionResponse } from './types/api';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [selectedTxId, setSelectedTxId] = useState<string>('230425980');
  const [reviewModalTx, setReviewModalTx] = useState<TransactionResponse | null>(null);

  const handleNavigate = (tab: TabType, txId?: string) => {
    if (txId) {
      setSelectedTxId(txId);
    }
    setActiveTab(tab);
  };

  return (
    <div className="app-container">
      {/* Navigation Sidebar */}
      <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />

      {/* Main Workspace Area */}
      <div className="main-content">
        <Header />

        <main className="content-body">
          {activeTab === 'overview' && (
            <OverviewDashboard onNavigate={handleNavigate} />
          )}

          {activeTab === 'explorer' && (
            <TransactionExplorer
              onSelectTx={(txId) => handleNavigate('graph', txId)}
              onNavigateTab={handleNavigate}
              onOpenReviewModal={(tx) => setReviewModalTx(tx)}
            />
          )}

          {activeTab === 'graph' && (
            <InteractiveGraphView
              initialTxId={selectedTxId}
              onNavigateTab={handleNavigate}
            />
          )}

          {activeTab === 'explain' && (
            <ExplainabilityPanel
              txId={selectedTxId}
              onNavigateTab={handleNavigate}
            />
          )}

          {activeTab === 'hitl' && (
            <TriageQueueView onNavigateTab={handleNavigate} />
          )}

          {activeTab === 'analytics' && <AnalyticsView />}
        </main>
      </div>

      {/* Global Review Modal */}
      {reviewModalTx && (
        <ReviewModal
          txId={reviewModalTx.tx_id}
          timestep={reviewModalTx.timestep}
          predictedProb={reviewModalTx.predicted_prob}
          riskLevel={reviewModalTx.risk_level}
          isOpen={true}
          onClose={() => setReviewModalTx(null)}
          onSuccess={(fb: FeedbackResponse) => {
            console.log('Feedback submitted:', fb);
            setReviewModalTx(null);
          }}
        />
      )}
    </div>
  );
};

export default App;
