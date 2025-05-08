import { useState, useEffect } from 'react'
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Link,
} from 'react-router-dom'
import './App.css'

const MODEL_OPTIONS = [
  { label: 'Baseline', value: 'baseline' },
  { label: 'Naive Bayes', value: 'nb' },
  { label: 'SVM', value: 'svm' },
];

function useModelStatus() {
  const [modelsReady, setModelsReady] = useState(true);
  const [missingFiles, setMissingFiles] = useState([]);
  const checkStatus = async () => {
    try {
      const res = await fetch('http://localhost:5002/status');
      const data = await res.json();
      setModelsReady(data.models_ready);
      setMissingFiles(data.missing_files || []);
    } catch {
      setModelsReady(false);
      setMissingFiles(['API not reachable']);
    }
  };
  useEffect(() => { checkStatus(); }, []); // Only on initial load
  return { modelsReady, missingFiles, checkStatus };
}

function WarningBanner({ modelsReady, missingFiles }) {
  if (modelsReady) return null;
  return (
    <div className="warning-banner">
      <b>Model files missing.</b> Please run training first.<br />
      Missing: {missingFiles.join(', ')}
    </div>
  );
}

function DetectPage({ modelsReady }) {
  const [email, setEmail] = useState('');
  const [model, setModel] = useState('nb');
  const [useGemini, setUseGemini] = useState(false);
  const [geminiApiKey, setGeminiApiKey] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  // 动态调整文本框高度
  const adjustTextareaHeight = (element) => {
    if (element) {
      element.style.height = 'auto';
      element.style.height = `${element.scrollHeight}px`;
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setResult(null);
    if (!email.trim()) {
      setError('Please enter email content!');
      return;
    }
    if (useGemini && !geminiApiKey.trim()) {
      setError('Please enter Gemini API key!');
      return;
    }
    setLoading(true);
    try {
      const res = await fetch('http://localhost:5002/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          email, 
          model, 
          use_gemini: useGemini,
          gemini_api_key: geminiApiKey 
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'API Error');
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fancy-bg">
      <div className="detector-card">
        <h1 className="title">🚀 Spam Detector</h1>
        <form onSubmit={handleSubmit}>
          <textarea
            className="email-input"
            rows={5}
            placeholder="Enter email content..."
            value={email}
            onChange={e => {
              setEmail(e.target.value);
              adjustTextareaHeight(e.target);
            }}
            onInput={e => adjustTextareaHeight(e.target)}
            disabled={loading || !modelsReady}
          />
          <div className="model-select-row">
            <span>Model:</span>
            <select
              value={model}
              onChange={e => setModel(e.target.value)}
              disabled={loading || !modelsReady || useGemini}
              className="model-select"
            >
              {MODEL_OPTIONS.map(opt => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>
          <div className="gemini-toggle">
            <label>
              <input
                type="checkbox"
                checked={useGemini}
                onChange={e => setUseGemini(e.target.checked)}
                disabled={loading}
              />
              Use Gemini Analysis
            </label>
          </div>
          {useGemini && (
            <input
              type="password"
              className="api-key-input"
              placeholder="Enter Gemini API Key"
              value={geminiApiKey}
              onChange={e => setGeminiApiKey(e.target.value)}
              disabled={loading}
            />
          )}
          <button className="detect-btn" type="submit" disabled={loading || (!modelsReady && !useGemini)}>
            {loading ? 'Detecting...' : 'Detect'}
          </button>
        </form>
        {error && <div className="error-tip">❌ {error}</div>}
        {result && (
          <div className={`result-card ${result.result === 'SPAM' ? 'spam' : 'ham'}`}>
            <div className="result-title">
              {result.result === 'SPAM' ? '🚨 Spam!' : '✅ Not Spam'}
            </div>
            {!useGemini && (
              <>
                <div className="prob-bar-wrap">
                  <div className="prob-bar spam-bar" style={{width: `${result.spam_probability*100}%`}} />
                  <div className="prob-bar ham-bar" style={{width: `${result.ham_probability*100}%`}} />
                </div>
                <div className="prob-labels">
                  <span>Spam: {(result.spam_probability*100).toFixed(1)}%</span>
                  <span>Ham: {(result.ham_probability*100).toFixed(1)}%</span>
                </div>
              </>
            )}
            {result.analysis && (
              <div className="gemini-analysis">
                <div className="analysis-content">
                  {result.analysis.split('\n').map((line, index) => {
                    if (line.startsWith('Decision:')) {
                      return <div key={index} className="decision-line">{line}</div>;
                    } else if (line.startsWith('Reason:')) {
                      return <div key={index} className="reason-line">{line}</div>;
                    }
                    return <div key={index}>{line}</div>;
                  })}
                </div>
              </div>
            )}
          </div>
        )}
        <footer className="footer">Made by Kangwei Zhu, Kaiwen Zhu</footer>
      </div>
    </div>
  )
}

function ManagePage({ modelsReady, checkStatus }) {
  const [trainStatus, setTrainStatus] = useState('');
  const [testStatus, setTestStatus] = useState('');
  const [loading, setLoading] = useState(false);

  const handleTrain = async () => {
    setLoading(true);
    setTrainStatus('Training...');
    try {
      const res = await fetch('http://localhost:5002/train', { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.message || 'Train failed');
      setTrainStatus('✅ ' + data.message);
      await checkStatus(); // Only refresh after training
    } catch (e) {
      setTrainStatus('❌ ' + e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleTest = async () => {
    setLoading(true);
    setTestStatus('Testing...');
    try {
      const res = await fetch('http://localhost:5002/test', { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.message || 'Test failed');
      setTestStatus('✅ ' + data.message);
    } catch (e) {
      setTestStatus('❌ ' + e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fancy-bg">
      <div className="detector-card">
        <h1 className="title">🛠️ Model Management</h1>
        <button className="detect-btn" onClick={handleTrain} disabled={loading}>
          {loading ? 'Training...' : 'Start Training'}
        </button>
        {trainStatus && <div className="result-card ham">{trainStatus}</div>}
        <button className="detect-btn" onClick={handleTest} disabled={loading || !modelsReady}>
          {loading ? 'Testing...' : 'Start Testing'}
        </button>
        {testStatus && <div className="result-card ham">{testStatus}</div>}
        <footer className="footer">CS410 Spam Detector · Powered by React & Flask</footer>
      </div>
    </div>
  );
}

function ShowcasePage() {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchReport = async () => {
      try {
        const res = await fetch('http://localhost:5002/latest_report');
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Failed to fetch report');
        setReport(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchReport();
  }, []);

  const renderMetric = (value) => {
    if (value === null) return 'N/A';
    return `${(value * 100).toFixed(2)}%`;
  };

  const renderModelCard = (modelName, data) => (
    <div className="model-card" key={modelName}>
      <h3 className="model-name">{modelName}</h3>
      <div className="metrics-grid">
        <div className="metric">
          <span className="metric-label">Accuracy</span>
          <span className="metric-value">{renderMetric(data.accuracy)}</span>
        </div>
        <div className="metric">
          <span className="metric-label">Precision</span>
          <span className="metric-value">{renderMetric(data.precision)}</span>
        </div>
        <div className="metric">
          <span className="metric-label">Recall</span>
          <span className="metric-value">{renderMetric(data.recall)}</span>
        </div>
        <div className="metric">
          <span className="metric-label">F1 Score</span>
          <span className="metric-value">{renderMetric(data.f1_score)}</span>
        </div>
      </div>
      {data.confusion_matrix && (
        <div className="confusion-matrix">
          <h4>Confusion Matrix</h4>
          <pre>{data.confusion_matrix}</pre>
        </div>
      )}
    </div>
  );

  const renderImages = () => {
    if (!report?.images) return null;
    const { performance, baseline_cm, nb_cm, svm_cm } = report.images;

    return (
      <div className="images-section">
        {performance && (
          <div className="image-card">
            <h3>Performance Comparison</h3>
            <img src={performance} alt="Model Performance Comparison" />
          </div>
        )}
        <div className="confusion-matrices">
          {baseline_cm && (
            <div className="image-card">
              <h3>Baseline Confusion Matrix</h3>
              <img src={baseline_cm} alt="Baseline Confusion Matrix" />
            </div>
          )}
          {nb_cm && (
            <div className="image-card">
              <h3>Naive Bayes Confusion Matrix</h3>
              <img src={nb_cm} alt="Naive Bayes Confusion Matrix" />
            </div>
          )}
          {svm_cm && (
            <div className="image-card">
              <h3>SVM Confusion Matrix</h3>
              <img src={svm_cm} alt="SVM Confusion Matrix" />
            </div>
          )}
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="fancy-bg">
        <div className="detector-card">
          <h1 className="title">📊 Model Showcase</h1>
          <div className="loading">Loading latest test results...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="fancy-bg">
        <div className="detector-card">
          <h1 className="title">📊 Model Showcase</h1>
          <div className="error-tip">❌ {error}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="fancy-bg">
      <div className="showcase-card">
        <h1 className="title">📊 Model Showcase</h1>
        {report?.timestamp && (
          <div className="timestamp">
            Last updated: {report.timestamp}
          </div>
        )}
        {renderImages()}
        <div className="models-grid">
          {Object.entries(report?.models || {}).map(([modelName, data]) => 
            renderModelCard(modelName, data)
          )}
        </div>
        <footer className="footer">CS410 Spam Detector · Powered by React & Flask</footer>
      </div>
    </div>
  );
}

function NavBar() {
  return (
    <nav className="navbar">
      <Link to="/" className="nav-link">Detect</Link>
      <Link to="/manage" className="nav-link">Model Management</Link>
      <Link to="/showcase" className="nav-link">Showcase</Link>
    </nav>
  );
}

function App() {
  const { modelsReady, missingFiles, checkStatus } = useModelStatus();
  return (
    <Router>
      <WarningBanner modelsReady={modelsReady} missingFiles={missingFiles} />
      <NavBar />
      <Routes>
        <Route path="/" element={<DetectPage modelsReady={modelsReady} />} />
        <Route path="/manage" element={<ManagePage modelsReady={modelsReady} checkStatus={checkStatus} />} />
        <Route path="/showcase" element={<ShowcasePage />} />
      </Routes>
    </Router>
  );
}

export default App
