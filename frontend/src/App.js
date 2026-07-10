import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const INITIAL_FEATURES = {
  mdvp_fo: '', mdvp_fhi: '', mdvp_flo: '',
  mdvp_jitter_pct: '', mdvp_jitter_abs: '',
  mdvp_rap: '', mdvp_ppq: '', jitter_ddp: '',
  mdvp_shimmer: '', mdvp_shimmer_db: '',
  shimmer_apq3: '', shimmer_apq5: '',
  mdvp_apq: '', shimmer_dda: '',
  nhr: '', hnr: '', rpde: '', dfa: '',
  spread1: '', spread2: '', d2: '', ppe: ''
};

const LABELS = {
  mdvp_fo: 'Avg Pitch (Hz)', mdvp_fhi: 'Max Pitch (Hz)',
  mdvp_flo: 'Min Pitch (Hz)', mdvp_jitter_pct: 'Jitter (%)',
  mdvp_jitter_abs: 'Jitter (Abs)', mdvp_rap: 'RAP',
  mdvp_ppq: 'PPQ', jitter_ddp: 'DDP',
  mdvp_shimmer: 'Shimmer', mdvp_shimmer_db: 'Shimmer (dB)',
  shimmer_apq3: 'APQ3', shimmer_apq5: 'APQ5',
  mdvp_apq: 'APQ', shimmer_dda: 'DDA',
  nhr: 'NHR', hnr: 'HNR',
  rpde: 'RPDE', dfa: 'DFA',
  spread1: 'Spread1', spread2: 'Spread2',
  d2: 'D2', ppe: 'PPE'
};

function VoiceTab() {
  const [features, setFeatures] = useState(INITIAL_FEATURES);
  const [result, setResult]     = useState(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);

  const handleChange = (e) => {
    setFeatures({ ...features, [e.target.name]: e.target.value });
  };

  const loadSample = () => {
    setFeatures({
      mdvp_fo: 119.992, mdvp_fhi: 157.302, mdvp_flo: 74.997,
      mdvp_jitter_pct: 0.00784, mdvp_jitter_abs: 0.00007,
      mdvp_rap: 0.0037, mdvp_ppq: 0.00554, jitter_ddp: 0.01109,
      mdvp_shimmer: 0.04374, mdvp_shimmer_db: 0.426,
      shimmer_apq3: 0.02182, shimmer_apq5: 0.0313,
      mdvp_apq: 0.02971, shimmer_dda: 0.06545,
      nhr: 0.02211, hnr: 21.033, rpde: 0.414783,
      dfa: 0.815285, spread1: -4.813031, spread2: 0.266482,
      d2: 2.301442, ppe: 0.284654
    });
    setResult(null); setError(null);
  };

  const handleSubmit = async () => {
    setLoading(true); setResult(null); setError(null);
    try {
      const payload = {};
      for (let key in features) payload[key] = parseFloat(features[key]);
      const res = await axios.post('http://127.0.0.1:8000/predict/voice', payload);
      setResult(res.data);
    } catch {
      setError('Backend se connection nahi ho saka. Server chal raha hai?');
    }
    setLoading(false);
  };

  return (
    <div className="card">
      <div className="card-header">
        <h2>Voice Features Input</h2>
        <button className="btn-sample" onClick={loadSample}>Load Sample Data</button>
      </div>
      <div className="grid">
        {Object.keys(INITIAL_FEATURES).map((key) => (
          <div className="field" key={key}>
            <label>{LABELS[key]}</label>
            <input type="number" name={key} value={features[key]}
              onChange={handleChange} step="any" placeholder="0.00" />
          </div>
        ))}
      </div>
      <button className="btn-predict" onClick={handleSubmit} disabled={loading}>
        {loading ? 'Analyzing...' : '🔍 Analyze Voice Features'}
      </button>
      {error && <div className="card error"><p>⚠️ {error}</p></div>}
      {result && <ResultCard result={result} />}
    </div>
  );
}

function SpiralTab() {
  const [file, setFile]       = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  const handleFile = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null); setError(null);
  };

  const handleSubmit = async () => {
    if (!file) { setError('Pehle ek spiral image select karein!'); return; }
    setLoading(true); setResult(null); setError(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await axios.post(
        'http://127.0.0.1:8000/predict/spiral', formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );
      setResult(res.data);
    } catch {
      setError('Backend se connection nahi ho saka. Server chal raha hai?');
    }
    setLoading(false);
  };

  return (
    <div className="card">
      <div className="card-header">
        <h2>Spiral Drawing Analysis</h2>
      </div>
      <p className="subtitle">
        Upload a hand-drawn spiral sketch from the patient. 
  Our AI model analyzes tremor patterns in the drawing 
  to detect early Parkinson's indicators.
      </p>
      <div className="upload-area">
        <input type="file" accept="image/*"
          onChange={handleFile} id="spiral-upload" hidden />
        <label htmlFor="spiral-upload" className="upload-label">
          {preview
            ? <img src={preview} alt="preview" className="preview-img" />
            : <div className="upload-placeholder">
                <span>🖼️</span>
                <p>Click to upload spiral drawing</p>
                <small>PNG, JPG supported</small>
              </div>
          }
        </label>
      </div>
      <button className="btn-predict" onClick={handleSubmit} disabled={loading}>
        {loading ? 'Analyzing...' : '🔍 Analyze Spiral Drawing'}
      </button>
      {error && <div className="error-box"><p>⚠️ {error}</p></div>}
      {result && <ResultCard result={result} />}
    </div>
  );
}

function ResultCard({ result }) {
  return (
    <div className={`result-card ${result.risk_label}`}>
      <h2>Analysis Result</h2>
      <div className="result-main">
        <span className="result-icon">
          {result.risk_label === 'high_risk' ? '🔴' : '🟢'}
        </span>
        <span className="result-text">{result.prediction}</span>
      </div>
      <div className="result-details">
        <div className="detail-box">
          <span>Probability</span>
          <strong>{(result.probability_pd * 100).toFixed(1)}%</strong>
        </div>
        <div className="detail-box">
          <span>Confidence</span>
          <strong>{result.confidence.toUpperCase()}</strong>
        </div>
      </div>
      <p className="disclaimer">⚕️ {result.disclaimer}</p>
    </div>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState('voice');

  return (
    <div className="app">
      <header className="header">
        <h1>🧠 Parkinson's Disease Screening Tool</h1>
        <p>AI-powered early detection using voice biomarkers & spiral drawings</p>
      </header>
      <div className="tabs">
        <button
          className={`tab ${activeTab === 'voice' ? 'active' : ''}`}
          onClick={() => setActiveTab('voice')}>
          🎙️ Voice Analysis
        </button>
        <button
          className={`tab ${activeTab === 'spiral' ? 'active' : ''}`}
          onClick={() => setActiveTab('spiral')}>
          🌀 Spiral Drawing
        </button>
      </div>
      <div className="container">
        {activeTab === 'voice'  && <VoiceTab />}
        {activeTab === 'spiral' && <SpiralTab />}
      </div>
    </div>
  );
}

export default App;