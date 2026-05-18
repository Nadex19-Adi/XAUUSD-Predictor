import { useState, useEffect, useRef } from 'react';
import { 
  Activity, 
  Cpu, 
  Layers, 
  RefreshCw, 
  Sliders, 
  Database, 
  ShieldAlert, 
  Globe, 
  Clock, 
  Sparkles, 
  Play,
  AlertTriangle
} from 'lucide-react';

// API configuration
const API_BASE_URL = "http://127.0.0.1:8000";
const API_KEY = "gold-standard-2026";

interface SimilarPattern {
  document: string;
  timestamp: string;
  move: number;
  similarity: number;
  regime: string;
  raw_cosine: number;
}

interface PredictionResponse {
  direction: string;
  confidence: number;
  signal_strength: string;
  should_trade: boolean;
  tp: number;
  sl: number;
  rag_insight: {
    sim_win_rate: number;
    sim_avg_return: number;
    regime_used: string;
  };
  similar_patterns: SimilarPattern[];
}

interface SystemHealth {
  status: string;
  system: {
    ram_used_gb: number;
    ram_total_gb: number;
    cpu_percent: number;
  };
  models: {
    predictor_loaded: boolean;
    rag_loaded: boolean;
    active_shards: string[];
    version: string;
  };
  timestamp: string;
}

// Indicator Presets
const PRESETS = {
  bullish: {
    rsi: 38.5,
    macd: 0.24,
    macd_signal: 0.12,
    macd_hist: 0.12,
    atr: 1.85,
    bb_width: 5.2,
    returns: 0.0008,
    close: 2045.20,
    atr_percentile: 0.35,
    body_ratio: 0.75,
    lower_wick_ratio: 0.65,
    upper_wick_ratio: 0.10,
    momentum_10: 0.0015,
    returns_roll_mean_10: 0.0002,
    returns_roll_std_10: 0.0005,
    ema_cross: 1.0,
    rsi_roll_mean_10: 42.5,
    rsi_roll_std_10: 3.2,
    bb_squeeze: 0,
    macd_hist_roll_mean_10: 0.06,
    returns_lag1: 0.0003,
    trend_alignment: 0.60,
    macro: "Inflation print cooling down, gold demand soaring"
  },
  bearish: {
    rsi: 72.1,
    macd: -0.32,
    macd_signal: -0.16,
    macd_hist: -0.16,
    atr: 2.10,
    bb_width: 6.8,
    returns: -0.0012,
    close: 2038.50,
    atr_percentile: 0.65,
    body_ratio: 0.80,
    lower_wick_ratio: 0.12,
    upper_wick_ratio: 0.70,
    momentum_10: -0.0022,
    returns_roll_mean_10: -0.0004,
    returns_roll_std_10: 0.0007,
    ema_cross: -1.0,
    rsi_roll_mean_10: 68.2,
    rsi_roll_std_10: 4.1,
    bb_squeeze: 0,
    macd_hist_roll_mean_10: -0.08,
    returns_lag1: -0.0005,
    trend_alignment: -0.75,
    macro: "Fed hawkish statement, high yields strengthening USD"
  },
  flat: {
    rsi: 50.5,
    macd: 0.02,
    macd_signal: 0.01,
    macd_hist: 0.01,
    atr: 0.95,
    bb_width: 2.1,
    returns: 0.0001,
    close: 2041.80,
    atr_percentile: 0.15,
    body_ratio: 0.30,
    lower_wick_ratio: 0.35,
    upper_wick_ratio: 0.35,
    momentum_10: 0.0001,
    returns_roll_mean_10: 0.0000,
    returns_roll_std_10: 0.0002,
    ema_cross: 0.0,
    rsi_roll_mean_10: 49.8,
    rsi_roll_std_10: 1.1,
    bb_squeeze: 1,
    macd_hist_roll_mean_10: 0.00,
    returns_lag1: 0.0001,
    trend_alignment: 0.02,
    macro: "no major news"
  }
};

export default function App() {
  const [activeTab, setActiveTab] = useState<'chart' | 'predict'>('chart');
  const [isOnline, setIsOnline] = useState<boolean>(false);
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  
  // Indicators State
  const [indicators, setIndicators] = useState(PRESETS.bullish);
  const [macroNews, setMacroNews] = useState<string>("Inflation print cooling down, gold demand soaring");
  const [timestamp, setTimestamp] = useState<string>("");
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.55);
  
  // Prediction Response State
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [expandedPattern, setExpandedPattern] = useState<number | null>(null);
  
  // Chart Initialized Ref
  const tvWidgetRef = useRef<boolean>(false);

  // Auto-fill Timestamp with client time
  useEffect(() => {
    const now = new Date();
    const formatted = now.toISOString().replace('T', ' ').substring(0, 19);
    setTimestamp(formatted);
  }, []);

  // Fetch API Health on startup and polling
  const fetchHealth = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/health`);
      if (res.ok) {
        const data = await res.json();
        setSystemHealth(data);
        setIsOnline(true);
      } else {
        setIsOnline(false);
      }
    } catch {
      setIsOnline(false);
    }
  };

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  // Apply Preset Values
  const applyPreset = (presetName: 'bullish' | 'bearish' | 'flat') => {
    const selected = PRESETS[presetName];
    setIndicators(selected);
    setMacroNews(selected.macro);
    // Flash dynamic preview
    const btn = document.getElementById(`btn-${presetName}`);
    if (btn) {
      btn.style.borderColor = 'var(--gold-accent)';
      setTimeout(() => btn.style.borderColor = 'var(--border-color)', 500);
    }
  };

  // Run Deep Prediction Request
  const runPrediction = async () => {
    setIsPredicting(true);
    setErrorMsg(null);
    setPrediction(null);
    setExpandedPattern(null);

    // Filter NaN / Inf
    const cleanIndicators: Record<string, number> = {};
    Object.entries(indicators).forEach(([k, v]) => {
      if (typeof v === 'number') {
        cleanIndicators[k] = isNaN(v) || !isFinite(v) ? 0.0 : v;
      }
    });

    const payload = {
      current_indicators: cleanIndicators,
      macro_snippet: macroNews,
      timestamp: timestamp || new Date().toISOString().replace('T', ' ').substring(0, 19),
      confidence_threshold: confidenceThreshold
    };

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': API_KEY
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Server Error ${response.status}`);
      }

      const data = await response.json();
      setPrediction(data);
    } catch (e: any) {
      setErrorMsg(e.message || "Could not establish connection to the FastAPI server.");
    } finally {
      setIsPredicting(false);
    }
  };

  // Render TradingView script widget once
  useEffect(() => {
    if (activeTab === 'chart' && !tvWidgetRef.current) {
      // Small timeout to guarantee container is fully painted
      setTimeout(() => {
        const container = document.getElementById('tv-chart-container');
        if (container) {
          container.innerHTML = '';
          const script = document.createElement('script');
          script.src = 'https://s3.tradingview.com/tv.js';
          script.type = 'text/javascript';
          script.onload = () => {
            if ((window as any).TradingView) {
              new (window as any).TradingView.widget({
                "autosize": true,
                "symbol": "OANDA:XAUUSD",
                "interval": "5",
                "timezone": "Etc/UTC",
                "theme": "dark",
                "style": "1",
                "locale": "en",
                "toolbar_bg": "#1e222d",
                "enable_publishing": false,
                "hide_side_toolbar": false,
                "allow_symbol_change": true,
                "studies": ["RSI@tv-basicstudies", "MASimple@tv-basicstudies"],
                "container_id": "tv-chart-container"
              });
            }
          };
          document.head.appendChild(script);
          tvWidgetRef.current = true;
        }
      }, 300);
    }
    
    if (activeTab !== 'chart') {
      tvWidgetRef.current = false;
    }
  }, [activeTab]);

  return (
    <div className="app-container">
      {/* Header section */}
      <header>
        <div className="brand">
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <span className="brand-logo">🏆 XAUUSD PRO PREDICTOR</span>
            <span className="brand-tagline">Hybrid RAG + XGBoost + AI Vision Dashboard</span>
          </div>
        </div>

        {/* Live Status indicator */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div className="tabs-container">
            <button 
              className={`tab-btn ${activeTab === 'chart' ? 'active' : ''}`}
              onClick={() => setActiveTab('chart')}
            >
              <Globe size={16} /> Live TradingView
            </button>
            <button 
              className={`tab-btn ${activeTab === 'predict' ? 'active' : ''}`}
              onClick={() => setActiveTab('predict')}
            >
              <Cpu size={16} /> Deep Predictor Engine
            </button>
          </div>

          <div className={`status-badge ${isOnline ? '' : 'offline'}`}>
            <span className={`status-dot ${isOnline ? '' : 'offline'}`}></span>
            {isOnline ? 'FASTAPI: ONLINE' : 'FASTAPI: OFFLINE'}
          </div>
        </div>
      </header>

      {/* Main Tab Views */}
      {activeTab === 'chart' ? (
        <div className="glass-card" style={{ padding: '8px', display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '16px', alignItems: 'center' }}>
            <span style={{ fontWeight: 700, fontSize: '16px', color: 'var(--gold-accent)', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Activity size={18} /> Real-Time Live XAUUSD Chart (Zero Delay)
            </span>
            <div style={{ display: 'flex', gap: '10px', fontSize: '12px', color: 'var(--text-secondary)' }}>
              <span>Broker Feed: OANDA</span>
              <span>•</span>
              <span>Timeframe: 5m</span>
            </div>
          </div>
          <div id="tv-chart-container" style={{ flex: 1, minHeight: 0, borderRadius: '12px', overflow: 'hidden' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-secondary)' }}>
              Loading TradingView Interactive Interface...
            </div>
          </div>
        </div>
      ) : (
        <div className="dashboard-grid" style={{ flex: 1, minHeight: 0 }}>
          
          {/* LEFT COLUMN: Controls & Form */}
          <div className="glass-card sidebar-panel">
            <div>
              <h3 className="sidebar-title"><Sliders size={18} /> Input Parameters</h3>
              <p style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                Populate values instantly using the presets below, then click Deep Predict.
              </p>
            </div>

            {/* Presets selector */}
            <div>
              <span className="form-label" style={{ fontWeight: 700 }}>Quick Presets</span>
              <div className="preset-grid">
                <button id="btn-bullish" className="preset-btn" onClick={() => applyPreset('bullish')}>📈 BULLISH</button>
                <button id="btn-bearish" className="preset-btn" onClick={() => applyPreset('bearish')}>📉 BEARISH</button>
                <button id="btn-flat" className="preset-btn" onClick={() => applyPreset('flat')}>⚖️ SIDEWAYS</button>
              </div>
            </div>

            {/* Configurable Sliders & Inputs */}
            <div style={{ flex: 1, overflowY: 'auto', paddingRight: '6px' }}>
              <span className="form-section-title">Core Technicals</span>
              
              <div className="form-group">
                <label className="form-label">
                  <span>RSI (14)</span>
                  <span style={{ color: 'var(--gold-accent)' }}>{indicators.rsi.toFixed(1)}</span>
                </label>
                <input 
                  type="range" min="10" max="90" step="0.5" 
                  value={indicators.rsi}
                  onChange={(e) => setIndicators({...indicators, rsi: parseFloat(e.target.value)})}
                  style={{ accentColor: 'var(--gold-accent)' }}
                />
              </div>

              <div className="form-group">
                <label className="form-label">
                  <span>MACD Hist</span>
                  <span style={{ color: 'var(--gold-accent)' }}>{indicators.macd_hist.toFixed(3)}</span>
                </label>
                <input 
                  type="number" step="0.01" 
                  className="form-input"
                  value={indicators.macd_hist}
                  onChange={(e) => setIndicators({...indicators, macd_hist: parseFloat(e.target.value) || 0})}
                />
              </div>

              <div className="form-group">
                <label className="form-label">
                  <span>ATR (Volatility Level)</span>
                  <span style={{ color: 'var(--gold-accent)' }}>{indicators.atr.toFixed(2)}</span>
                </label>
                <input 
                  type="number" step="0.05" 
                  className="form-input"
                  value={indicators.atr}
                  onChange={(e) => setIndicators({...indicators, atr: parseFloat(e.target.value) || 0})}
                />
              </div>

              <div className="form-group">
                <label className="form-label">
                  <span>Bollinger Band Width</span>
                  <span style={{ color: 'var(--gold-accent)' }}>{indicators.bb_width.toFixed(2)}</span>
                </label>
                <input 
                  type="number" step="0.1" 
                  className="form-input"
                  value={indicators.bb_width}
                  onChange={(e) => setIndicators({...indicators, bb_width: parseFloat(e.target.value) || 0})}
                />
              </div>

              <div className="form-group">
                <label className="form-label">
                  <span>Current Spot Price (Close)</span>
                  <span style={{ color: 'var(--gold-accent)' }}>${indicators.close.toFixed(2)}</span>
                </label>
                <input 
                  type="number" step="0.5" 
                  className="form-input"
                  value={indicators.close}
                  onChange={(e) => setIndicators({...indicators, close: parseFloat(e.target.value) || 0})}
                />
              </div>

              <span className="form-section-title">Candle Patterns & Trend</span>

              <div className="form-group">
                <label className="form-label">
                  <span>Trend Alignment (-1.0 to 1.0)</span>
                  <span style={{ color: 'var(--gold-accent)' }}>{indicators.trend_alignment.toFixed(2)}</span>
                </label>
                <input 
                  type="range" min="-1" max="1" step="0.05"
                  value={indicators.trend_alignment}
                  onChange={(e) => setIndicators({...indicators, trend_alignment: parseFloat(e.target.value)})}
                  style={{ accentColor: 'var(--gold-accent)' }}
                />
              </div>

              <div className="form-group">
                <label className="form-label">
                  <span>Body Ratio</span>
                  <span style={{ color: 'var(--gold-accent)' }}>{indicators.body_ratio.toFixed(2)}</span>
                </label>
                <input 
                  type="range" min="0.05" max="0.95" step="0.05"
                  value={indicators.body_ratio}
                  onChange={(e) => setIndicators({...indicators, body_ratio: parseFloat(e.target.value)})}
                  style={{ accentColor: 'var(--gold-accent)' }}
                />
              </div>

              <span className="form-section-title">Context & Config</span>

              <div className="form-group">
                <label className="form-label">Macro News Snippet</label>
                <input 
                  type="text" 
                  className="form-input" 
                  value={macroNews}
                  onChange={(e) => setMacroNews(e.target.value)}
                />
              </div>

              <div className="form-group">
                <label className="form-label">Confidence Threshold</label>
                <select 
                  className="form-input"
                  value={confidenceThreshold}
                  onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                >
                  <option value="0.55">Moderate (55%)</option>
                  <option value="0.60">Strong (60%)</option>
                  <option value="0.70">Institutional (70%)</option>
                </select>
              </div>
            </div>

            <button 
              className="run-btn" 
              onClick={runPrediction}
              disabled={isPredicting || !isOnline}
            >
              {isPredicting ? (
                <>
                  <RefreshCw className="animate-spin" size={18} /> Analyzing 13Y History...
                </>
              ) : (
                <>
                  <Play size={16} /> RUN DEEP PREDICTION
                </>
              )}
            </button>
          </div>

          {/* RIGHT COLUMN: Results Screen */}
          <div className="output-layout">
            
            {/* API Health & Machine info */}
            {systemHealth && (
              <div className="glass-card system-details-grid" style={{ padding: '12px 18px' }}>
                <div className="system-detail-item">
                  <span className="system-detail-label"><Database size={12} /> Master Memory:</span>
                  <span className="system-detail-val">Active (2.4M Twins)</span>
                </div>
                <div className="system-detail-item">
                  <span className="system-detail-label"><Cpu size={12} /> Core Memory:</span>
                  <span className="system-detail-val">{systemHealth.system.ram_used_gb} GB / {systemHealth.system.ram_total_gb} GB</span>
                </div>
                <div className="system-detail-item">
                  <span className="system-detail-label"><Clock size={12} /> Server Engine:</span>
                  <span className="system-detail-val">v{systemHealth.models.version}</span>
                </div>
                <div className="system-detail-item">
                  <span className="system-detail-label"><ShieldAlert size={12} /> Active Shards:</span>
                  <span className="system-detail-val">{systemHealth.models.active_shards.length} Shards</span>
                </div>
              </div>
            )}

            {/* Error handling */}
            {errorMsg && (
              <div className="glass-card" style={{ borderColor: 'rgba(239,68,68,0.4)', background: 'rgba(239,68,68,0.05)', display: 'flex', gap: '12px', alignItems: 'center' }}>
                <AlertTriangle color="var(--crimson-down)" size={24} />
                <div>
                  <h4 style={{ color: 'var(--crimson-down)', margin: '0 0 4px 0', fontSize: '14px', fontWeight: 700 }}>Connection Error</h4>
                  <p style={{ fontSize: '12px', color: 'var(--text-secondary)', margin: 0 }}>{errorMsg}</p>
                </div>
              </div>
            )}

            {/* Waiting/Empty state */}
            {!prediction && !errorMsg && !isPredicting && (
              <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '60px 24px', textAlign: 'center', gap: '16px' }}>
                <Sparkles size={48} color="var(--gold-accent)" style={{ opacity: 0.7 }} />
                <div>
                  <h3 style={{ fontSize: '18px', fontWeight: 700, margin: '0 0 6px 0' }}>Deep Prediction Pipeline Ready</h3>
                  <p style={{ fontSize: '13px', color: 'var(--text-secondary)', maxWidth: '450px' }}>
                    Configure the technical indicators on the left side or load a preset, then launch the hybrid XGBoost + RAG memory scanner!
                  </p>
                </div>
              </div>
            )}

            {/* Predicting State Loader */}
            {isPredicting && (
              <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '60px 24px', textAlign: 'center', gap: '20px' }}>
                <RefreshCw size={48} className="animate-spin" color="var(--gold-accent)" />
                <div>
                  <h3 style={{ fontSize: '18px', fontWeight: 700, margin: '0 0 8px 0' }}>Scanning Market Memories</h3>
                  <p style={{ fontSize: '13px', color: 'var(--text-secondary)', maxWidth: '400px' }}>
                    Scanning sharded vector collections (legacy, mid, recent)... Matching candle geometries & regime matching...
                  </p>
                </div>
              </div>
            )}

            {/* Prediction Output Results */}
            {prediction && (
              <>
                {/* 4 core metric badges */}
                <div className="metrics-row">
                  <div className="metric-panel">
                    <span className="metric-label">Predicted Move</span>
                    <span className={`metric-value ${
                      prediction.direction === 'UP' ? 'direction-up' : 
                      prediction.direction === 'DOWN' ? 'direction-down' : 'direction-neutral'
                    }`}>
                      {prediction.direction}
                    </span>
                  </div>

                  <div className="metric-panel">
                    <span className="metric-label">Probability</span>
                    <span className="metric-value" style={{ color: 'var(--gold-accent)' }}>
                      {(prediction.confidence * 100).toFixed(1)}%
                    </span>
                  </div>

                  <div className="metric-panel">
                    <span className="metric-label">Time Horizon</span>
                    <span className="metric-value" style={{ color: '#fff', fontSize: '16px', fontWeight: 800 }}>
                      15 MINS <span style={{ fontSize: '11px', color: 'var(--text-secondary)', display: 'block', fontWeight: 500 }}>3 Candles (5m)</span>
                    </span>
                  </div>

                  <div className="metric-panel">
                    <span className="metric-label">Target Zone</span>
                    <div className="target-zone-container">
                      <span className="target-tp">TP: ${prediction.tp.toFixed(2)}</span>
                      <span className="target-sl">SL: ${prediction.sl.toFixed(2)}</span>
                    </div>
                  </div>
                </div>

                {/* RAG Memory Twins section */}
                <div className="glass-card">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '12px' }}>
                    <h3 style={{ fontSize: '16px', fontWeight: 700, margin: 0, display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--gold-accent)' }}>
                      <Layers size={18} /> Market Memory: Top 3 Similar Twins
                    </h3>
                    <div style={{ fontSize: '12px', color: 'var(--text-secondary)', display: 'flex', gap: '12px' }}>
                      <span>Scan Mode: {prediction.rag_insight.regime_used}</span>
                      <span>•</span>
                      <span>Memory Win Rate: {(prediction.rag_insight.sim_win_rate * 100).toFixed(0)}%</span>
                    </div>
                  </div>

                  <div className="twins-grid">
                    {prediction.similar_patterns.slice(0, 3).map((pattern, idx) => (
                      <div key={idx} className="twin-card">
                        <div className="twin-header">
                          <span className="twin-title">Analog {idx + 1}</span>
                          <span className="twin-similarity">Match: {(pattern.similarity * 100).toFixed(0)}%</span>
                        </div>
                        <div>
                          <div className="twin-outcome" style={{ 
                            color: pattern.move === 1 ? 'var(--emerald-up)' : 'var(--crimson-down)' 
                          }}>
                            Outcome: {pattern.move === 1 ? '📈 BULLISH' : '📉 BEARISH'}
                          </div>
                          <div className="twin-date">Date: {pattern.timestamp}</div>
                        </div>

                        <button 
                          className="twin-expander"
                          onClick={() => setExpandedPattern(expandedPattern === idx ? null : idx)}
                        >
                          {expandedPattern === idx ? 'Hide technical twin' : 'Show technical twin'}
                        </button>

                        {expandedPattern === idx && (
                          <div className="twin-details">
                            {pattern.document}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}

            <div style={{ textAlign: 'center', fontSize: '11px', color: 'var(--text-secondary)', opacity: 0.6 }}>
              Disclaimer: This AI system is built for quantitative research and educational simulation only. Not financial trading advice.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
