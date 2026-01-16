import React, { useState, useEffect, useCallback } from 'react';
import { 
  Upload, 
  Search, 
  LogIn, 
  LogOut, 
  CheckCircle, 
  AlertCircle, 
  Loader2,
  Star,
  Orbit,
  FileImage,
  Trash2,
  RefreshCw,
  Download,
  ExternalLink
} from 'lucide-react';

// Configuration - Update these with your deployed endpoints
const CONFIG = {
  API_BASE_URL: process.env.REACT_APP_API_URL || 'https://us-central1-YOUR_PROJECT.cloudfunctions.net',
  FIREBASE_CONFIG: {
    apiKey: process.env.REACT_APP_FIREBASE_API_KEY || "YOUR_API_KEY",
    authDomain: process.env.REACT_APP_FIREBASE_AUTH_DOMAIN || "YOUR_PROJECT.firebaseapp.com",
    projectId: process.env.REACT_APP_FIREBASE_PROJECT_ID || "YOUR_PROJECT",
    storageBucket: process.env.REACT_APP_FIREBASE_STORAGE_BUCKET || "YOUR_PROJECT.appspot.com",
    messagingSenderId: process.env.REACT_APP_FIREBASE_MESSAGING_SENDER_ID || "123456789",
    appId: process.env.REACT_APP_FIREBASE_APP_ID || "1:123456789:web:abc123"
  }
};

// Styles
const styles = {
  app: {
    minHeight: '100vh',
    background: 'linear-gradient(135deg, #0a0a1a 0%, #1a1a3a 50%, #0a1a2a 100%)',
    color: '#e0e0e0',
    fontFamily: "'Segoe UI', system-ui, sans-serif"
  },
  header: {
    background: 'rgba(0, 0, 0, 0.3)',
    backdropFilter: 'blur(10px)',
    borderBottom: '1px solid rgba(100, 200, 255, 0.1)',
    padding: '1rem 2rem',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center'
  },
  logo: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
    fontSize: '1.5rem',
    fontWeight: '600',
    color: '#4fd1c5'
  },
  main: {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '2rem'
  },
  card: {
    background: 'rgba(255, 255, 255, 0.05)',
    borderRadius: '16px',
    border: '1px solid rgba(100, 200, 255, 0.1)',
    padding: '1.5rem',
    marginBottom: '1.5rem'
  },
  button: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '0.5rem',
    padding: '0.75rem 1.5rem',
    borderRadius: '8px',
    border: 'none',
    cursor: 'pointer',
    fontWeight: '500',
    transition: 'all 0.2s'
  },
  primaryButton: {
    background: 'linear-gradient(135deg, #4fd1c5 0%, #38b2ac 100%)',
    color: '#0a0a1a'
  },
  secondaryButton: {
    background: 'rgba(255, 255, 255, 0.1)',
    color: '#e0e0e0',
    border: '1px solid rgba(100, 200, 255, 0.2)'
  },
  dropzone: {
    border: '2px dashed rgba(100, 200, 255, 0.3)',
    borderRadius: '12px',
    padding: '3rem',
    textAlign: 'center',
    cursor: 'pointer',
    transition: 'all 0.3s'
  },
  dropzoneActive: {
    borderColor: '#4fd1c5',
    background: 'rgba(79, 209, 197, 0.1)'
  },
  fileList: {
    marginTop: '1rem'
  },
  fileItem: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '0.75rem 1rem',
    background: 'rgba(0, 0, 0, 0.2)',
    borderRadius: '8px',
    marginBottom: '0.5rem'
  },
  progress: {
    height: '8px',
    background: 'rgba(255, 255, 255, 0.1)',
    borderRadius: '4px',
    overflow: 'hidden',
    marginTop: '1rem'
  },
  progressBar: {
    height: '100%',
    background: 'linear-gradient(90deg, #4fd1c5, #38b2ac)',
    transition: 'width 0.3s'
  },
  resultsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
    gap: '1rem',
    marginTop: '1rem'
  },
  resultCard: {
    background: 'rgba(0, 0, 0, 0.3)',
    borderRadius: '12px',
    padding: '1rem',
    border: '1px solid rgba(100, 200, 255, 0.1)'
  },
  badge: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '0.25rem',
    padding: '0.25rem 0.75rem',
    borderRadius: '999px',
    fontSize: '0.75rem',
    fontWeight: '600'
  },
  discoveryBadge: {
    background: 'rgba(246, 173, 85, 0.2)',
    color: '#f6ad55'
  },
  knownBadge: {
    background: 'rgba(72, 187, 120, 0.2)',
    color: '#48bb78'
  }
};

// Mock Firebase for demo (replace with real Firebase in production)
const useMockAuth = () => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Simulate auth check
    setTimeout(() => setLoading(false), 500);
  }, []);

  const signIn = () => {
    setUser({ 
      uid: 'demo-user', 
      email: 'demo@example.com',
      displayName: 'Demo User'
    });
  };

  const signOut = () => setUser(null);

  return { user, loading, signIn, signOut };
};

// File Upload Component
const FileUploader = ({ files, setFiles, disabled }) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDragIn = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragOut = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const droppedFiles = Array.from(e.dataTransfer.files);
    const validFiles = droppedFiles.filter(f => 
      /\.(fits?|fts|tiff?|jpe?g|xisf)$/i.test(f.name)
    );
    setFiles(prev => [...prev, ...validFiles]);
  }, [setFiles]);

  const handleFileSelect = (e) => {
    const selectedFiles = Array.from(e.target.files);
    setFiles(prev => [...prev, ...selectedFiles]);
  };

  const removeFile = (index) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  return (
    <div>
      <div
        style={{
          ...styles.dropzone,
          ...(isDragging ? styles.dropzoneActive : {}),
          opacity: disabled ? 0.5 : 1,
          pointerEvents: disabled ? 'none' : 'auto'
        }}
        onDragEnter={handleDragIn}
        onDragLeave={handleDragOut}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => document.getElementById('file-input').click()}
      >
        <input
          id="file-input"
          type="file"
          multiple
          accept=".fits,.fit,.fts,.tiff,.tif,.jpg,.jpeg,.xisf"
          style={{ display: 'none' }}
          onChange={handleFileSelect}
        />
        <Upload size={48} style={{ color: '#4fd1c5', marginBottom: '1rem' }} />
        <h3 style={{ margin: '0 0 0.5rem', color: '#fff' }}>
          Drop astronomical images here
        </h3>
        <p style={{ margin: 0, color: '#888' }}>
          Supports FITS, TIFF, JPEG, and XISF formats
        </p>
        <p style={{ margin: '0.5rem 0 0', fontSize: '0.875rem', color: '#666' }}>
          Minimum 2 images required for motion detection
        </p>
      </div>

      {files.length > 0 && (
        <div style={styles.fileList}>
          <h4 style={{ margin: '0 0 0.75rem', color: '#4fd1c5' }}>
            Selected Files ({files.length})
          </h4>
          {files.map((file, index) => (
            <div key={index} style={styles.fileItem}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <FileImage size={20} style={{ color: '#4fd1c5' }} />
                <span>{file.name}</span>
                <span style={{ color: '#666', fontSize: '0.875rem' }}>
                  ({(file.size / 1024 / 1024).toFixed(2)} MB)
                </span>
              </div>
              <button
                onClick={() => removeFile(index)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: '#e53e3e',
                  cursor: 'pointer',
                  padding: '0.25rem'
                }}
              >
                <Trash2 size={18} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Detection Options Component
const DetectionOptions = ({ options, setOptions }) => {
  return (
    <div style={{ marginTop: '1rem' }}>
      <h4 style={{ margin: '0 0 1rem', color: '#4fd1c5' }}>Detection Options</h4>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
        <div>
          <label style={{ display: 'block', marginBottom: '0.5rem', color: '#888' }}>
            Detection Threshold (σ)
          </label>
          <input
            type="number"
            value={options.threshold}
            onChange={(e) => setOptions({ ...options, threshold: parseFloat(e.target.value) })}
            min="1"
            max="10"
            step="0.5"
            style={{
              width: '100%',
              padding: '0.5rem',
              borderRadius: '6px',
              border: '1px solid rgba(100, 200, 255, 0.2)',
              background: 'rgba(0, 0, 0, 0.3)',
              color: '#fff'
            }}
          />
        </div>
        <div>
          <label style={{ display: 'block', marginBottom: '0.5rem', color: '#888' }}>
            Minimum Detections
          </label>
          <input
            type="number"
            value={options.minDetections}
            onChange={(e) => setOptions({ ...options, minDetections: parseInt(e.target.value) })}
            min="2"
            max="10"
            style={{
              width: '100%',
              padding: '0.5rem',
              borderRadius: '6px',
              border: '1px solid rgba(100, 200, 255, 0.2)',
              background: 'rgba(0, 0, 0, 0.3)',
              color: '#fff'
            }}
          />
        </div>
      </div>
    </div>
  );
};

// Job Status Component
const JobStatus = ({ job, onRefresh }) => {
  const statusColors = {
    pending: '#f6ad55',
    processing: '#4fd1c5',
    completed: '#48bb78',
    failed: '#fc8181'
  };

  return (
    <div style={styles.resultCard}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h4 style={{ margin: 0 }}>Job: {job.job_id?.slice(0, 8)}...</h4>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{
            ...styles.badge,
            background: `rgba(${statusColors[job.status] === '#48bb78' ? '72, 187, 120' : '79, 209, 197'}, 0.2)`,
            color: statusColors[job.status]
          }}>
            {job.status === 'processing' && <Loader2 size={12} style={{ animation: 'spin 1s linear infinite' }} />}
            {job.status}
          </span>
          <button
            onClick={onRefresh}
            style={{
              background: 'transparent',
              border: 'none',
              color: '#4fd1c5',
              cursor: 'pointer',
              padding: '0.25rem'
            }}
          >
            <RefreshCw size={16} />
          </button>
        </div>
      </div>

      {job.progress !== undefined && (
        <div style={styles.progress}>
          <div style={{ ...styles.progressBar, width: `${job.progress}%` }} />
        </div>
      )}

      {job.status === 'completed' && job.result && (
        <div style={{ marginTop: '1rem' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.5rem', textAlign: 'center' }}>
            <div>
              <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#4fd1c5' }}>
                {job.result.moving_objects_count || 0}
              </div>
              <div style={{ fontSize: '0.75rem', color: '#888' }}>Moving Objects</div>
            </div>
            <div>
              <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#48bb78' }}>
                {job.result.known_objects_count || 0}
              </div>
              <div style={{ fontSize: '0.75rem', color: '#888' }}>Known</div>
            </div>
            <div>
              <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#f6ad55' }}>
                {job.result.potential_discoveries_count || 0}
              </div>
              <div style={{ fontSize: '0.75rem', color: '#888' }}>Discoveries</div>
            </div>
          </div>
        </div>
      )}

      {job.status === 'failed' && job.error && (
        <div style={{ marginTop: '1rem', color: '#fc8181' }}>
          <AlertCircle size={16} style={{ marginRight: '0.5rem' }} />
          {job.error}
        </div>
      )}
    </div>
  );
};

// Results Display Component
const ResultsDisplay = ({ result }) => {
  if (!result) return null;

  return (
    <div style={styles.card}>
      <h3 style={{ margin: '0 0 1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <CheckCircle style={{ color: '#48bb78' }} />
        Detection Results
      </h3>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
        <div style={{ textAlign: 'center', padding: '1rem', background: 'rgba(0,0,0,0.2)', borderRadius: '8px' }}>
          <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#4fd1c5' }}>
            {result.processing_time?.toFixed(1)}s
          </div>
          <div style={{ color: '#888', fontSize: '0.875rem' }}>Processing Time</div>
        </div>
        <div style={{ textAlign: 'center', padding: '1rem', background: 'rgba(0,0,0,0.2)', borderRadius: '8px' }}>
          <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#4fd1c5' }}>
            {result.moving_objects_count || 0}
          </div>
          <div style={{ color: '#888', fontSize: '0.875rem' }}>Moving Objects</div>
        </div>
        <div style={{ textAlign: 'center', padding: '1rem', background: 'rgba(0,0,0,0.2)', borderRadius: '8px' }}>
          <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#48bb78' }}>
            {result.known_objects_count || 0}
          </div>
          <div style={{ color: '#888', fontSize: '0.875rem' }}>Known Objects</div>
        </div>
        <div style={{ textAlign: 'center', padding: '1rem', background: 'rgba(0,0,0,0.2)', borderRadius: '8px' }}>
          <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#f6ad55' }}>
            {result.potential_discoveries_count || 0}
          </div>
          <div style={{ color: '#888', fontSize: '0.875rem' }}>Potential Discoveries</div>
        </div>
      </div>

      {result.moving_objects && result.moving_objects.length > 0 && (
        <div>
          <h4 style={{ margin: '0 0 1rem', color: '#4fd1c5' }}>Detected Objects</h4>
          <div style={styles.resultsGrid}>
            {result.moving_objects.map((obj, index) => (
              <div key={index} style={styles.resultCard}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                  <span style={{ fontWeight: '600' }}>{obj.id}</span>
                  <span style={{
                    ...styles.badge,
                    ...(obj.is_known ? styles.knownBadge : styles.discoveryBadge)
                  }}>
                    {obj.is_known ? (
                      <><Star size={12} /> Known</>
                    ) : (
                      <><Orbit size={12} /> Discovery?</>
                    )}
                  </span>
                </div>
                {obj.matched_name && (
                  <div style={{ marginBottom: '0.5rem', color: '#48bb78' }}>
                    {obj.matched_name}
                  </div>
                )}
                <div style={{ fontSize: '0.875rem', color: '#888' }}>
                  <div>Velocity: {obj.velocity_arcsec_per_hour?.toFixed(2)}"/hr</div>
                  <div>Position Angle: {obj.position_angle?.toFixed(1)}°</div>
                  <div>Confidence: {(obj.confidence * 100).toFixed(0)}%</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {result.report_url && (
        <div style={{ marginTop: '1.5rem', textAlign: 'center' }}>
          <a
            href={result.report_url}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              ...styles.button,
              ...styles.secondaryButton,
              textDecoration: 'none'
            }}
          >
            <Download size={18} />
            Download Full Report
            <ExternalLink size={14} />
          </a>
        </div>
      )}
    </div>
  );
};

// Main App Component
function App() {
  const { user, loading: authLoading, signIn, signOut } = useMockAuth();
  const [files, setFiles] = useState([]);
  const [options, setOptions] = useState({ threshold: 3.0, minDetections: 3 });
  const [jobs, setJobs] = useState([]);
  const [currentResult, setCurrentResult] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState(null);

  // Demo mode processing
  const runDetection = async () => {
    if (files.length < 2) {
      setError('Please select at least 2 images for motion detection');
      return;
    }

    setProcessing(true);
    setError(null);

    // Simulate processing for demo
    const jobId = `demo-${Date.now()}`;
    const newJob = {
      job_id: jobId,
      status: 'processing',
      progress: 0,
      created_at: new Date().toISOString()
    };
    setJobs(prev => [newJob, ...prev]);

    // Simulate progress
    for (let i = 0; i <= 100; i += 10) {
      await new Promise(resolve => setTimeout(resolve, 300));
      setJobs(prev => prev.map(j => 
        j.job_id === jobId ? { ...j, progress: i } : j
      ));
    }

    // Generate demo result
    const demoResult = {
      processing_time: 12.5,
      moving_objects_count: 3,
      known_objects_count: 2,
      potential_discoveries_count: 1,
      moving_objects: [
        {
          id: 'obj_0001',
          velocity_arcsec_per_hour: 45.2,
          position_angle: 127.3,
          confidence: 0.95,
          is_known: true,
          matched_name: '(433) Eros'
        },
        {
          id: 'obj_0002',
          velocity_arcsec_per_hour: 23.8,
          position_angle: 84.1,
          confidence: 0.88,
          is_known: true,
          matched_name: '(1) Ceres'
        },
        {
          id: 'obj_0003',
          velocity_arcsec_per_hour: 67.4,
          position_angle: 256.9,
          confidence: 0.72,
          is_known: false,
          matched_name: null
        }
      ]
    };

    setJobs(prev => prev.map(j => 
      j.job_id === jobId ? { ...j, status: 'completed', progress: 100, result: demoResult } : j
    ));
    setCurrentResult(demoResult);
    setProcessing(false);
    setFiles([]);
  };

  if (authLoading) {
    return (
      <div style={{ ...styles.app, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Loader2 size={48} style={{ color: '#4fd1c5', animation: 'spin 1s linear infinite' }} />
      </div>
    );
  }

  return (
    <div style={styles.app}>
      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>

      <header style={styles.header}>
        <div style={styles.logo}>
          <Orbit size={32} />
          <span>Asteroid Detection Platform</span>
        </div>
        <div>
          {user ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <span style={{ color: '#888' }}>{user.email}</span>
              <button
                onClick={signOut}
                style={{ ...styles.button, ...styles.secondaryButton }}
              >
                <LogOut size={18} />
                Sign Out
              </button>
            </div>
          ) : (
            <button
              onClick={signIn}
              style={{ ...styles.button, ...styles.primaryButton }}
            >
              <LogIn size={18} />
              Sign In with Google
            </button>
          )}
        </div>
      </header>

      <main style={styles.main}>
        {!user ? (
          <div style={{ textAlign: 'center', padding: '4rem 2rem' }}>
            <Orbit size={80} style={{ color: '#4fd1c5', marginBottom: '2rem' }} />
            <h1 style={{ marginBottom: '1rem' }}>Welcome to Asteroid Detection Platform</h1>
            <p style={{ color: '#888', maxWidth: '600px', margin: '0 auto 2rem' }}>
              Upload astronomical images to detect moving objects like asteroids and comets.
              Our system uses advanced motion detection algorithms and queries NASA databases
              to identify known objects or flag potential new discoveries.
            </p>
            <button
              onClick={signIn}
              style={{ ...styles.button, ...styles.primaryButton, fontSize: '1.125rem', padding: '1rem 2rem' }}
            >
              <LogIn size={20} />
              Get Started
            </button>
          </div>
        ) : (
          <>
            <div style={styles.card}>
              <h2 style={{ margin: '0 0 1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Search size={24} style={{ color: '#4fd1c5' }} />
                New Detection
              </h2>

              <FileUploader files={files} setFiles={setFiles} disabled={processing} />
              <DetectionOptions options={options} setOptions={setOptions} />

              {error && (
                <div style={{ 
                  marginTop: '1rem', 
                  padding: '0.75rem 1rem',
                  background: 'rgba(252, 129, 129, 0.1)',
                  border: '1px solid rgba(252, 129, 129, 0.3)',
                  borderRadius: '8px',
                  color: '#fc8181',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}>
                  <AlertCircle size={18} />
                  {error}
                </div>
              )}

              <div style={{ marginTop: '1.5rem', textAlign: 'right' }}>
                <button
                  onClick={runDetection}
                  disabled={processing || files.length < 2}
                  style={{
                    ...styles.button,
                    ...styles.primaryButton,
                    opacity: (processing || files.length < 2) ? 0.5 : 1,
                    cursor: (processing || files.length < 2) ? 'not-allowed' : 'pointer'
                  }}
                >
                  {processing ? (
                    <><Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} /> Processing...</>
                  ) : (
                    <><Search size={18} /> Start Detection</>
                  )}
                </button>
              </div>
            </div>

            {currentResult && <ResultsDisplay result={currentResult} />}

            {jobs.length > 0 && (
              <div style={styles.card}>
                <h3 style={{ margin: '0 0 1rem' }}>Recent Jobs</h3>
                <div style={styles.resultsGrid}>
                  {jobs.slice(0, 6).map((job, index) => (
                    <JobStatus
                      key={job.job_id || index}
                      job={job}
                      onRefresh={() => {}}
                    />
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </main>

      <footer style={{ 
        textAlign: 'center', 
        padding: '2rem', 
        color: '#666',
        borderTop: '1px solid rgba(100, 200, 255, 0.1)'
      }}>
        <p>Asteroid Detection Platform v1.0.0</p>
        <p style={{ fontSize: '0.875rem' }}>
          Built with ♥ for the astronomical community
        </p>
      </footer>
    </div>
  );
}

export default App;
