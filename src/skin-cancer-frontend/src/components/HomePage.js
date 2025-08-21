import React, { useState } from 'react';
import { Upload, Camera, AlertTriangle, Info, Activity, CheckCircle, Stethoscope } from 'lucide-react';
import './HomePage.css';

const HomePage = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000'

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (file) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setPrediction(null);
      setError(null);
    } else {
      setError('Please select a valid image file');
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const analyzeSkin = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Analysis failed. Please try again.');
      }

      const result = await response.json();
      setPrediction(result);
    } catch (err) {
      console.error('API Error:', err);
      setError(err.message || 'Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const resetAnalysis = () => {
    setSelectedFile(null);
    setPrediction(null);
    setError(null);
  };

  return (
    <div className="home-page">
      <div className="medical-disclaimer">
        <div className="disclaimer-content">
          <AlertTriangle className="disclaimer-icon" />
          <div className="disclaimer-text">
            <p className="disclaimer-title">Professional Medical Tool</p>
            <p>This tool is designed exclusively for professional dermatoscopic images captured in clinical settings. For use by qualified medical professionals only. Always consult comprehensive clinical examination and patient history.</p>
          </div>
        </div>
      </div>

      <div className="main-grid">
        <div className="upload-section">
          <div className="section-header">
            <h2>Upload Dermatoscopic Image</h2>
            <p>Upload a professional dermatoscopic image for clinical AI analysis</p>
          </div>

          <div
            className={`upload-zone ${dragActive ? 'drag-active' : ''} ${selectedFile ? 'has-file' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            {selectedFile ? (
              <div className="file-preview">
                <img
                  src={URL.createObjectURL(selectedFile)}
                  alt="Selected dermatoscopic image"
                  className="preview-image"
                />
                <div className="file-info">
                  <p className="file-name">{selectedFile.name}</p>
                  <p>{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
                </div>
                <button
                  onClick={resetAnalysis}
                  className="change-file-btn"
                >
                  Choose different image
                </button>
              </div>
            ) : (
              <div>
                <Upload className="upload-icon" />
                <div className="upload-text">
                  <p className="upload-title">Drop dermatoscopic image here</p>
                  <p className="upload-subtitle">or click to browse professional medical images</p>
                </div>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="file-input"
                />
              </div>
            )}
          </div>

          {selectedFile && (
            <button
              onClick={analyzeSkin}
              disabled={loading}
              className="analyze-button"
            >
              {loading ? (
                <>
                  <div className="loading-spinner"></div>
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <Camera className="button-icon" />
                  <span>Analyze Dermatoscopic Image</span>
                </>
              )}
            </button>
          )}
        </div>

        <div className="results-section">
          <div className="section-header">
            <h3>Clinical Analysis Results</h3>
            <p>AI-powered dermatoscopic image analysis results</p>
          </div>

          {error && (
            <div className="error-message">
              <div className="error-header">
                <AlertTriangle className="error-icon" />
                <p className="error-title">Analysis Notice</p>
              </div>
              <p className="error-text">{error}</p>
            </div>
          )}

          {prediction && (
            <div className="results-card">
              <div className="result-header">
                <h4 className="result-title">Clinical Analysis Result</h4>
                <div className={`result-badge ${prediction.prediction === 'Melanoma' ? 'melanoma' : 'non-melanoma'}`}>
                  {prediction.prediction}
                </div>
              </div>

              {prediction.demo_mode && (
                <div className="demo-notice">
                  <Info className="demo-icon" />
                  <span>Demo Mode - Using sample prediction</span>
                </div>
              )}

              {prediction.demo_mode && (
                <div className="demo-notice">
                  <Info className="demo-icon" />
                  <span>Demo Mode - Using sample prediction</span>
                </div>
              )}

              <div className="confidence-section">
                <div className="confidence-header">
                  <span className="confidence-label">Model Confidence</span>
                  <span className="confidence-value">
                    {(prediction.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="confidence-bar">
                  <div 
                    className="confidence-fill"
                    style={{ width: `${prediction.confidence * 100}%` }}
                  ></div>
                </div>
              </div>

              {prediction.probabilities && (
                <div className="probabilities-section">
                  <h5>Detailed Classification Probabilities</h5>
                  {Object.entries(prediction.probabilities).map(([key, value]) => (
                    <div key={key} className="probability-item">
                      <span className="probability-label">{key}</span>
                      <span className="probability-value">{(value * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              )}

              <div className="result-footer">
                <div className="result-timestamp">
                  <Info className="timestamp-icon" />
                  <p className="timestamp-text">
                    Clinical analysis completed: {new Date(prediction.timestamp).toLocaleString()}
                  </p>
                </div>
              </div>
            </div>
          )}

          {!prediction && !error && !loading && (
            <div className="results-placeholder">
              <Activity className="placeholder-icon" />
              <p className="placeholder-text">Upload a dermatoscopic image to view clinical analysis</p>
            </div>
          )}
        </div>
      </div>

      <div className="image-requirements">
        <h4>Professional Dermatoscopic Image Requirements</h4>
        <div className="requirements-grid">
          <div className="requirement-card">
            <CheckCircle className="requirement-icon" />
            <div className="requirement-content">
              <h5>Clinical Quality Standards</h5>
              <ul>
                <li>Professional dermatoscopic images only</li>
                <li>Minimum resolution: 600x600px recommended</li>
                <li>Clear focus and proper lighting</li>
                <li>Standardized medical photography protocols</li>
              </ul>
            </div>
          </div>
          
          <div className="requirement-card">
            <AlertTriangle className="requirement-icon warning" />
            <div className="requirement-content">
              <h5>Not Suitable For Analysis</h5>
              <ul>
                <li>Smartphone or amateur photography</li>
                <li>Images with poor lighting or shadows</li>
                <li>Blurry, angled, or low-resolution images</li>
                <li>Non-clinical image capture methods</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      <div className="clinical-context">
        <div className="context-header">
          <Stethoscope className="context-icon" />
          <h4>Clinical Context & Interpretation</h4>
        </div>
        <div className="context-content">
          <p><strong>For Medical Professionals:</strong> This AI analysis should be interpreted within the broader clinical context including patient history, physical examination, and additional diagnostic procedures as appropriate.</p>
          <p><strong>Training Foundation:</strong> Model trained on HAM10000 and ISIC international clinical datasets, representing global standards in dermatoscopic imaging and analysis.</p>
          <p><strong>Clinical Integration:</strong> Designed to complement, not replace, professional medical judgment in dermatological practice.</p>
        </div>
      </div>
    </div>
  );
};

export default HomePage;