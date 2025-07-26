import React, { useState } from 'react';
import { Upload, Camera, AlertTriangle, Loader2, Info, Activity } from 'lucide-react';
import './HomePage.css';

const HomePage = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);

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
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Analysis failed. Please try again.');
      }

      const result = await response.json();
      setPrediction(result);
    } catch (err) {
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
            <p className="disclaimer-title">Important Medical Disclaimer</p>
            <p>This tool is for educational purposes only and should not replace professional medical advice. Always consult with a qualified dermatologist for proper diagnosis and treatment.</p>
          </div>
        </div>
      </div>

      <div className="main-grid">
        <div className="upload-section">
          <div className="section-header">
            <h2>Upload Skin Image</h2>
            <p>Upload a clear photo of the skin lesion for AI analysis</p>
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
                  alt="Selected skin image"
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
                  <p className="upload-title">Drop your image here</p>
                  <p className="upload-subtitle">or click to browse</p>
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
                  <span>Analyze Skin Lesion</span>
                </>
              )}
            </button>
          )}
        </div>

        <div className="results-section">
          <div className="section-header">
            <h3>Analysis Results</h3>
            <p>AI-powered skin lesion analysis results will appear here</p>
          </div>

          {error && (
            <div className="error-message">
              <div className="error-header">
                <AlertTriangle className="error-icon" />
                <p className="error-title">Analysis Error</p>
              </div>
              <p className="error-text">{error}</p>
            </div>
          )}

          {prediction && (
            <div className="results-card">
              <div className="result-header">
                <h4 className="result-title">Prediction Result</h4>
                <div className={`result-badge ${prediction.prediction === 'Melanoma' ? 'melanoma' : 'non-melanoma'}`}>
                  {prediction.prediction}
                </div>
              </div>

              <div className="confidence-section">
                <div className="confidence-header">
                  <span className="confidence-label">Confidence Score</span>
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
                  <h5>Detailed Probabilities</h5>
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
                    Analyzed on {new Date(prediction.timestamp).toLocaleString()}
                  </p>
                </div>
              </div>
            </div>
          )}

          {!prediction && !error && !loading && (
            <div className="results-placeholder">
              <Activity className="placeholder-icon" />
              <p className="placeholder-text">Upload an image to see analysis results</p>
            </div>
          )}
        </div>
      </div>

      <div className="guidelines">
        <h4>For Best Results</h4>
        <div className="guidelines-grid">
          <div className="guidelines-column">
            <p className="guideline-item">• Use good lighting and clear focus</p>
            <p className="guideline-item">• Ensure the lesion fills most of the frame</p>
            <p className="guideline-item">• Avoid shadows or reflections</p>
          </div>
          <div className="guidelines-column">
            <p className="guideline-item">• Take photos straight-on, not at an angle</p>
            <p className="guideline-item">• Use a high-resolution camera</p>
            <p className="guideline-item">• Include a reference object for scale if possible</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;