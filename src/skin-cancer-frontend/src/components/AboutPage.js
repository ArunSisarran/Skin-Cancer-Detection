import React from 'react';
import { Heart, Target, Users, Shield, Brain, Stethoscope } from 'lucide-react';
import './AboutPage.css';

const AboutPage = () => {
  return (
    <div className="about-page">
      <div className="page-header">
        <h1 className="page-title">About SkinGuard AI</h1>
        <p className="page-subtitle">
          Professional dermatoscopic melanoma analysis tool designed for medical professionals. Trained on international clinical datasets to assist in analysis of professional medical imagery.
        </p>
      </div>

      <div className="mission-card">
        <div className="mission-header">
          <div className="mission-icon">
            <Heart />
          </div>
          <h2 className="mission-title">Our Mission</h2>
        </div>
        
        <div className="mission-content">
          <p>
            Supporting dermatological professionals with AI-assisted analysis of dermatoscopic images. Our tool is specifically trained on clinical datasets to help analyze professional medical imagery, complementing expert diagnosis in clinical settings.
          </p>
          <p>
            By leveraging machine learning trained on international medical imaging standards, we aim to provide healthcare professionals with an additional analytical tool that can assist in the complex task of melanoma detection from dermatoscopic imagery.
          </p>
          <p>
            This project represents the intersection of computer vision and medical imaging, designed to work within established clinical workflows while maintaining the highest standards of medical ethics and professional responsibility.
          </p>
        </div>
      </div>

      <div className="features-grid">
        <div className="feature-card">
          <div className="feature-icon blue">
            <Brain />
          </div>
          <h3 className="feature-title">Clinical-Grade AI Analysis</h3>
          <p className="feature-description">
            Advanced deep learning model trained on HAM10000 and ISIC datasets - the international gold standard for dermatoscopic image analysis.
          </p>
        </div>

        <div className="feature-card">
          <div className="feature-icon green">
            <Target />
          </div>
          <h3 className="feature-title">Professional Medical Focus</h3>
          <p className="feature-description">
            Designed specifically for dermatoscopic images captured in clinical settings using standardized medical imaging protocols.
          </p>
        </div>

        <div className="feature-card">
          <div className="feature-icon purple">
            <Users />
          </div>
          <h3 className="feature-title">Clinical Decision Support</h3>
          <p className="feature-description">
            Assists medical professionals in analyzing dermatoscopic imagery within comprehensive clinical examination workflows.
          </p>
        </div>
      </div>

      <div className="technical-section">
        <h2 className="section-title">Technical Approach</h2>
        <div className="technical-steps">
          <div className="technical-step">
            <div className="step-number">
              <span>1</span>
            </div>
            <div className="step-content">
              <h3>International Clinical Datasets</h3>
              <p>
                Trained on HAM10000 (Human Against Machine) and ISIC (International Skin Imaging Collaboration) datasets, representing over 20,000 professional dermatoscopic images from multiple international medical institutions.
              </p>
            </div>
          </div>

          <div className="technical-step">
            <div className="step-number">
              <span>2</span>
            </div>
            <div className="step-content">
              <h3>EfficientNet-B0 Architecture</h3>
              <p>
                Utilizes EfficientNet-B0 backbone with transfer learning, specifically optimized for dermatoscopic image characteristics including lighting, magnification, and clinical photography standards.
              </p>
            </div>
          </div>

          <div className="technical-step">
            <div className="step-number">
              <span>3</span>
            </div>
            <div className="step-content">
              <h3>Clinical Validation & Performance</h3>
              <p>
                Optimized for high sensitivity on melanoma detection using focal loss and class balancing techniques. Performance metrics validated specifically on professional medical imagery.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="target-users-section">
        <h2 className="section-title">Intended Users</h2>
        <div className="users-grid">
          <div className="user-card">
            <div className="user-icon">
              <Stethoscope />
            </div>
            <h4>Dermatologists</h4>
            <p>Clinical decision support for analyzing dermatoscopic images in practice</p>
          </div>
          <div className="user-card">
            <div className="user-icon">
              <Brain />
            </div>
            <h4>Medical Residents</h4>
            <p>Educational tool for learning dermatoscopic image analysis</p>
          </div>
          <div className="user-card">
            <div className="user-icon">
              <Users />
            </div>
            <h4>Medical Researchers</h4>
            <p>Research applications with standardized clinical imagery</p>
          </div>
        </div>
      </div>

      <div className="disclaimers-section">
        <div className="disclaimers-header">
          <Shield className="disclaimers-icon" />
          <h2 className="disclaimers-title">Important Medical Disclaimers</h2>
        </div>
        
        <div className="disclaimers-list">
          <div className="disclaimer-item">
            <Stethoscope className="disclaimer-item-icon" />
            <p className="disclaimer-content">
              <strong>Professional Use Only:</strong> This tool is designed exclusively for analysis of professional dermatoscopic images captured in clinical settings. Not suitable for consumer smartphone photos.
            </p>
          </div>
          
          <div className="disclaimer-item">
            <Users className="disclaimer-item-icon" />
            <p className="disclaimer-content">
              <strong>Clinical Context Required:</strong> Results should only be interpreted by qualified medical professionals within the context of comprehensive clinical examination and patient history.
            </p>
          </div>
          
          <div className="disclaimer-item">
            <Target className="disclaimer-item-icon" />
            <p className="disclaimer-content">
              <strong>Decision Support Tool:</strong> This AI system provides analysis assistance only. It cannot replace professional medical judgment and should never be used as a primary diagnostic tool.
            </p>
          </div>

          <div className="disclaimer-item">
            <Brain className="disclaimer-item-icon" />
            <p className="disclaimer-content">
              <strong>Training Data Scope:</strong> Model trained on HAM10000 and ISIC clinical datasets. Performance validated only on professional dermatoscopic imagery following medical imaging standards.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AboutPage;