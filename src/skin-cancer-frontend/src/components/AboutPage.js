import React from 'react';
import { Heart, Target, Users, Shield, Brain, Stethoscope } from 'lucide-react';
import './AboutPage.css';

const AboutPage = () => {
  return (
    <div className="about-page">
      <div className="page-header">
        <h1 className="page-title">About SkinGuard AI</h1>
        <p className="page-subtitle">
          Leveraging artificial intelligence to assist in early skin cancer detection and promote better skin health awareness.
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
            As a fourth-year Computer Science student, I’ve developed a strong interest in machine learning and its real-world applications—particularly in the field of computer vision. While my coursework provided a solid theoretical foundation, I wanted to challenge myself with a hands-on project that would push me to apply those concepts in a practical and impactful way.
          </p>
          <p>
            This website was built as a personal learning experience to deepen my understanding of convolutional neural networks (CNNs) and the process of fine-tuning pre-trained models. It also served as an opportunity to explore the full pipeline of developing, training, and deploying a machine learning model in a real-world context.
          </p>
          <p>
            Beyond the technical goals, this project was driven by a personal motivation: helping people who often delay going to the doctor, even when they know they shouldn’t. Whether due to busy schedules, fear, or simply putting it off, many let their health slide until it becomes a serious issue. I wanted to create something that could act as a small but meaningful step—a tool that encourages users to reflect on their well-being and consider seeking professional medical advice sooner rather than later.
          </p>
        </div>
      </div>

      <div className="features-grid">
        <div className="feature-card">
          <div className="feature-icon blue">
            <Brain />
          </div>
          <h3 className="feature-title">AI-Powered Analysis</h3>
          <p className="feature-description">
            Advanced deep learning model trained on thousands of skin lesion images for accurate classification.
          </p>
        </div>

        <div className="feature-card">
          <div className="feature-icon green">
            <Target />
          </div>
          <h3 className="feature-title">Early Detection</h3>
          <p className="feature-description">
            Designed to assist in identifying potential melanomas and concerning skin lesions early.
          </p>
        </div>

        <div className="feature-card">
          <div className="feature-icon purple">
            <Users />
          </div>
          <h3 className="feature-title">Accessible Healthcare</h3>
          <p className="feature-description">
            Making preliminary skin analysis more accessible to communities with limited healthcare access.
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
              <h3>Dataset & Training</h3>
              <p>
                Trained on the HAM10000 dataset containing over 10,000 dermatoscopic images, focusing on binary classification between melanoma and non-melanoma lesions.
              </p>
            </div>
          </div>

          <div className="technical-step">
            <div className="step-number">
              <span>2</span>
            </div>
            <div className="step-content">
              <h3>Model Architecture</h3>
              <p>
                Utilizes EfficientNet-B0 as the backbone with transfer learning, optimized for both accuracy and inference speed.
              </p>
            </div>
          </div>

          <div className="technical-step">
            <div className="step-number">
              <span>3</span>
            </div>
            <div className="step-content">
              <h3>Validation & Performance</h3>
              <p>
                Achieved high sensitivity for melanoma detection through careful hyperparameter tuning and class balancing techniques.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="disclaimers-section">
        <div className="disclaimers-header">
          <Shield className="disclaimers-icon" />
          <h2 className="disclaimers-title">Important Disclaimers</h2>
        </div>
        
        <div className="disclaimers-list">
          <div className="disclaimer-item">
            <Stethoscope className="disclaimer-item-icon" />
            <p className="disclaimer-content">
              <strong>Not a Medical Device:</strong> This tool is for educational and research purposes only. It is not intended to diagnose, treat, cure, or prevent any disease.
            </p>
          </div>
          
          <div className="disclaimer-item">
            <Users className="disclaimer-item-icon" />
            <p className="disclaimer-content">
              <strong>Consult Healthcare Professionals:</strong> Always seek the advice of a qualified dermatologist or healthcare provider for any skin concerns or lesions.
            </p>
          </div>
          
          <div className="disclaimer-item">
            <Target className="disclaimer-item-icon" />
            <p className="disclaimer-content">
              <strong>AI Limitations:</strong> While AI can assist in analysis, it cannot replace professional medical judgment and may produce false positives or negatives.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AboutPage;