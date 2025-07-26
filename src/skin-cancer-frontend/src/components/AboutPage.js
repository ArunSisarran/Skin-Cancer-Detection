import React from 'react';
import { Heart, Target, Users, Shield, Brain, Stethoscope } from 'lucide-react';

const AboutPage = () => {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">About SkinGuard AI</h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Leveraging artificial intelligence to assist in early skin cancer detection and promote better skin health awareness.
        </p>
      </div>

      {/* Mission Statement */}
      <div className="bg-gradient-to-br from-blue-50 to-purple-50 border border-blue-200 rounded-2xl p-8 mb-12">
        <div className="flex items-center space-x-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
            <Heart className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-2xl font-bold text-gray-900">Our Mission</h2>
        </div>
        
        <div className="prose prose-lg text-gray-700 max-w-none">
          <p className="mb-4">
            [Your mission statement will go here - replace this placeholder text with your personal mission and intentions for building this project.]
          </p>
          <p className="mb-4">
            [You can describe your motivation, the problem you're trying to solve, and the impact you hope to achieve.]
          </p>
          <p>
            [Feel free to include your background, why this project matters to you, and your vision for the future of AI-assisted healthcare.]
          </p>
        </div>
      </div>

      {/* Key Features */}
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
        <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
          <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
            <Brain className="w-6 h-6 text-blue-600" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">AI-Powered Analysis</h3>
          <p className="text-gray-600 text-sm">
            Advanced deep learning model trained on thousands of skin lesion images for accurate classification.
          </p>
        </div>

        <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
          <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mb-4">
            <Target className="w-6 h-6 text-green-600" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Early Detection</h3>
          <p className="text-gray-600 text-sm">
            Designed to assist in identifying potential melanomas and concerning skin lesions early.
          </p>
        </div>

        <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
          <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mb-4">
            <Users className="w-6 h-6 text-purple-600" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Accessible Healthcare</h3>
          <p className="text-gray-600 text-sm">
            Making preliminary skin analysis more accessible to communities with limited healthcare access.
          </p>
        </div>
      </div>

      {/* Technical Approach */}
      <div className="bg-white border border-gray-200 rounded-2xl p-8 mb-12">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Technical Approach</h2>
        <div className="space-y-6">
          <div className="flex items-start space-x-4">
            <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
              <span className="text-blue-600 font-semibold text-sm">1</span>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 mb-2">Dataset & Training</h3>
              <p className="text-gray-600 text-sm">
                Trained on the HAM10000 dataset containing over 10,000 dermatoscopic images, focusing on binary classification between melanoma and non-melanoma lesions.
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-4">
            <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
              <span className="text-blue-600 font-semibold text-sm">2</span>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 mb-2">Model Architecture</h3>
              <p className="text-gray-600 text-sm">
                Utilizes EfficientNet-B0 as the backbone with transfer learning, optimized for both accuracy and inference speed.
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-4">
            <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
              <span className="text-blue-600 font-semibold text-sm">3</span>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 mb-2">Validation & Performance</h3>
              <p className="text-gray-600 text-sm">
                Achieved high sensitivity for melanoma detection through careful hyperparameter tuning and class balancing techniques.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Important Disclaimers */}
      <div className="bg-amber-50 border border-amber-200 rounded-2xl p-8">
        <div className="flex items-center space-x-3 mb-6">
          <Shield className="w-8 h-8 text-amber-600" />
          <h2 className="text-2xl font-bold text-gray-900">Important Disclaimers</h2>
        </div>
        
        <div className="space-y-4 text-sm text-amber-800">
          <div className="flex items-start space-x-3">
            <Stethoscope className="w-5 h-5 text-amber-600 mt-0.5 flex-shrink-0" />
            <p>
              <strong>Not a Medical Device:</strong> This tool is for educational and research purposes only. It is not intended to diagnose, treat, cure, or prevent any disease.
            </p>
          </div>
          
          <div className="flex items-start space-x-3">
            <Users className="w-5 h-5 text-amber-600 mt-0.5 flex-shrink-0" />
            <p>
              <strong>Consult Healthcare Professionals:</strong> Always seek the advice of a qualified dermatologist or healthcare provider for any skin concerns or lesions.
            </p>
          </div>
          
          <div className="flex items-start space-x-3">
            <Target className="w-5 h-5 text-amber-600 mt-0.5 flex-shrink-0" />
            <p>
              <strong>AI Limitations:</strong> While AI can assist in analysis, it cannot replace professional medical judgment and may produce false positives or negatives.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AboutPage;