import React, { useState } from 'react';
import Header from './components/Header';
import HomePage from './components/HomePage';
import AboutPage from './components/AboutPage';

const App = () => {
  const [activeTab, setActiveTab] = useState('home');

  return (
    <div className="min-h-screen bg-gray-50">
      <Header activeTab={activeTab} setActiveTab={setActiveTab} />
      
      <main className="pt-4">
        {activeTab === 'home' && <HomePage />}
        {activeTab === 'about' && <AboutPage />}
      </main>
    </div>
  );
};

export default App;
