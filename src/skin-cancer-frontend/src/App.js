import React, { useState } from 'react';
import './App.css';
import Header from './components/Header';
import HomePage from './components/HomePage';
import AboutPage from './components/AboutPage';

const App = () => {
  const [activeTab, setActiveTab] = useState('home');

  return (
    <div className="App">
      <Header activeTab={activeTab} setActiveTab={setActiveTab} />
      
      <main>
        {activeTab === 'home' && <HomePage />}
        {activeTab === 'about' && <AboutPage />}
      </main>
    </div>
  );
};

export default App;