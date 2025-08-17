import React from 'react';
import { Home, FileText } from 'lucide-react';
import LogoIcon from './LogoIcon'; // Import the new LogoIcon
import './Header.css';

const Header = ({ activeTab, setActiveTab }) => {
  return (
    <header className="header">
      <div className="header-container">
        <div className="header-content">
          <div className="header-left">
            <div className="logo-container">
              <div className="logo-icon">
                <LogoIcon size={32} />
              </div>
              <h1 className="logo-text">SkinGuard AI</h1>
            </div>
          </div>
          
          <nav className="nav">
            <button
              onClick={() => setActiveTab('home')}
              className={`nav-button ${activeTab === 'home' ? 'active' : ''}`}
            >
              <Home className="nav-icon" />
              <span>Home</span>
            </button>
            <button
              onClick={() => setActiveTab('about')}
              className={`nav-button ${activeTab === 'about' ? 'active' : ''}`}
            >
              <FileText className="nav-icon" />
              <span>About</span>
            </button>
          </nav>
        </div>
      </div>
    </header>
  );
};

export default Header;