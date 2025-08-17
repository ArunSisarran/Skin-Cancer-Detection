import React from 'react';

const LogoIcon = ({ className = "", size = 24 }) => {
  return (
    <svg 
      xmlns="http://www.w3.org/2000/svg" 
      viewBox="0 0 64 64" 
      fill="none"
      width={size}
      height={size}
      className={className}
    >
      <defs>
        <linearGradient id="bgGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" style={{stopColor:"#3b82f6", stopOpacity:1}} />
          <stop offset="100%" style={{stopColor:"#8b5cf6", stopOpacity:1}} />
        </linearGradient>
        <linearGradient id="iconGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" style={{stopColor:"#ffffff", stopOpacity:1}} />
          <stop offset="100%" style={{stopColor:"#e0e7ff", stopOpacity:1}} />
        </linearGradient>
      </defs>
      
      {/* Background */}
      <circle cx="32" cy="32" r="30" fill="url(#bgGradient)" stroke="#1e40af" strokeWidth="2"/>
      
      {/* Medical cross/plus */}
      <rect x="28" y="18" width="8" height="28" fill="url(#iconGradient)" rx="2"/>
      <rect x="18" y="28" width="28" height="8" fill="url(#iconGradient)" rx="2"/>
      
      {/* Small diagnostic circles */}
      <circle cx="22" cy="22" r="2" fill="#fbbf24" opacity="0.8"/>
      <circle cx="42" cy="22" r="2" fill="#10b981" opacity="0.8"/>
      <circle cx="22" cy="42" r="2" fill="#ef4444" opacity="0.8"/>
      <circle cx="42" cy="42" r="2" fill="#3b82f6" opacity="0.8"/>
    </svg>
  );
};

export default LogoIcon;