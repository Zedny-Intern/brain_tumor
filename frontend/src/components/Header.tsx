/**
 * Header Component
 * Navigation and branding for the application
 */

import React from 'react';
import './Header.css';

const Header: React.FC = () => {
    return (
        <header className="header">
            <div className="container">
                <div className="header-content">
                    <div className="logo">
                        <div className="logo-icon">🧠</div>
                        <div className="logo-text">
                            <h2>Brain Tumor</h2>
                            <span>Classification & Segmentation</span>
                        </div>
                    </div>

                    <nav className="nav">
                        <a href="#home" className="nav-link">Home</a>
                        <a href="#predict" className="nav-link">Predict</a>
                        <a href="#models" className="nav-link">Models</a>
                        <a href="#about" className="nav-link">About</a>
                    </nav>
                </div>
            </div>
        </header>
    );
};

export default Header;
