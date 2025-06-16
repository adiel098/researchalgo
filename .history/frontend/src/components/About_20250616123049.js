import React from 'react';

const About = () => {
  return (
    <div className="about-section">
      <h2 className="section-title">About Us</h2>
      
      <div className="row">
        <div className="col-md-6">
          <div className="card mb-4">
            <div className="card-header bg-info text-white">
              <h3 className="card-title mb-0">Roey Shmilovitch</h3>
            </div>
            <div className="card-body">
              <div className="text-center mb-3">
                <div 
                  className="rounded-circle mx-auto d-flex align-items-center justify-content-center" 
                  style={{ 
                    width: '150px', 
                    height: '150px', 
                    backgroundColor: '#e9ecef',
                    fontSize: '60px',
                    color: '#495057'
                  }}
                >
                  RS
                </div>
              </div>
              <p className="card-text">
                סטודנט שנה שלישית למדעי המחשב עם התמחות באלגוריתמים ואופטימיזציה. בעל ניסיון בפיתוח תוכנה ויישום אלגוריתמים מורכבים.
              </p>
              <p className="card-text">
                Third-year Computer Science student specializing in algorithms and optimization. Experienced in software development and implementing complex algorithms.
              </p>
              <div className="text-center mt-3">
                <a 
                  href="#" 
                  className="btn btn-primary" 
                  style={{
                    background: 'linear-gradient(135deg, #4b6cb7, #182848)',
                    border: 'none',
                    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '8px'
                  }}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" className="bi bi-file-earmark-text" viewBox="0 0 16 16">
                    <path d="M5.5 7a.5.5 0 0 0 0 1h5a.5.5 0 0 0 0-1h-5zM5 9.5a.5.5 0 0 1 .5-.5h5a.5.5 0 0 1 0 1h-5a.5.5 0 0 1-.5-.5zm0 2a.5.5 0 0 1 .5-.5h2a.5.5 0 0 1 0 1h-2a.5.5 0 0 1-.5-.5z"/>
                    <path d="M9.5 0H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V4.5L9.5 0zm0 1v2A1.5 1.5 0 0 0 11 4.5h2V14a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1h5.5z"/>
                  </svg>
                  Download CV
                </a>
              </div>
            </div>
          </div>
        </div>
        
        <div className="col-md-6">
          <div className="card mb-4">
            <div className="card-header bg-success text-white">
              <h3 className="card-title mb-0">Adiel Halevi</h3>
            </div>
            <div className="card-body">
              <div className="text-center mb-3">
                <div 
                  className="rounded-circle mx-auto d-flex align-items-center justify-content-center" 
                  style={{ 
                    width: '150px', 
                    height: '150px', 
                    backgroundColor: '#e9ecef',
                    fontSize: '60px',
                    color: '#495057'
                  }}
                >
                  AH
                </div>
              </div>
              <p className="card-text">
                סטודנט שנה שלישית למדעי המחשב עם התמחות בבינה מלאכותית ולמידת מכונה. מתמחה בפיתוח אלגוריתמים יעילים ופתרונות תוכנה חדשניים.
              </p>
              <p className="card-text">
                Third-year Computer Science student specializing in artificial intelligence and machine learning. Focused on developing efficient algorithms and innovative software solutions.
              </p>
              <div className="text-center mt-3">
                <a 
                  href="https://drive.google.com/file/d/1KX-3S1jZgos6IFLdp8dz3yXMXEoyPdf-/view?usp=sharing" 
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn btn-success" 
                  style={{
                    background: 'linear-gradient(135deg, #43a047, #1b5e20)',
                    border: 'none',
                    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '8px'
                  }}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" className="bi bi-file-earmark-text" viewBox="0 0 16 16">
                    <path d="M5.5 7a.5.5 0 0 0 0 1h5a.5.5 0 0 0 0-1h-5zM5 9.5a.5.5 0 0 1 .5-.5h5a.5.5 0 0 1 0 1h-5a.5.5 0 0 1-.5-.5zm0 2a.5.5 0 0 1 .5-.5h2a.5.5 0 0 1 0 1h-2a.5.5 0 0 1-.5-.5z"/>
                    <path d="M9.5 0H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V4.5L9.5 0zm0 1v2A1.5 1.5 0 0 0 11 4.5h2V14a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1h5.5z"/>
                  </svg>
                  Download CV
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <div className="card mb-4">
        <div className="card-header bg-dark text-white">
          <h3 className="card-title mb-0">Our Project</h3>
        </div>
        <div className="card-body">
          <p>
            This project was developed as part of our Research Algorithms course. We implemented the Santa Claus Problem algorithm with a focus on the restricted assignment case, and created a user-friendly web interface to demonstrate its functionality.
          </p>
          <p>
            The implementation includes both the algorithm logic and a full-stack web application with React frontend and Flask backend. The application allows users to input their own data or generate random examples that follow the restricted assignment case constraints.
          </p>
        </div>
      </div>
      
    </div>
  );
};

export default About;
