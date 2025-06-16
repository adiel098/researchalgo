import React from 'react';
import { motion } from 'framer-motion';

const About = () => {
  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.3
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        type: 'spring',
        stiffness: 100,
        damping: 10
      }
    }
  };

  const cardVariants = {
    hover: {
      y: -10,
      boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
      transition: { type: 'spring', stiffness: 400, damping: 10 }
    }
  };

  const buttonVariants = {
    hover: { 
      scale: 1.05, 
      boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
      transition: { type: 'spring', stiffness: 400, damping: 10 }
    },
    tap: { scale: 0.98 }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 py-12 px-4 sm:px-6 lg:px-8">
      <motion.div 
        className="max-w-6xl mx-auto"
        initial="hidden"
        animate="visible"
        variants={containerVariants}
      >
        <motion.h1 
          className="text-4xl font-bold text-center mb-12 bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-purple-600"
          variants={itemVariants}
        >
          About Us
        </motion.h1>
        
        <motion.div 
          className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12"
          variants={itemVariants}
        >
          {/* Roey's Card */}
          <motion.div 
            className="card backdrop-blur-sm bg-white/90 overflow-hidden"
            variants={cardVariants}
            whileHover="hover"
          >
            <div className="h-2 bg-gradient-to-r from-blue-500 to-indigo-600"></div>
            <div className="p-6">
              <div className="flex items-center mb-6">
                <motion.div 
                  className="h-20 w-20 rounded-full bg-gradient-to-br from-blue-400 to-indigo-600 flex items-center justify-center text-white text-2xl font-bold mr-4 shadow-lg"
                  initial={{ rotate: -10 }}
                  animate={{ rotate: 0 }}
                  transition={{ type: 'spring', stiffness: 200 }}
                >
                  RS
                </motion.div>
                <div>
                  <h2 className="text-2xl font-bold text-slate-800">Roey Shmilovitch</h2>
                  <p className="text-indigo-600 font-medium">Algorithm Specialist</p>
                </div>
              </div>
              
              <div className="mb-6">
                <p className="text-slate-600 leading-relaxed">
                  Third-year Computer Science student specializing in algorithms and optimization. Experienced in software development and implementing complex algorithms.
                </p>
              </div>
              
              <div className="flex justify-center">
                <motion.div
                  whileHover="hover"
                  whileTap="tap"
                  variants={buttonVariants}
                >
                  <a 
                    href="#" 
                    className="btn btn-primary px-6 py-2 shadow-md"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
                    </svg>
                    Download CV
                  </a>
                </motion.div>
              </div>
            </div>
          </motion.div>
          
          {/* Adiel's Card */}
          <motion.div 
            className="card backdrop-blur-sm bg-white/90 overflow-hidden"
            variants={cardVariants}
            whileHover="hover"
          >
            <div className="h-2 bg-gradient-to-r from-green-500 to-emerald-600"></div>
            <div className="p-6">
              <div className="flex items-center mb-6">
                <motion.div 
                  className="h-20 w-20 rounded-full bg-gradient-to-br from-green-400 to-emerald-600 flex items-center justify-center text-white text-2xl font-bold mr-4 shadow-lg"
                  initial={{ rotate: 10 }}
                  animate={{ rotate: 0 }}
                  transition={{ type: 'spring', stiffness: 200 }}
                >
                  AH
                </motion.div>
                <div>
                  <h2 className="text-2xl font-bold text-slate-800">Adiel Halevi</h2>
                  <p className="text-emerald-600 font-medium">AI & ML Specialist</p>
                </div>
              </div>
              
              <div className="mb-6">
                <p className="text-slate-600 leading-relaxed">
                  Third-year Computer Science student specializing in artificial intelligence and machine learning. Focused on developing efficient algorithms and innovative software solutions.
                </p>
              </div>
              
              <div className="flex justify-center">
                <motion.div
                  whileHover="hover"
                  whileTap="tap"
                  variants={buttonVariants}
                >
                  <a 
                    href="https://drive.google.com/file/d/1KX-3S1jZgos6IFLdp8dz3yXMXEoyPdf-/view?usp=sharing" 
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn btn-success px-6 py-2 shadow-md"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
                    </svg>
                    Download CV
                  </a>
                </motion.div>
              </div>
            </div>
          </motion.div>
        </motion.div>
        
        <motion.div 
          className="card backdrop-blur-sm bg-white/90 overflow-hidden mb-12"
          variants={itemVariants}
          whileHover={{ y: -5, transition: { duration: 0.2 } }}
        >
          <div className="p-6 border-l-4 border-purple-500">
            <div className="flex items-center mb-4">
              <div className="h-10 w-10 rounded-full bg-purple-100 flex items-center justify-center mr-3">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
              </div>
              <h2 className="text-2xl font-bold text-slate-800">Our Project</h2>
            </div>
            
            <div className="space-y-4 text-slate-600">
              <motion.p 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.6 }}
              >
                This project was developed as part of our Research Algorithms course. We implemented the Santa Claus Problem algorithm with a focus on the restricted assignment case, and created a user-friendly web interface to demonstrate its functionality.
              </motion.p>
              
              <motion.p 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.8 }}
              >
                The implementation includes both the algorithm logic and a full-stack web application with React frontend and Flask backend. The application allows users to input their own data or generate random examples that follow the restricted assignment case constraints.
              </motion.p>
              
              <motion.div 
                className="flex flex-wrap gap-3 mt-4"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 1 }}
              >
                {['React', 'Flask', 'Python', 'JavaScript', 'Algorithm Design', 'Linear Programming'].map((tag, index) => (
                  <span 
                    key={index} 
                    className="px-3 py-1 rounded-full bg-purple-100 text-purple-800 text-sm font-medium"
                  >
                    {tag}
                  </span>
                ))}
              </motion.div>
            </div>
          </div>
        </motion.div>
        
        <motion.div 
          className="text-center text-slate-500 text-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2 }}
        >
          <p>&copy; {new Date().getFullYear()} Research Algorithms Project</p>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default About;
