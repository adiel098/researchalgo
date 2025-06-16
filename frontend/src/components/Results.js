import React from 'react';
import { useLocation, Link } from 'react-router-dom';

const Results = () => {
  const location = useLocation();
  const results = location.state?.results;

  // If no results are available, show a message
  if (!results) {
    return (
      <div className="results-container">
        <div className="alert alert-warning">
          No results available. Please run the algorithm first.
        </div>
        <Link to="/input" className="btn btn-primary">
          Go to Input Form
        </Link>
      </div>
    );
  }

  // Check if there was an error
  if (results.error) {
    return (
      <div className="results-container">
        <h2 className="section-title">Algorithm Error</h2>
        <div className="alert alert-danger">{results.error}</div>
        
        <h3>Debug Logs</h3>
        <div className="log-container">
          {results.logs && results.logs.map((log, index) => (
            <div key={index} className="log-entry">
              {log}
            </div>
          ))}
        </div>
        
        <Link to="/input" className="btn btn-primary mt-3">
          Back to Input Form
        </Link>
      </div>
    );
  }

  // Format the input data for display
  const { kids, presents, valuations } = results.input;
  
  return (
    <div className="results-container">
      <h2 className="section-title">Algorithm Results</h2>
      
      <div className="row">
        <div className="col-md-6">
          <div className="card">
            <div className="card-header bg-primary text-white">
              <h3 className="card-title mb-0">Input Data</h3>
            </div>
            <div className="card-body">
              <h4>Kids</h4>
              <ul className="list-group mb-3">
                {kids.map((kid, index) => (
                  <li key={index} className="list-group-item">{kid}</li>
                ))}
              </ul>
              
              <h4>Presents</h4>
              <ul className="list-group mb-3">
                {presents.map((present, index) => (
                  <li key={index} className="list-group-item">{present}</li>
                ))}
              </ul>
              
              <h4>Valuations</h4>
              <div className="table-responsive">
                <table className="table table-bordered table-sm">
                  <thead>
                    <tr>
                      <th>Kid / Present</th>
                      {presents.map((present, index) => (
                        <th key={index}>{present}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {kids.map((kid, kidIndex) => (
                      <tr key={kidIndex}>
                        <td>{kid}</td>
                        {presents.map((present, presentIndex) => (
                          <td key={presentIndex}>
                            {valuations[kid]?.[present] || 0}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
        
        <div className="col-md-6">
          <div className="card">
            <div className="card-header bg-success text-white">
              <h3 className="card-title mb-0">Algorithm Output</h3>
            </div>
            <div className="card-body">
              <h4>Optimal Value: {results.optimal_value}</h4>
              <p className="text-muted">
                This is the minimum happiness value achieved by any kid in the allocation.
              </p>
              
              <h4>Final Allocation</h4>
              {Object.entries(results.allocation).map(([kid, presents]) => (
                <div key={kid} className="card allocation-card">
                  <div className="card-header">
                    <strong>{kid}</strong>
                  </div>
                  <div className="card-body">
                    {presents.length > 0 ? (
                      <ul className="list-group">
                        {presents.map((present, index) => (
                          <li key={index} className="list-group-item">
                            {present} (Value: {valuations[kid]?.[present] || 0})
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="text-muted">No presents allocated</p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
      
      <div className="card mt-4">
        <div className="card-header bg-info text-white">
          <h3 className="card-title mb-0">Algorithm Logs</h3>
        </div>
        <div className="card-body">
          <div className="log-container">
            {results.logs && results.logs.map((log, index) => (
              <div key={index} className="log-entry">
                {log}
              </div>
            ))}
          </div>
        </div>
      </div>
      
      <div className="mt-4">
        <Link to="/input" className="btn btn-primary me-2">
          Back to Input Form
        </Link>
        <button 
          className="btn btn-secondary"
          onClick={() => {
            const dataStr = JSON.stringify(results, null, 2);
            const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
            const exportFileDefaultName = 'santa_claus_results.json';
            
            const linkElement = document.createElement('a');
            linkElement.setAttribute('href', dataUri);
            linkElement.setAttribute('download', exportFileDefaultName);
            linkElement.click();
          }}
        >
          Export Results
        </button>
      </div>
    </div>
  );
};

export default Results;
