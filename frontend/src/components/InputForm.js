import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const InputForm = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [formData, setFormData] = useState({
    kids: ['Kid_1', 'Kid_2', 'Kid_3'],
    presents: ['Present_1', 'Present_2', 'Present_3', 'Present_4'],
    valuations: {
      'Kid_1': {
        'Present_1': 5,  // Kid_1 can receive Present_1 with value 5
        'Present_2': 3,  // Kid_1 can receive Present_2 with value 3
        'Present_3': 0,  // Kid_1 cannot receive Present_3
        'Present_4': 0   // Kid_1 cannot receive Present_4
      },
      'Kid_2': {
        'Present_1': 5,  // Kid_2 can receive Present_1 with same value 5
        'Present_2': 0,  // Kid_2 cannot receive Present_2
        'Present_3': 7,  // Kid_2 can receive Present_3 with value 7
        'Present_4': 4   // Kid_2 can receive Present_4 with value 4
      },
      'Kid_3': {
        'Present_1': 0,  // Kid_3 cannot receive Present_1
        'Present_2': 3,  // Kid_3 can receive Present_2 with same value 3
        'Present_3': 7,  // Kid_3 can receive Present_3 with same value 7
        'Present_4': 4   // Kid_3 can receive Present_4 with same value 4
      }
    }
  });

  // Handle kid name change
  const handleKidNameChange = (index, value) => {
    const newKids = [...formData.kids];
    const oldName = newKids[index];
    newKids[index] = value;

    // Update the valuations object with the new kid name
    const newValuations = { ...formData.valuations };
    newValuations[value] = newValuations[oldName] || {};
    if (oldName !== value && newValuations[oldName]) {
      delete newValuations[oldName];
    }

    setFormData({
      ...formData,
      kids: newKids,
      valuations: newValuations
    });
  };

  // Handle present name change
  const handlePresentNameChange = (index, value) => {
    const newPresents = [...formData.presents];
    const oldName = newPresents[index];
    newPresents[index] = value;

    // Update the valuations for each kid with the new present name
    const newValuations = { ...formData.valuations };
    for (const kid in newValuations) {
      newValuations[kid][value] = newValuations[kid][oldName] || 0;
      if (oldName !== value && newValuations[kid][oldName] !== undefined) {
        delete newValuations[kid][oldName];
      }
    }

    setFormData({
      ...formData,
      presents: newPresents,
      valuations: newValuations
    });
  };

  // Handle valuation change
  const handleValuationChange = (kid, present, value) => {
    const numValue = parseInt(value) || 0;
    setFormData({
      ...formData,
      valuations: {
        ...formData.valuations,
        [kid]: {
          ...formData.valuations[kid],
          [present]: numValue
        }
      }
    });
  };

  // Add a new kid
  const addKid = () => {
    const newKidName = `Kid_${formData.kids.length + 1}`;
    const newValuations = { ...formData.valuations };
    
    // Initialize valuations for the new kid
    newValuations[newKidName] = {};
    formData.presents.forEach(present => {
      newValuations[newKidName][present] = 0;
    });

    setFormData({
      ...formData,
      kids: [...formData.kids, newKidName],
      valuations: newValuations
    });
  };

  // Remove a kid
  const removeKid = (index) => {
    if (formData.kids.length <= 1) {
      setError('You need at least one kid');
      return;
    }

    const newKids = [...formData.kids];
    const kidToRemove = newKids[index];
    newKids.splice(index, 1);

    // Remove the kid's valuations
    const newValuations = { ...formData.valuations };
    delete newValuations[kidToRemove];

    setFormData({
      ...formData,
      kids: newKids,
      valuations: newValuations
    });
    setError('');
  };

  // Add a new present
  const addPresent = () => {
    const newPresentName = `Present_${formData.presents.length + 1}`;
    const newValuations = { ...formData.valuations };
    
    // Initialize valuations for the new present for all kids
    for (const kid of formData.kids) {
      if (!newValuations[kid]) {
        newValuations[kid] = {};
      }
      newValuations[kid][newPresentName] = 0;
    }

    setFormData({
      ...formData,
      presents: [...formData.presents, newPresentName],
      valuations: newValuations
    });
  };

  // Remove a present
  const removePresent = (index) => {
    if (formData.presents.length <= 1) {
      setError('You need at least one present');
      return;
    }

    const newPresents = [...formData.presents];
    const presentToRemove = newPresents[index];
    newPresents.splice(index, 1);

    // Remove the present's valuations for all kids
    const newValuations = { ...formData.valuations };
    for (const kid in newValuations) {
      delete newValuations[kid][presentToRemove];
    }

    setFormData({
      ...formData,
      presents: newPresents,
      valuations: newValuations
    });
    setError('');
  };

  // State for random options modal
  const [showRandomOptions, setShowRandomOptions] = useState(false);
  const [randomizeOption, setRandomizeOption] = useState('all');
  const [numKids, setNumKids] = useState(3);
  const [numPresents, setNumPresents] = useState(5);

  // Open random options modal
  const openRandomOptions = () => {
    setShowRandomOptions(true);
  };

  // Close random options modal
  const closeRandomOptions = () => {
    setShowRandomOptions(false);
  };

  // Generate random input with selected options
  const generateRandomInput = async (option = randomizeOption) => {
    try {
      setLoading(true);
      setError('');
      
      // Create URL with query parameters
      let url = 'http://localhost:5006/api/generate-random?';
      url += `randomize_type=${option}`;
      url += `&num_kids=${numKids}`;
      url += `&num_presents=${numPresents}`;
      
      console.log(`Generating with params: type=${option}, kids=${numKids}, presents=${numPresents}`);
      
      // If we're doing partial randomization, pass the current data
      if (option !== 'all') {
        const currentData = JSON.stringify(formData);
        url += `&current_data=${encodeURIComponent(currentData)}`;
        console.log('Including current data for partial randomization');
      }
      
      // Get random data from the backend
      console.log(`Sending request to: ${url}`);
      const response = await axios.get(url);
      
      // Log what we received
      console.log('Received data:', response.data);
      console.log(`New data has ${response.data.kids.length} kids and ${response.data.presents.length} presents`);
      
      // Update form data with the response
      setFormData(response.data);
      
      // Close modal after generating
      setShowRandomOptions(false);
    } catch (error) {
      console.error('Error generating random input:', error);
      // Show more detailed error message
      let errorMessage = 'Failed to generate random input. Please try again.';
      if (error.response) {
        // The request was made and the server responded with an error status code
        errorMessage = `Server error (${error.response.status}): ${JSON.stringify(error.response.data)}`;
      } else if (error.request) {
        // The request was made but no response was received
        errorMessage = 'No response received from server. Is the backend running?';
      } else {
        // Something else happened
        errorMessage = `Error: ${error.message}`;
      }
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  // Check if the input follows the restricted assignment case
  const validateRestrictedAssignment = () => {
    // For each present, check if it has the same value for all kids who can receive it
    const presentValues = {};
    const errors = [];
    
    for (const present of formData.presents) {
      presentValues[present] = {};
      let valueForThisPresent = null;
      let hasValue = false;
      
      for (const kid of formData.kids) {
        const value = formData.valuations[kid]?.[present] || 0;
        
        if (value > 0) {
          if (valueForThisPresent === null) {
            // First kid who can receive this present
            valueForThisPresent = value;
            hasValue = true;
          } else if (value !== valueForThisPresent) {
            // Value doesn't match the expected value for this present
            errors.push(`Present ${present} has different values for different kids (${valueForThisPresent} vs ${value}). In the restricted assignment case, each present must have the same value for all kids who can receive it.`);
            break;
          }
        }
      }
    }
    
    return errors;
  };

  // Submit the form
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validate the form
    if (formData.kids.length === 0) {
      setError('You need at least one kid');
      return;
    }
    if (formData.presents.length === 0) {
      setError('You need at least one present');
      return;
    }
    
    // Validate restricted assignment case
    const restrictedAssignmentErrors = validateRestrictedAssignment();
    if (restrictedAssignmentErrors.length > 0) {
      setError(
        <div>
          <p><strong>Input does not follow the restricted assignment case:</strong></p>
          <ul>
            {restrictedAssignmentErrors.map((err, index) => (
              <li key={index}>{err}</li>
            ))}
          </ul>
          <p>In the restricted assignment case, each present must have the same value for all kids who can receive it (and 0 for kids who cannot).</p>
        </div>
      );
      return;
    }

    try {
      setLoading(true);
      setError('');
      
      // Send the data to the backend
      const response = await axios.post('http://localhost:5006/api/run-algorithm', formData);
      
      console.log('Algorithm response received:', response.data);
      
      // Navigate to the results page with the data
      console.log('Attempting to navigate to results page...');
      navigate('/results', { state: { results: response.data } });
      console.log('Navigation function called');
    } catch (error) {
      console.error('Error running algorithm:', error);
      setError('Failed to run algorithm. Please check your input and try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="input-form-container">
      <h2 className="section-title">Input Form</h2>
      
      {error && <div className="alert alert-danger">{error}</div>}
      
      <form onSubmit={handleSubmit}>
        <div className="form-section">
          <h3>Kids</h3>
          {formData.kids.map((kid, index) => (
            <div key={`kid-${index}`} className="input-group mb-3">
              <input
                type="text"
                className="form-control"
                value={kid}
                onChange={(e) => handleKidNameChange(index, e.target.value)}
                placeholder="Kid name"
                required
              />
              <button
                type="button"
                className="btn btn-outline-danger"
                onClick={() => removeKid(index)}
              >
                Remove
              </button>
            </div>
          ))}
          <button
            type="button"
            className="btn btn-outline-primary"
            onClick={addKid}
          >
            Add Kid
          </button>
        </div>

        <div className="form-section">
          <h3>Presents</h3>
          {formData.presents.map((present, index) => (
            <div key={`present-${index}`} className="input-group mb-3">
              <input
                type="text"
                className="form-control"
                value={present}
                onChange={(e) => handlePresentNameChange(index, e.target.value)}
                placeholder="Present name"
                required
              />
              <button
                type="button"
                className="btn btn-outline-danger"
                onClick={() => removePresent(index)}
              >
                Remove
              </button>
            </div>
          ))}
          <button
            type="button"
            className="btn btn-outline-primary"
            onClick={addPresent}
          >
            Add Present
          </button>
        </div>

        <div className="form-section">
          <h3>Valuations</h3>
          <p className="text-muted">Enter how much each kid values each present (0 means they can't receive it)</p>
          
          <div className="table-responsive">
            <table className="table table-bordered valuation-table">
              <thead>
                <tr>
                  <th>Kid / Present</th>
                  {formData.presents.map((present, index) => (
                    <th key={`header-${present}`}>{present}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {formData.kids.map((kid, kidIndex) => (
                  <tr key={`row-${kid}`}>
                    <td>{kid}</td>
                    {formData.presents.map((present, presentIndex) => (
                      <td key={`cell-${kid}-${present}`}>
                        <input
                          type="number"
                          className="form-control form-control-sm"
                          min="0"
                          value={formData.valuations[kid]?.[present] || 0}
                          onChange={(e) => handleValuationChange(kid, present, e.target.value)}
                        />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="d-flex justify-content-between mt-4">
          <button
            type="button"
            className="btn btn-secondary"
            onClick={openRandomOptions}
            disabled={loading}
          >
            {loading ? 'Loading...' : 'Generate Random Input'}
          </button>
          <button
            type="submit"
            className="btn btn-primary"
            disabled={loading}
          >
            {loading ? 'Running...' : 'Run Algorithm'}
          </button>
        </div>
        
        {/* Random Options Modal */}
        {showRandomOptions && (
          <div className="modal show" style={{display: 'block', backgroundColor: 'rgba(0,0,0,0.5)'}}>
            <div className="modal-dialog">
              <div className="modal-content">
                <div className="modal-header">
                  <h5 className="modal-title">Random Generation Options</h5>
                  <button type="button" className="btn-close" onClick={closeRandomOptions}></button>
                </div>
                <div className="modal-body">
                  <p>Choose what you want to randomize:</p>
                  
                  <div className="form-group mb-3">
                    <label htmlFor="numKids">Number of Kids:</label>
                    <input
                      type="number"
                      className="form-control"
                      id="numKids"
                      min="1"
                      max="10"
                      value={numKids}
                      onChange={(e) => setNumKids(Math.min(10, Math.max(1, parseInt(e.target.value) || 1)))}
                    />
                    <small className="text-muted">Maximum: 10</small>
                  </div>
                  
                  <div className="form-group mb-3">
                    <label htmlFor="numPresents">Number of Presents:</label>
                    <input
                      type="number"
                      className="form-control"
                      id="numPresents"
                      min="1"
                      max="20"
                      value={numPresents}
                      onChange={(e) => setNumPresents(Math.min(20, Math.max(1, parseInt(e.target.value) || 1)))}
                    />
                    <small className="text-muted">Maximum: 20</small>
                  </div>
                  
                  <p className="mt-3">What to randomize:</p>
                  
                  <div className="form-check mb-2">
                    <input
                      className="form-check-input"
                      type="radio"
                      name="randomizeOption"
                      id="randomize-all"
                      value="all"
                      checked={randomizeOption === 'all'}
                      onChange={() => setRandomizeOption('all')}
                    />
                    <label className="form-check-label" htmlFor="randomize-all">
                      Everything (kids, presents, and valuations)
                    </label>
                  </div>
                  
                  {/* האפשרויות "Only Kids" ו-"Only Presents" הוסרו בהתאם לבקשה */}
                  
                  <div className="form-check mb-2">
                    <input
                      className="form-check-input"
                      type="radio"
                      name="randomizeOption"
                      id="randomize-valuations"
                      value="valuations"
                      checked={randomizeOption === 'valuations'}
                      onChange={() => setRandomizeOption('valuations')}
                    />
                    <label className="form-check-label" htmlFor="randomize-valuations">
                      Only Valuations (keep kids and presents the same)
                    </label>
                  </div>
                </div>
                <div className="modal-footer">
                  <button type="button" className="btn btn-secondary" onClick={closeRandomOptions}>Cancel</button>
                  <button 
                    type="button" 
                    className="btn btn-primary" 
                    onClick={() => generateRandomInput()}
                    disabled={loading}
                  >
                    {loading ? 'Generating...' : 'Generate'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </form>
    </div>
  );
};

export default InputForm;
