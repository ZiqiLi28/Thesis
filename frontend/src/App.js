import React, { useState } from 'react';
import './App.css';

function App() {
  const [fileName, setFileName] = useState('');

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setFileName(file.name);
    }
  };

  return (
    <div className="App">
      <h1>Number Plate Recognition System</h1>
      <div className="upload-section">
        <input 
          type="text" 
          value={fileName} 
          placeholder="No file selected" 
          readOnly 
        />
        <label htmlFor="upload-button" className="upload-button">
          Upload Image
        </label>
        <input 
          type="file" 
          id="upload-button" 
          style={{ display: 'none' }} 
          onChange={handleFileChange} 
        />
      </div>
      <button className="recognize-button">Recognize</button>
      <footer>
        <p>Produced by Ziqi Li</p>
      </footer>
    </div>
  );
}

export default App;
