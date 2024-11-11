import React, { useState } from 'react';
// import {BrowserRouter as Router, Route, Routes,Link} from 'react-router-dom';
// import Error from './Error';
// import Success from './Success';
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
      <p>Please ensure the image is as clear as possible</p>
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
      <p>Produced by <a href="https://github.com/ZiqiLi28" target="_blank" rel="noopener noreferrer">Ziqi Li</a></p>
      </footer>
    </div>
  );
}

export default App;
