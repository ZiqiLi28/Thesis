import React, { useState } from 'react';
import './App.css';

function App() {
  const [fileName, setFileName] = useState('');
  const [imageFile, setImageFile] = useState(null);
  const [logs, setLogs] = useState('');
  const [error, setError] = useState('');

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setFileName(file.name);
      setImageFile(file);
    }
  };

  const handleRecognize = async () => {
    if (!imageFile) {
      alert('Please upload an image first!');
      return;
    }

    const formData = new FormData();
    formData.append('image', imageFile);

    try {
      const response = await fetch('http://127.0.0.1:5000/recognize', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();

      if (response.ok) {
        setLogs(data.logs);
        setError('');
      } else {
        setError(data.error || 'Something went wrong');
        setLogs(data.logs || '');
      }
    } catch (err) {
      setError('Failed to connect to server');
      setLogs('');
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
      <button className="recognize-button" onClick={handleRecognize}>Recognize</button>
      {error && <p className="error">{error}</p>}
      {logs && <pre className="logs">{logs}</pre>}
      <footer>
        <p>Produced by <a href="https://github.com/ZiqiLi28" target="_blank" rel="noopener noreferrer">Ziqi Li</a></p>
      </footer>
    </div>
  );
}

export default App;
