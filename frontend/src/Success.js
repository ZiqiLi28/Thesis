import React from 'react';
import './App.css';

function Success({ licensePlate }) {
  return (
    <div className="App">
      <h1>Recognition Result</h1>
      <div className="result-section">
        <p className="result-message">{licensePlate}</p>
      </div>
      <footer>
        <p>Produced by <a href="https://github.com/ZiqiLi28" target="_blank" rel="noopener noreferrer">Ziqi Li</a></p>
      </footer>
    </div>
  );
}

export default Success;