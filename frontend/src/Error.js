import React from 'react';
import './App.css';

function Error() {
  return (
    <div className="App">
      <h1>Recognition Result</h1>
      <div className="result-section">
        <p className="result-message">Recognition error, please try again.</p>
      </div>
      <footer>
        <p>Produced by <a href="https://github.com/ZiqiLi28" target="_blank" rel="noopener noreferrer">Ziqi Li</a></p>
      </footer>
    </div>
  );
}

export default Error;