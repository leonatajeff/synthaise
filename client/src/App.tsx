import { useState, useEffect } from 'react'
import './App.css'

function App() {

  const [data, setData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetch('https://api.example.com/data')
      .then((response) => response.json())
      .then((data) => setData(data));
  }, []);

  return (
    <div className="App">
      <h1> synthaise ai </h1>
      <div className="input-container">
        <input className="input" placeholder='Describe your sound....'></input>
         <button className="generate-btn"> Generate </button>
      </div>
      <div className="loading">
              <div className="lds-spinner"><div></div><div></div><div></div><div></div><div></div><div></div><div></div><div></div><div></div><div></div><div></div><div></div></div>
              <h2> Generating </h2>
         </div>
    </div>
  )
}

export default App
