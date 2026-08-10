import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [health, setHealth] = useState<string>('Loading...')

  useEffect(() => {
    fetch('/api/health')
      .then(res => res.json())
      .then(data => setHealth(data.status))
      .catch(() => setHealth('Error connecting to backend'))
  }, [])

  return (
    <>
      <h1>TrustGuard AI</h1>
      <div className="card">
        <p>Backend Status: {health}</p>
      </div>
    </>
  )
}

export default App
