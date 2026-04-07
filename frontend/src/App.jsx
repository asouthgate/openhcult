import { useState, useEffect } from 'react'
import { HashRouter, Routes, Route, Link } from 'react-router-dom'
import { getToken } from './api'
import Login from './Login'
import Setup from './Setup'
import Sensors from './Sensors'

function Home() {
  return (
    <div className="app">
      <div className="app-header"><h1>HCult</h1></div>
      <nav className="home-nav">
        <Link to="/sensors">Sensors</Link>
      </nav>
    </div>
  )
}

export default function App() {
  const [configured, setConfigured] = useState(null)

  useEffect(() => {
    fetch('/auth/status').then(r => r.json()).then(d => setConfigured(d.configured))
  }, [])

  if (configured === null) return null
  if (!configured) return <Setup />
  if (!getToken()) return <Login />
  return (
    <HashRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/sensors" element={<Sensors />} />
      </Routes>
    </HashRouter>
  )
}
