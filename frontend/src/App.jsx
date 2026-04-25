import { useState, useEffect } from 'react'
import { HashRouter, Routes, Route, Link } from 'react-router-dom'
import { getToken } from './api'
import AuthForm from './AuthForm'
import Sensors from './Sensors'

function Home() {
  return (
    <div className="app">
      <div className="app-header"><h1>HCult</h1><div className="logo"><img src={`${import.meta.env.BASE_URL}teal-no-bg.png`} alt="SWB" /><span>SWB</span></div></div>
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
  if (!configured) return <AuthForm endpoint="/auth/setup" submitLabel="Create account" withConfirm subtitle="Set up your account to get started." />
  if (!getToken()) return <AuthForm endpoint="/auth/login" submitLabel="Login" />
  return (
    <HashRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/sensors" element={<Sensors />} />
      </Routes>
    </HashRouter>
  )
}
