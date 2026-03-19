import { HashRouter, Routes, Route, Link } from 'react-router-dom'
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
  return (
    <HashRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/sensors" element={<Sensors />} />
      </Routes>
    </HashRouter>
  )
}
