import { useState, useEffect } from 'react'
import { apiFetch } from './api'
import SensorsPane from './SensorsPane'
import CalibrationPane from './CalibrationPane'

export default function App() {
  const [view, setView] = useState('sensors')
  const [plantFilter, setPlantFilter] = useState('')
  const [plants, setPlants] = useState([])
  const [calibration, setCalibration] = useState(null)
  const [calibError, setCalibError] = useState(null)
  const [calibLoading, setCalibLoading] = useState(false)

  useEffect(() => {
    apiFetch('/plants?limit=1000').then(r => r.json()).then(d => setPlants(d.data ?? []))
  }, [])

  useEffect(() => {
    if (!plantFilter) { setCalibration(null); setCalibError(null); return }
    setCalibLoading(true)
    setCalibError(null)
    apiFetch(`/water_calibration?plant=${encodeURIComponent(plantFilter)}`)
      .then(r => r.ok ? r.json() : r.json().then(body => Promise.reject(body.detail ?? `HTTP ${r.status}`)))
      .then(d => setCalibration(d))
      .catch(err => { setCalibration(null); setCalibError(String(err)) })
      .finally(() => setCalibLoading(false))
  }, [plantFilter])

  return (
    <div className="app">
      <div className="app-header">
        <h1>Hcult</h1>
        <div className="controls">
          <div className="range-btns">
            <button className={view === 'sensors' ? 'active' : ''} onClick={() => setView('sensors')}>Sensors</button>
            <button className={view === 'calibration' ? 'active' : ''} onClick={() => setView('calibration')}>Calibration</button>
          </div>
          <select value={plantFilter} onChange={e => setPlantFilter(e.target.value)}>
            <option value="">All plants</option>
            {plants.map(p => <option key={p.plant_name} value={p.plant_name}>{p.plant_name}</option>)}
          </select>
        </div>
      </div>

      {view === 'sensors'
        ? <SensorsPane plantFilter={plantFilter} calibration={calibration} />
        : <CalibrationPane plantFilter={plantFilter} calibration={calibration} calibError={calibError} calibLoading={calibLoading} />
      }
    </div>
  )
}
