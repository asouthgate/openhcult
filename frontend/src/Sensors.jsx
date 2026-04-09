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
  const [offsetMin, setOffsetMin] = useState(10)
  const [widthMin, setWidthMin] = useState(50)

  useEffect(() => {
    apiFetch('/plants?limit=1000').then(r => r.json()).then(d => setPlants(d.data ?? []))
  }, [])

  useEffect(() => {
    if (!plantFilter) { setCalibration(null); setCalibError(null); return }
    setCalibLoading(true)
    setCalibError(null)
    const params = new URLSearchParams({
      plant: plantFilter,
      offset_ms: offsetMin * 60 * 1000,
      width_ms: widthMin * 60 * 1000,
    })
    apiFetch(`/water_calibration?${params}`)
      .then(r => r.ok ? r.json() : r.json().then(body => Promise.reject(body.detail ?? `HTTP ${r.status}`)))
      .then(d => setCalibration(d))
      .catch(err => { setCalibration(null); setCalibError(String(err)) })
      .finally(() => setCalibLoading(false))
  }, [plantFilter, offsetMin, widthMin])

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
        ? <SensorsPane plantFilter={plantFilter} calibration={calibration}
            offsetMin={offsetMin} widthMin={widthMin}
            onOffsetChange={setOffsetMin} onWidthChange={setWidthMin} />
        : <CalibrationPane plantFilter={plantFilter} calibration={calibration} calibError={calibError} calibLoading={calibLoading} />
      }
    </div>
  )
}
