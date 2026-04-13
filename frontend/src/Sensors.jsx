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
  const [gpStdMl, setGpStdMl] = useState(5)
  const [scalePriorMean, setScalePriorMean] = useState('')
  const [scalePriorStd, setScalePriorStd] = useState('')

  useEffect(() => {
    apiFetch('/plants?limit=1000').then(r => r.json()).then(d => setPlants(d.data ?? []))
  }, [])

  useEffect(() => {
    if (!plantFilter) { setCalibration(null); setCalibError(null); return }
    const controller = new AbortController()
    setCalibLoading(true)
    setCalibError(null)
    const params = new URLSearchParams({
      plant: plantFilter,
      offset_ms: offsetMin * 60 * 1000,
      width_ms: widthMin * 60 * 1000,
      gp_std_ml: gpStdMl,
    })
    if (scalePriorMean !== '') params.set('scale_prior_mean', scalePriorMean)
    if (scalePriorStd !== '') params.set('scale_prior_std', scalePriorStd)
    apiFetch(`/water_calibration?${params}`, { signal: controller.signal })
      .then(r => r.ok ? r.json() : r.json().then(body => Promise.reject(body.detail ?? `HTTP ${r.status}`)))
      .then(d => setCalibration(d))
      .catch(err => { if (err.name !== 'AbortError') { setCalibration(null); setCalibError(String(err)) } })
      .finally(() => setCalibLoading(false))
    return () => controller.abort()
  }, [plantFilter, offsetMin, widthMin, gpStdMl, scalePriorMean, scalePriorStd])

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
          <div className="window-controls">
            <label>offset <input type="number" min="0" value={offsetMin} onChange={e => setOffsetMin(Number(e.target.value))} style={{ width: 52 }} /> min</label>
            <label>width <input type="number" min="1" value={widthMin} onChange={e => setWidthMin(Number(e.target.value))} style={{ width: 52 }} /> min</label>
            <label>GP std <input type="number" min="0.1" step="0.1" value={gpStdMl} onChange={e => setGpStdMl(Number(e.target.value))} style={{ width: 52 }} /> ml</label>
            <label>scale prior μ <input type="number" min="0" step="10" value={scalePriorMean} onChange={e => setScalePriorMean(e.target.value)} placeholder="off" style={{ width: 60 }} /> ml</label>
            <label>scale prior σ <input type="number" min="0" step="10" value={scalePriorStd} onChange={e => setScalePriorStd(e.target.value)} placeholder="off" style={{ width: 60 }} /> ml</label>
          </div>
        </div>
      </div>

      {view === 'sensors'
        ? <SensorsPane plantFilter={plantFilter} calibration={calibration}
            offsetMin={offsetMin} widthMin={widthMin} />
        : <CalibrationPane plantFilter={plantFilter} calibration={calibration} calibError={calibError} calibLoading={calibLoading} />
      }
    </div>
  )
}
