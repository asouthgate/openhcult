import { useState, useEffect } from 'react'
import { apiFetch } from './api'
import SensorsPane from './SensorsPane'
import CalibrationPane from './CalibrationPane'

export default function App() {
  const [view, setView] = useState('sensors')
  const [plantFilter, setPlantFilter] = useState('')
  const [plants, setPlants] = useState([])
  const [plantSensors, setPlantSensors] = useState([])
  const [sensorFilter, setSensorFilter] = useState('')
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
    apiFetch('/plant_sensors').then(r => r.json()).then(d => setPlantSensors(d.data ?? []))
  }, [])

  const sensorsForPlant = plantFilter
    ? plantSensors.filter(ps => ps.plant_name === plantFilter).map(ps => ps.sensor)
    : []

  function handlePlantChange(plant) {
    setPlantFilter(plant)
    setSensorFilter('')
  }

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
    if (sensorFilter) params.set('sensor', sensorFilter)
    if (scalePriorMean !== '') params.set('scale_prior_mean', scalePriorMean)
    if (scalePriorStd !== '') params.set('scale_prior_std', scalePriorStd)
    apiFetch(`/water_calibration?${params}`, { signal: controller.signal })
      .then(r => r.ok ? r.json() : r.json().then(body => Promise.reject(body.detail ?? `HTTP ${r.status}`)))
      .then(d => setCalibration(d))
      .catch(err => { if (err.name !== 'AbortError') { setCalibration(null); setCalibError(String(err)) } })
      .finally(() => setCalibLoading(false))
    return () => controller.abort()
  }, [plantFilter, sensorFilter, offsetMin, widthMin, gpStdMl, scalePriorMean, scalePriorStd])

  return (
    <div className="app">
      <div className="app-header">
        <h1>Hcult</h1>
        <div className="controls">
          <div className="range-btns">
            <button className={view === 'sensors' ? 'active' : ''} onClick={() => setView('sensors')}>Sensors</button>
            <button className={view === 'calibration' ? 'active' : ''} onClick={() => setView('calibration')}>Calibration</button>
          </div>
          <select value={plantFilter} onChange={e => handlePlantChange(e.target.value)}>
            <option value="">All plants</option>
            {plants.map(p => <option key={p.plant_name} value={p.plant_name}>{p.plant_name}</option>)}
          </select>
          {sensorsForPlant.length > 0 && (
            <select value={sensorFilter} onChange={e => setSensorFilter(e.target.value)}>
              <option value="">All sensors</option>
              {sensorsForPlant.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          )}
        </div>
      </div>

      {view === 'sensors'
        ? <SensorsPane
            plantFilter={plantFilter} sensorFilter={sensorFilter} calibration={calibration}
            offsetMin={offsetMin} setOffsetMin={setOffsetMin}
            widthMin={widthMin} setWidthMin={setWidthMin}
            gpStdMl={gpStdMl} setGpStdMl={setGpStdMl}
            scalePriorMean={scalePriorMean} setScalePriorMean={setScalePriorMean}
            scalePriorStd={scalePriorStd} setScalePriorStd={setScalePriorStd}
          />
        : <CalibrationPane
            plantFilter={plantFilter} sensorFilter={sensorFilter}
            calibration={calibration} calibError={calibError} calibLoading={calibLoading}
            offsetMin={offsetMin} setOffsetMin={setOffsetMin}
            widthMin={widthMin} setWidthMin={setWidthMin}
            gpStdMl={gpStdMl} setGpStdMl={setGpStdMl}
            scalePriorMean={scalePriorMean} setScalePriorMean={setScalePriorMean}
            scalePriorStd={scalePriorStd} setScalePriorStd={setScalePriorStd}
          />
      }
    </div>
  )
}
