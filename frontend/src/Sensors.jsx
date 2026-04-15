import { useState, useEffect } from 'react'
import { apiJson } from './api'
import { sensorKey, sensorPart } from './utils'
import SensorsPane from './SensorsPane'
import CalibrationPane from './CalibrationPane'

export default function Sensors() {
  const [view, setView] = useState('sensors')
  const [plantFilter, setPlantFilter] = useState('')
  const [plants, setPlants] = useState([])
  const [plantSensors, setPlantSensors] = useState([])
  const [sensorFilter, setSensorFilter] = useState('')
  const [calibration, setCalibration] = useState(null)
  const [calibError, setCalibError] = useState(null)
  const [calibLoading, setCalibLoading] = useState(false)
  const [calibParams, setCalibParams] = useState({
    offsetMin: '10', widthMin: '50', gpStdMl: '5', scalePriorMean: '', scalePriorStd: '',
  })

  const setCalibParam = (key, val) => setCalibParams(p => ({ ...p, [key]: val }))

  useEffect(() => {
    apiJson('/plants?limit=1000').then(d => setPlants(d.data ?? []))
    apiJson('/plant_sensors').then(d => setPlantSensors(d.data ?? []))
  }, [])

  const sensorsForPlant = plantFilter
    ? plantSensors.filter(ps => ps.plant_name === plantFilter).map(ps => ({
        key: sensorKey(ps.device_address, ps.sensor),
        label: `${ps.device_address} / ${ps.sensor}`,
        sensor: ps.sensor,
        assignedAt: ps.assigned_at ?? 0,
      }))
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
    const { offsetMin, widthMin, gpStdMl, scalePriorMean, scalePriorStd } = calibParams
    const params = new URLSearchParams({
      plant: plantFilter,
      offset_ms: Number(offsetMin) * 60 * 1000,
      width_ms: Number(widthMin) * 60 * 1000,
      gp_std_ml: Number(gpStdMl),
    })
    if (sensorFilter) params.set('sensor', sensorPart(sensorFilter))
    if (scalePriorMean !== '') params.set('scale_prior_mean', scalePriorMean)
    if (scalePriorStd !== '') params.set('scale_prior_std', scalePriorStd)
    apiJson(`/water_calibration?${params}`, { signal: controller.signal })
      .then(d => setCalibration(d))
      .catch(err => { if (err.name !== 'AbortError') { setCalibration(null); setCalibError(String(err)) } })
      .finally(() => setCalibLoading(false))
    return () => controller.abort()
  }, [plantFilter, sensorFilter, calibParams])

  return (
    <div className="app">
      <div className="app-header">
        <h1>HCult</h1>
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
              {sensorsForPlant.map(s => <option key={s.key} value={s.key}>{s.label}</option>)}
            </select>
          )}
        </div>
      </div>

      {view === 'sensors'
        ? <SensorsPane
            plantFilter={plantFilter} sensorFilter={sensorFilter} calibration={calibration}
            sensorAssignedAt={sensorsForPlant.find(s => s.key === sensorFilter)?.assignedAt ?? null}
            calibParams={calibParams} setCalibParam={setCalibParam} plantSensors={plantSensors}
          />
        : <CalibrationPane
            plantFilter={plantFilter} sensorFilter={sensorFilter}
            calibration={calibration} calibError={calibError} calibLoading={calibLoading}
            calibParams={calibParams} setCalibParam={setCalibParam}
          />
      }
    </div>
  )
}
