import { useState, useEffect } from 'react'
import { apiJson, clearToken } from './api'
import { sensorKey } from './utils'
import SensorsPane from './SensorsPane'
import CalibrationPane from './CalibrationPane'
import { useCalibrationData } from './useCalibrationData'

const DEFAULT_CALIB_PARAMS = {
  offsetMin: '5', widthMin: '50', prior: 'calibrated', priorMin: '867', priorMax: '2009',
  estimator: 'exp_mcmc', priorWeight: '1.0', nBurn: '10', nSteps: '30',
  xminMu: '850', xminSigma: '75', xminHigh: '1100', emaTauMin: '60',
}

export default function Sensors() {
  const [view, setView] = useState('sensors')
  const [plantFilter, setPlantFilter] = useState('')
  const [plants, setPlants] = useState([])
  const [plantSensors, setPlantSensors] = useState([])
  const [sensorFilter, setSensorFilter] = useState('')
  const [calibParams, setCalibParams] = useState(DEFAULT_CALIB_PARAMS)
  const [rangeHours, setRangeHours] = useState(48)

  const setCalibParam = (key, val) => setCalibParams(p => ({ ...p, [key]: val }))

  const { calibration, calibError, calibLoading, dryingRate, combinedSwc } = useCalibrationData({
    plantFilter,
    sensorFilter,
    calibParams,
    rangeHours,
    setCalibParam,
  })

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

  const handleSensorChange = sensor => {
    setSensorFilter(sensor)
  }

  return (
    <div className="app">
      <div className="app-header">
        <div className="logo"><img src={`${import.meta.env.BASE_URL}teal-no-bg.png`} alt="HCult" /><h1>HCult</h1></div>
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
            <select value={sensorFilter} onChange={e => handleSensorChange(e.target.value)}>
              <option value="">All sensors</option>
              <option value="__combined__">Combined</option>
              {sensorsForPlant.map(s => <option key={s.key} value={s.key}>{s.label}</option>)}
            </select>
          )}
          <button className="logout-btn" onClick={() => { clearToken(); window.location.reload() }}>Logout</button>
        </div>
      </div>

      {view === 'sensors'
        ? <SensorsPane
            plantFilter={plantFilter} sensorFilter={sensorFilter} calibration={calibration}
            sensorAssignedAt={sensorsForPlant.find(s => s.key === sensorFilter)?.assignedAt ?? null}
            calibParams={calibParams} setCalibParam={setCalibParam} plantSensors={plantSensors}
            combinedSwc={combinedSwc} calibLoading={calibLoading} calibError={calibError}
            dryingRate={dryingRate}
            rangeHours={rangeHours} setRangeHours={setRangeHours}
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
