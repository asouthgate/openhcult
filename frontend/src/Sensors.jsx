import { useState, useEffect } from 'react'
import { apiJson, clearToken } from './api'
import { sensorKey } from './utils'
import SensorsPane from './SensorsPane'
import CalibrationPane from './CalibrationPane'
import { useCalibrationData } from './useCalibrationData'

const DEFAULT_CALIB_PARAMS = {
  offsetMin: '5', widthMin: '50', prior: 'calibrated', priorMin: '867', priorMax: '2009',
  estimator: 'exp_mcmc', priorWeight: '1.0', nBurn: '10', nSteps: '30', emaTauMin: '60',
  systemCapacityMean: '', systemCapacityStd: '',
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

  const mlData = useCalibrationData({
    plantFilter, sensorFilter, calibParams, rangeHours, setCalibParam, returnFractional: false,
  })
  const fracData = useCalibrationData({
    plantFilter, sensorFilter, calibParams, rangeHours, setCalibParam, returnFractional: true,
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
    if (plant) {
      const p = plants.find(p => p.plant_name === plant)
      if (p?.soil_volume != null) {
        setCalibParams(cp => ({ ...cp, systemCapacityMean: String(Math.round(p.soil_volume)) }))
      } else {
        setCalibParams(cp => ({ ...cp, systemCapacityMean: '' }))
      }
    } else {
      setCalibParams(cp => ({ ...cp, systemCapacityMean: '' }))
    }
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
            plantFilter={plantFilter} sensorFilter={sensorFilter}
            calibrationMl={mlData.calibration} calibrationFrac={fracData.calibration}
            sensorAssignedAt={sensorsForPlant.find(s => s.key === sensorFilter)?.assignedAt ?? null}
            calibParams={calibParams} setCalibParam={setCalibParam} plantSensors={plantSensors}
            combinedSwcMl={mlData.combinedSwc} combinedSwcFrac={fracData.combinedSwc}
            calibLoadingMl={mlData.calibLoading} calibLoadingFrac={fracData.calibLoading}
            calibErrorMl={mlData.calibError} calibErrorFrac={fracData.calibError}
            dryingRate={mlData.dryingRate}
            rangeHours={rangeHours} setRangeHours={setRangeHours}
            recalculateMl={mlData.recalculate} recalculateFrac={fracData.recalculate}
          />
        : <CalibrationPane
            plantFilter={plantFilter} sensorFilter={sensorFilter}
            calibrationMl={mlData.calibration} calibrationFrac={fracData.calibration}
            calibErrorMl={mlData.calibError} calibErrorFrac={fracData.calibError}
            calibLoadingMl={mlData.calibLoading} calibLoadingFrac={fracData.calibLoading}
            calibParams={calibParams} setCalibParam={setCalibParam}
            recalculateMl={mlData.recalculate} recalculateFrac={fracData.recalculate}
          />
      }
    </div>
  )
}
