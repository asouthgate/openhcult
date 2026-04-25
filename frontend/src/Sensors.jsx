import { useState, useEffect, useRef } from 'react'
import { apiJson } from './api'
import { sensorKey, sensorPart } from './utils'
import { clearToken } from './api'
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
    offsetMin: '5', widthMin: '50', gpStdMl: '50', scalePriorMean: '', scalePriorStd: '', prior: 'power', priorMin: '867', priorMax: '2009', priorAlpha: '5.4523129367441685', estimator: 'exp_mcmc', priorWeight: '1.0', nBurn: '10', nSteps: '30', xminLow: '800', xminHigh: '1100',
  })

  const autoStdSet = useRef(false)
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
    autoStdSet.current = false
  }

  const handleSensorChange = sensor => {
    setSensorFilter(sensor)
    autoStdSet.current = false
  }

  useEffect(() => {
    if (!plantFilter) { setCalibration(null); setCalibError(null); return }
    const controller = new AbortController()
    setCalibLoading(true)
    setCalibError(null)
    const { offsetMin, widthMin, gpStdMl, scalePriorMean, scalePriorStd, prior, priorMin, priorMax, priorAlpha, estimator, priorWeight, nBurn, nSteps, xminLow, xminHigh } = calibParams
    const params = new URLSearchParams({
      plant: plantFilter,
      offset_ms: Number(offsetMin) * 60 * 1000,
      width_ms: Number(widthMin) * 60 * 1000,
    })
    if (sensorFilter) {
      const sep = sensorFilter.lastIndexOf(':')
      params.set('sensor', sensorFilter.slice(sep + 1))
      params.set('device_address', sensorFilter.slice(0, sep))
    }
    if (estimator !== 'gp') params.set('estimator', estimator)
    if (estimator === 'gp') {
      params.set('gp_std_ml', Number(gpStdMl))
      if (scalePriorMean !== '') params.set('scale_prior_mean', scalePriorMean)
      if (scalePriorStd !== '') params.set('scale_prior_std', scalePriorStd)
    }
    if (estimator === 'powerlaw' && priorWeight !== '') params.set('prior_weight', priorWeight)
    if (estimator === 'exponential' && priorWeight !== '') params.set('prior_weight', priorWeight)
    if (estimator === 'exp_mcmc') {
      if (priorWeight !== '') params.set('prior_weight', priorWeight)
      if (nBurn !== '') params.set('n_burn', nBurn)
      if (nSteps !== '') params.set('n_steps', nSteps)
      if (xminLow !== '') params.set('xmin_low', xminLow)
      if (xminHigh !== '') params.set('xmin_high', xminHigh)
    }
    if (prior !== 'calibrated') params.set('prior', prior)
    if (prior === 'linear' || prior === 'power') {
      if (priorMin !== '') params.set('prior_min', priorMin)
      if (priorMax !== '') params.set('prior_max', priorMax)
    }
    if (estimator === 'exponential' || estimator === 'exp_mcmc') {
      if (priorMin !== '' && !params.has('prior_min')) params.set('prior_min', priorMin)
      if (priorMax !== '' && !params.has('prior_max')) params.set('prior_max', priorMax)
    }
    if (prior === 'power' && priorAlpha !== '') params.set('prior_alpha', priorAlpha)
    apiJson(`/water_calibration?${params}`, { signal: controller.signal })
      .then(d => {
        setCalibration(d)
        if (!autoStdSet.current && d.chords_dy?.length > 0) {
          const mean = d.chords_dy.reduce((a, b) => a + b, 0) / d.chords_dy.length
          setCalibParam('gpStdMl', String(Math.round(mean)))
          if (d.chords_x?.length > 0) {
            const endpoints = d.chords_x.map((x, i) => x + (d.chords_dx?.[i] ?? 0))
            setCalibParam('priorMin', String(Math.round(Math.min(...d.chords_x, ...endpoints))))
          }
          autoStdSet.current = true
        }
      })
      .catch(err => { if (err.name !== 'AbortError') { setCalibration(null); setCalibError(String(err)) } })
      .finally(() => setCalibLoading(false))
    return () => controller.abort()
  }, [plantFilter, sensorFilter, calibParams])

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
