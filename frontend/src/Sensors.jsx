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
  const [dryingRate, setDryingRate] = useState(null)
  const [combinedSwc, setCombinedSwc] = useState(null)
  const [calibParams, setCalibParams] = useState({
    offsetMin: '5', widthMin: '50', prior: 'calibrated', priorMin: '867', priorMax: '2009', estimator: 'exp_mcmc', priorWeight: '1.0', nBurn: '10', nSteps: '30', xminMu: '850', xminSigma: '75', xminHigh: '1100', emaTauMin: '60',
  })
  const [rangeHours, setRangeHours] = useState(48)

  const autoStdSet = useRef(false)
  const skipNextRefetch = useRef(false)
  const drControllerRef = useRef(null)
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

  const isCombined = sensorFilter === '__combined__'

  useEffect(() => {
    if (!plantFilter) { setCalibration(null); setCalibError(null); setDryingRate(null); setCombinedSwc(null); drControllerRef.current?.abort(); return }
    if (skipNextRefetch.current) { skipNextRefetch.current = false; return }
    drControllerRef.current?.abort()
    const controller = new AbortController()
    const drController = new AbortController()
    drControllerRef.current = drController
    setCalibLoading(true)
    setCalibError(null)

    if (isCombined) {
      const { offsetMin, widthMin, prior, priorMin, priorMax, priorWeight, nBurn, nSteps, xminMu, xminSigma, xminHigh, emaTauMin } = calibParams
      const endMs = Date.now()
      const params = new URLSearchParams({
        plant: plantFilter,
        offset_ms: Number(offsetMin) * 60 * 1000,
        width_ms: Number(widthMin) * 60 * 1000,
        start_ms: endMs - rangeHours * 3600 * 1000,
        end_ms: endMs,
      })
      if (priorMin !== '') params.set('prior_min', priorMin)
      if (priorMax !== '') params.set('prior_max', priorMax)
      if (prior !== 'calibrated') params.set('prior', prior)
      if (priorWeight !== '') params.set('prior_weight', priorWeight)
      if (nBurn !== '') params.set('n_burn', nBurn)
      if (nSteps !== '') params.set('n_steps', nSteps)
      if (xminMu !== '') params.set('xmin_mu', xminMu)
      if (xminSigma !== '') params.set('xmin_sigma', xminSigma)
      if (xminHigh !== '') params.set('xmin_high', xminHigh)
      setCalibration(null)
      apiJson(`/swc_timeseries?${params}`, { signal: controller.signal })
        .then(d => { setCombinedSwc(d); setCalibLoading(false) })
        .catch(err => { if (err.name !== 'AbortError') { console.error(err); setCalibError(err.message); setCalibLoading(false) } })
      const drParams = new URLSearchParams({
        plant: plantFilter,
        offset_ms: Number(offsetMin) * 60 * 1000,
        width_ms: Number(widthMin) * 60 * 1000,
        combined: 'true',
        start_utc: new Date(endMs - rangeHours * 3600 * 1000).toISOString(),
        end_utc: new Date(endMs).toISOString(),
      })
      if (priorMin !== '') drParams.set('prior_min', priorMin)
      if (priorMax !== '') drParams.set('prior_max', priorMax)
      if (prior !== 'calibrated') drParams.set('prior', prior)
      if (priorWeight !== '') drParams.set('prior_weight', priorWeight)
      if (nBurn !== '') drParams.set('n_burn', nBurn)
      if (nSteps !== '') drParams.set('n_steps', nSteps)
      if (xminMu !== '') drParams.set('xmin_mu', xminMu)
      if (xminSigma !== '') drParams.set('xmin_sigma', xminSigma)
      if (xminHigh !== '') drParams.set('xmin_high', xminHigh)
      if (emaTauMin !== '') drParams.set('ema_tau_min', emaTauMin)
      apiJson(`/drying_rate?${drParams}`, { signal: drController.signal }).then(dr => setDryingRate(dr)).catch(() => setDryingRate(null))
      return () => controller.abort()
    }

    setCombinedSwc(null)
    const { offsetMin, widthMin, prior, priorMin, priorMax, estimator, priorWeight, nBurn, nSteps, xminMu, xminSigma, xminHigh, emaTauMin } = calibParams
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
    if (estimator !== 'exp_mcmc') params.set('estimator', estimator)
    if (estimator === 'exponential' && priorWeight !== '') params.set('prior_weight', priorWeight)
    if (estimator === 'exp_mcmc') {
      if (priorWeight !== '') params.set('prior_weight', priorWeight)
      if (nBurn !== '') params.set('n_burn', nBurn)
      if (nSteps !== '') params.set('n_steps', nSteps)
      if (xminMu !== '') params.set('xmin_mu', xminMu)
      if (xminSigma !== '') params.set('xmin_sigma', xminSigma)
      if (xminHigh !== '') params.set('xmin_high', xminHigh)
    }
    if (prior !== 'calibrated') params.set('prior', prior)
    if (prior === 'linear') {
      if (priorMin !== '') params.set('prior_min', priorMin)
      if (priorMax !== '') params.set('prior_max', priorMax)
    }
    if (estimator === 'exponential' || estimator === 'exp_mcmc') {
      if (priorMin !== '' && !params.has('prior_min')) params.set('prior_min', priorMin)
      if (priorMax !== '' && !params.has('prior_max')) params.set('prior_max', priorMax)
    }
    apiJson(`/water_calibration?${params}`, { signal: controller.signal })
      .then(d => {
        setCalibration(d)
        if (!autoStdSet.current && d.chords_x?.length > 0) {
          const endpoints = d.chords_x.map((x, i) => x + (d.chords_dx?.[i] ?? 0))
          const newMin = String(Math.round(Math.min(...d.chords_x, ...endpoints)))
          if (newMin !== calibParams.priorMin) {
            skipNextRefetch.current = true
            setCalibParam('priorMin', newMin)
          }
          autoStdSet.current = true
        }
        const drEndMs = Date.now()
        const drParams = new URLSearchParams({ plant: plantFilter, offset_ms: Number(offsetMin) * 60 * 1000, width_ms: Number(widthMin) * 60 * 1000, start_utc: new Date(drEndMs - rangeHours * 3600 * 1000).toISOString(), end_utc: new Date(drEndMs).toISOString() })
        if (sensorFilter) { const sep = sensorFilter.lastIndexOf(':'); drParams.set('sensor', sensorFilter.slice(sep + 1)); drParams.set('device_address', sensorFilter.slice(0, sep)) }
        if (priorMin !== '') drParams.set('prior_min', priorMin)
        if (priorMax !== '') drParams.set('prior_max', priorMax)
        if (prior !== 'calibrated') drParams.set('prior', prior)
        if (estimator !== 'exp_mcmc') drParams.set('estimator', estimator)
        if (priorWeight !== '') drParams.set('prior_weight', priorWeight)
        if (estimator === 'exp_mcmc') { if (nBurn !== '') drParams.set('n_burn', nBurn); if (nSteps !== '') drParams.set('n_steps', nSteps); if (xminMu !== '') drParams.set('xmin_mu', xminMu); if (xminSigma !== '') drParams.set('xmin_sigma', xminSigma); if (xminHigh !== '') drParams.set('xmin_high', xminHigh) }
        if (emaTauMin !== '') drParams.set('ema_tau_min', emaTauMin)
        apiJson(`/drying_rate?${drParams}`, { signal: drController.signal }).then(dr => setDryingRate(dr)).catch(() => setDryingRate(null))
      })
      .catch(err => { if (err.name !== 'AbortError') { console.error(err); setCalibration(null); setCalibError(err.message) } })
      .finally(() => setCalibLoading(false))
    return () => controller.abort()
  }, [plantFilter, sensorFilter, calibParams, rangeHours])

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