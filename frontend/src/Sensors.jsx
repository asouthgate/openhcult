import { useState, useEffect } from 'react'
import { TimeseriesChart, PALETTE } from './TimeseriesChart'

const TIME_RANGES = [
  { label: '6h', hours: 6 },
  { label: '24h', hours: 24 },
  { label: '7d', hours: 24 * 7 },
  { label: '30d', hours: 24 * 30 },
]

function toUtc(d) {
  return d.toISOString()
}

export default function App() {
  const [rangeHours, setRangeHours] = useState(48)
  const [plantFilter, setPlantFilter] = useState('')
  const [measureMode, setMeasureMode] = useState('voltage')
  const [plants, setPlants] = useState([])
  const [series, setSeries] = useState([])
  const [observations, setObservations] = useState([])
  const [pendingTime, setPendingTime] = useState(null)
  const [pendingPlant, setPendingPlant] = useState('')
  const [loading, setLoading] = useState(false)
  const [showDuplicateModal, setShowDuplicateModal] = useState(false)

  useEffect(() => {
    fetch('/plants?limit=1000')
      .then(r => r.json())
      .then(d => setPlants(d.data ?? []))
  }, [])

  useEffect(() => {
    const end = new Date()
    const start = new Date(end - rangeHours * 3600 * 1000)
    const params = new URLSearchParams({
      start_utc: toUtc(start),
      end_utc: toUtc(end),
      limit: '50000',
    })
    if (plantFilter) params.set('plant', plantFilter)

    setLoading(true)
    setPendingTime(null)

    Promise.all([
      fetch(`/timeseries?${params}`).then(r => r.json()),
      fetch('/plant_sensors?limit=1000').then(r => r.json()),
      fetch(`/observations?${new URLSearchParams({ start_utc: toUtc(start), end_utc: toUtc(end), limit: '10000' })}`).then(r => r.json()),
    ])
      .then(([ts, ps, obs]) => {
        const labelMap = {}
        for (const row of ps.data ?? []) {
          labelMap[`${row.device_address}:${row.sensor}`] = `${row.plant_name} / ${row.device_address} / ${row.sensor}`
        }

        const grouped = {}
        for (const row of ts.data ?? []) {
          const key = `${row.device_address}:${row.sensor}`
          if (!grouped[key]) grouped[key] = { label: labelMap[key] ?? key, points: [] }
          grouped[key].points.push({ t: row.adjusted_time_ms, raw: row.measurement, mv: row.voltage_mv })
        }

        setSeries(
          Object.entries(grouped).map(([, s], i) => ({
            ...s,
            color: PALETTE[i % PALETTE.length],
          }))
        )
        setObservations(obs.data ?? [])
      })
      .finally(() => setLoading(false))
  }, [rangeHours, plantFilter])

  const handleTimePick = t => {
    setPendingTime(t)
    setPendingPlant(plantFilter || '')
    setShowDuplicateModal(false)
  }

  const submitWatering = () => {
    const payload = {
      note: 'WATER manual',
      observed_at: new Date(pendingTime).toISOString(),
    }
    if (pendingPlant) payload.plant_name = pendingPlant
    fetch('/observations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
      .then(r => r.json())
      .then(created => {
        setObservations(prev => [...prev, created])
        setPendingTime(null)
      })
  }

  return (
    <div className="app">
      <div className="app-header">
        <h1>Hcult</h1>
        <div className="controls">
          <div className="range-btns">
            {TIME_RANGES.map(r => (
              <button
                key={r.hours}
                className={rangeHours === r.hours ? 'active' : ''}
                onClick={() => setRangeHours(r.hours)}
              >
                {r.label}
              </button>
            ))}
          </div>
          <div className="range-btns">
            {['raw', 'voltage'].map(m => (
              <button key={m} className={measureMode === m ? 'active' : ''} onClick={() => setMeasureMode(m)}>
                {m === 'raw' ? 'Raw' : 'mV'}
              </button>
            ))}
          </div>
          <select value={plantFilter} onChange={e => setPlantFilter(e.target.value)}>
            <option value="">All plants</option>
            {plants.map(p => (
              <option key={p.plant_name} value={p.plant_name}>
                {p.plant_name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {loading && <div className="loading">Loading…</div>}

      {!loading && series.length === 0 && (
        <div className="empty">No sensor data in this time range.</div>
      )}

      {series.length > 0 && (
        <TimeseriesChart
          series={series.map(s => ({
            ...s,
            points: s.points.map(p => ({
              t: p.t,
              v: measureMode === 'voltage' && p.mv != null ? p.mv : p.raw,
            })),
          }))}
          observations={observations}
          rangeMs={rangeHours * 3600 * 1000}
          onTimePick={handleTimePick}
          pendingTime={pendingTime}
          yLabel={measureMode === 'voltage' ? 'mV' : 'raw'}
        />
      )}

      {pendingTime != null && (() => {
        const WARN_MS = 2 * 3600 * 1000
        const nearby = observations.find(o =>
          o.note?.includes('WATER') && !o.note?.includes('AUTO') &&
          (!pendingPlant ? !o.plant_name : o.plant_name === pendingPlant) &&
          Math.abs(o.observed_at - pendingTime) < WARN_MS
        )
        const handleRecord = () => nearby ? setShowDuplicateModal(true) : submitWatering()
        return (
          <>
            {showDuplicateModal && (
              <div className="modal-overlay">
                <div className="modal">
                  <p>A confirmed watering{nearby.plant_name ? <> for <strong>{nearby.plant_name}</strong></> : ''} already exists at <strong>{new Date(nearby.observed_at).toLocaleString()}</strong>.</p>
                  <p>Record another event anyway?</p>
                  <div className="modal-btns">
                    <button className="submit-btn" onClick={() => { setShowDuplicateModal(false); submitWatering() }}>Record anyway</button>
                    <button onClick={() => setShowDuplicateModal(false)}>Cancel</button>
                  </div>
                </div>
              </div>
            )}
            <div className="event-panel">
              <span>Watering at <strong>{new Date(pendingTime).toLocaleString()}</strong></span>
              {nearby && (
                <span className="event-warning">
                  ⚠ confirmed watering{nearby.plant_name ? ` for ${nearby.plant_name}` : ''} already at {new Date(nearby.observed_at).toLocaleString()}
                </span>
              )}
              <select value={pendingPlant} onChange={e => { setPendingPlant(e.target.value); setShowDuplicateModal(false) }}>
                <option value="">No plant</option>
                {plants.map(p => (
                  <option key={p.plant_name} value={p.plant_name}>{p.plant_name}</option>
                ))}
              </select>
              <button className="submit-btn" onClick={handleRecord}>Record</button>
              <button onClick={() => setPendingTime(null)}>Dismiss</button>
            </div>
          </>
        )
      })()}
    </div>
  )
}
