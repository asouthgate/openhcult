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

function _interp(x, xs, ys) {
  if (x <= xs[0]) return ys[0]
  if (x >= xs[xs.length - 1]) return ys[xs.length - 1]
  let lo = 0, hi = xs.length - 1
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (xs[mid] <= x) lo = mid; else hi = mid
  }
  const t = (x - xs[lo]) / (xs[hi] - xs[lo])
  return ys[lo] + t * (ys[hi] - ys[lo])
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
  const [pendingMl, setPendingMl] = useState('')
  const [loading, setLoading] = useState(false)
  const [showDuplicateModal, setShowDuplicateModal] = useState(false)
  const [calibration, setCalibration] = useState(null)
  const [calibError, setCalibError] = useState(null)
  const [calibLoading, setCalibLoading] = useState(false)

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

  useEffect(() => {
    if (measureMode !== 'water' || !plantFilter) {
      setCalibration(null)
      setCalibError(null)
      return
    }
    setCalibLoading(true)
    setCalibError(null)
    fetch(`/water_calibration?plant=${encodeURIComponent(plantFilter)}`)
      .then(r => r.ok ? r.json() : r.json().then(body => Promise.reject(body.detail ?? `HTTP ${r.status}`)))
      .then(d => setCalibration(d))
      .catch(err => { setCalibration(null); setCalibError(String(err)) })
      .finally(() => setCalibLoading(false))
  }, [measureMode, plantFilter])

  const handleTimePick = t => {
    setPendingTime(t)
    setPendingPlant(plantFilter || '')
    setPendingMl('')
    setShowDuplicateModal(false)
  }

  const submitWatering = () => {
    const payload = {
      note: `WATER manual ml=${pendingMl}`,
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

  const deleteObservation = id => {
    fetch(`/observations/${id}`, { method: 'DELETE' })
      .then(r => r.ok && setObservations(prev => prev.filter(o => o.id !== id)))
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
            {[['raw', 'Raw'], ['voltage', 'mV'], ['water', 'Water']].map(([m, label]) => (
              <button key={m} className={measureMode === m ? 'active' : ''} onClick={() => setMeasureMode(m)}>
                {label}
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

      {measureMode === 'water' && !plantFilter && (
        <div className="empty">Select a plant to view calibrated water estimate.</div>
      )}

      {measureMode === 'water' && plantFilter && calibLoading && (
        <div className="loading">Computing calibration…</div>
      )}

      {measureMode === 'water' && plantFilter && !calibLoading && !calibration && (
        <div className="empty">{calibError ?? 'No calibration data.'}</div>
      )}

      {series.length > 0 && (measureMode !== 'water' || (plantFilter && calibration)) && (
        <TimeseriesChart
          series={series.map(s => ({
            ...s,
            points: s.points.map(p => {
              let v
              if (measureMode === 'water' && calibration) {
                v = _interp(p.raw, calibration.prior_x, calibration.mean)
              } else if (measureMode === 'voltage' && p.mv != null) {
                v = p.mv
              } else {
                v = p.raw
              }
              return { t: p.t, v }
            }),
          }))}
          observations={observations}
          rangeMs={rangeHours * 3600 * 1000}
          onTimePick={handleTimePick}
          pendingTime={pendingTime}
          yLabel={measureMode === 'water' ? 'ml' : measureMode === 'voltage' ? 'mV' : 'raw'}
        />
      )}

      {observations.length > 0 && (
        <table className="obs-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Plant</th>
              <th>Note</th>
              <th>ml</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {[...observations].reverse().map(o => (
              <tr key={o.id}>
                <td>{new Date(o.observed_at).toLocaleString()}</td>
                <td>{o.plant_name ?? '—'}</td>
                <td>{o.note}</td>
                <td>{o.volume_ml ?? '—'}</td>
                <td><button onClick={() => deleteObservation(o.id)}>Delete</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {pendingTime != null && (() => {
        const WARN_MS = 2 * 3600 * 1000
        const nearby = observations.find(o =>
          o.note?.includes('WATER') && !o.note?.includes('AUTO') &&
          (!pendingPlant ? !o.plant_name : o.plant_name === pendingPlant) &&
          Math.abs(o.observed_at - pendingTime) < WARN_MS
        )
        const mlOk = pendingMl !== '' && Number(pendingMl) > 0
        const handleRecord = () => {
          if (!mlOk) return
          nearby ? setShowDuplicateModal(true) : submitWatering()
        }
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
              <input
                type="number"
                min="1"
                placeholder="ml *"
                value={pendingMl}
                onChange={e => setPendingMl(e.target.value)}
                className="ml-input"
              />
              <button className="submit-btn" onClick={handleRecord} disabled={!mlOk}>Record</button>
              <button onClick={() => setPendingTime(null)}>Dismiss</button>
            </div>
          </>
        )
      })()}
    </div>
  )
}
