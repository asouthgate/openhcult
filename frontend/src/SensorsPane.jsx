import { useState, useEffect, useMemo } from 'react'
import { apiJson, apiFetch } from './api'
import { sensorKey, interp, toWater } from './utils'
import { TimeseriesChart, PALETTE } from './TimeseriesChart'
import CalibrationParams from './CalibrationParams'

const TIME_RANGES = [
  { label: '6h', hours: 6 },
  { label: '24h', hours: 24 },
  { label: '7d', hours: 24 * 7 },
  { label: '30d', hours: 24 * 30 },
]

function ObservationsTable({ observations, sensorAssignedAt, calibration, onDelete }) {
  const visible = sensorAssignedAt != null
    ? observations.filter(o => new Date(o.observed_at).getTime() >= sensorAssignedAt)
    : observations
  if (!visible.length) return null

  const estMap = {}
  if (calibration?.chord_times) {
    calibration.chord_times.forEach((t, i) => { estMap[t] = calibration.estimated_chords_dx[i] })
  }

  return (
    <div className="table-container">
      <table className="obs-table">
        <thead>
          <tr><th>Time</th><th>Plant</th><th>Note</th><th>Dose (ml)</th><th>ΔmV (est.)</th><th></th></tr>
        </thead>
        <tbody>
          {[...visible].reverse().map(o => {
            const est = estMap[new Date(o.observed_at).getTime()]
            return (
              <tr key={o.id}>
                <td>{new Date(o.observed_at).toLocaleString()}</td>
                <td>{o.plant_name ?? '—'}</td>
                <td>{o.note}</td>
                <td>{o.volume_ml ?? '—'}</td>
                <td>{est != null ? est.toFixed(1) : '—'}</td>
                <td><button onClick={() => onDelete(o.id)}>Delete</button></td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

export default function SensorsPane({
  plantFilter, sensorFilter, sensorAssignedAt, calibration, calibParams, setCalibParam, plantSensors,
}) {
  const [rangeHours, setRangeHours] = useState(48)
  const [measureMode, setMeasureMode] = useState('voltage')
  const [series, setSeries] = useState([])
  const [observations, setObservations] = useState([])
  const [pendingTime, setPendingTime] = useState(null)
  const [pendingPlant, setPendingPlant] = useState('')
  const [pendingMl, setPendingMl] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    const end = new Date()
    const start = new Date(end - rangeHours * 3600 * 1000)
    const params = new URLSearchParams({ start_utc: start.toISOString(), end_utc: end.toISOString(), limit: '50000' })
    if (plantFilter) params.set('plant', plantFilter)

    setLoading(true)
    setPendingTime(null)

    const labelMap = {}
    for (const row of plantSensors) {
      labelMap[sensorKey(row.device_address, row.sensor)] = `${row.plant_name} / ${row.device_address} / ${row.sensor}`
    }

    Promise.all([
      apiJson(`/timeseries?${params}`),
      apiJson(`/observations?${new URLSearchParams({ start_utc: start.toISOString(), end_utc: end.toISOString(), limit: '10000' })}`),
    ])
      .then(([ts, obs]) => {
        const grouped = {}
        for (const row of ts.data ?? []) {
          const key = sensorKey(row.device_address, row.sensor)
          if (sensorFilter && key !== sensorFilter) continue
          if (!grouped[key]) grouped[key] = { label: labelMap[key] ?? key, points: [] }
          grouped[key].points.push({ t: row.adjusted_time_ms, raw: row.measurement, mv: row.voltage_mv })
        }
        setSeries(Object.entries(grouped).map(([, s], i) => ({ ...s, color: PALETTE[i % PALETTE.length] })))
        setObservations((obs.data ?? []).filter(o => !plantFilter || o.plant_name === plantFilter))
      })
      .finally(() => setLoading(false))
  }, [rangeHours, plantFilter, sensorFilter, plantSensors])

  const { mappedSeries, bands } = useMemo(() => {
    const isWater = (measureMode === 'water' || measureMode === 'water_pct') && calibration
    const scale = calibration?.scale ?? 1
    const toV = ml => toWater(ml, scale, measureMode === 'water_pct')

    const ms = series.map(s => ({
      ...s,
      points: s.points.map(p => {
        let v = (measureMode === 'voltage' && p.mv != null) ? p.mv : p.raw
        if (isWater) v = toV(interp(p.raw, calibration.prior_x, calibration.mean))
        return { t: p.t, v, raw: p.raw }
      }),
    }))

    if (!isWater) return { mappedSeries: ms, bands: [] }

    const loArr = calibration.mean.map((m, i) => Math.max(0, m - 2 * (calibration.std[i] || 0)))
    const hiArr = calibration.mean.map((m, i) => m + 2 * (calibration.std[i] || 0))
    const bs = ms.map(s => ({
      color: s.color,
      points: s.points.map(p => ({
        t: p.t,
        lo: toV(interp(p.raw, calibration.prior_x, loArr)),
        hi: toV(interp(p.raw, calibration.prior_x, hiArr)),
      })),
    }))
    return { mappedSeries: ms, bands: bs }
  }, [series, measureMode, calibration])

  const submitWatering = () => {
    const payload = { note: `WATER manual ml=${pendingMl}`, observed_at: new Date(pendingTime).toISOString() }
    if (pendingPlant) payload.plant_name = pendingPlant
    apiJson('/observations', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
      .then(created => { setObservations(prev => [...prev, created]); setPendingTime(null) })
  }

  const deleteObservation = id =>
    apiFetch(`/observations/${id}`, { method: 'DELETE' }).then(r => r.ok && setObservations(prev => prev.filter(o => o.id !== id)))

  return (
    <div className="pane-grid">
      <div className="full controls">
        <div className="range-btns">
          {TIME_RANGES.map(r => (
            <button key={r.hours} className={rangeHours === r.hours ? 'active' : ''} onClick={() => setRangeHours(r.hours)}>{r.label}</button>
          ))}
        </div>
        <div className="range-btns">
          {[['raw', 'Raw'], ['voltage', 'mV'], ['water', 'Water (ml)'], ['water_pct', 'Water (%FC)']].map(([m, label]) => (
            <button key={m} className={measureMode === m ? 'active' : ''} onClick={() => setMeasureMode(m)}>{label}</button>
          ))}
        </div>
      </div>

      <div className="full">
        {loading
          ? <div className="loading">Loading…</div>
          : series.length === 0
            ? <div className="empty">No data in range.</div>
            : <TimeseriesChart
                series={mappedSeries}
                bands={bands}
                observations={sensorAssignedAt != null ? observations.filter(o => new Date(o.observed_at).getTime() >= sensorAssignedAt) : observations}
                rangeMs={rangeHours * 3600 * 1000}
                onTimePick={t => { setPendingTime(t); setPendingPlant(plantFilter || ''); setPendingMl('') }}
                pendingTime={pendingTime}
                yLabel={measureMode}
                eventWindowOffset={Number(calibParams.offsetMin) * 60 * 1000}
                eventWindowWidth={Number(calibParams.widthMin) * 60 * 1000}
              />
        }
      </div>

      <div>
        <CalibrationParams calibParams={calibParams} setCalibParam={setCalibParam} />
      </div>

      <div className="full">
        <ObservationsTable
          observations={observations}
          sensorAssignedAt={sensorAssignedAt}
          calibration={calibration}
          onDelete={deleteObservation}
        />
      </div>

      {pendingTime && (
        <div className="full event-panel">
          <span>Watering at {new Date(pendingTime).toLocaleString()}</span>
          <input type="number" placeholder="ml" value={pendingMl} onChange={e => setPendingMl(e.target.value)} />
          <button onClick={submitWatering} disabled={!pendingMl}>Record</button>
          <button onClick={() => setPendingTime(null)}>Cancel</button>
        </div>
      )}
    </div>
  )
}
