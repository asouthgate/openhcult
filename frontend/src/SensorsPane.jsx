import { useState, useEffect, useMemo } from 'react'
import { TimeseriesChart, PALETTE } from './TimeseriesChart'

const TIME_RANGES = [
  { label: '6h', hours: 6 },
  { label: '24h', hours: 24 },
  { label: '7d', hours: 24 * 7 },
  { label: '30d', hours: 24 * 30 },
]

function _toUtc(d) { return d.toISOString() }

function _interp(x, xs, ys) {
  if (!xs || !ys || xs.length === 0) return 0
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

export default function SensorsPane({ plantFilter, calibration }) {
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
    const params = new URLSearchParams({ start_utc: _toUtc(start), end_utc: _toUtc(end), limit: '50000' })
    if (plantFilter) params.set('plant', plantFilter)

    setLoading(true)
    setPendingTime(null)

    Promise.all([
      fetch(`/timeseries?${params}`).then(r => r.json()),
      fetch('/plant_sensors?limit=1000').then(r => r.json()),
      fetch(`/observations?${new URLSearchParams({ start_utc: _toUtc(start), end_utc: _toUtc(end), limit: '10000' })}`).then(r => r.json()),
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
        setSeries(Object.entries(grouped).map(([, s], i) => ({ ...s, color: PALETTE[i % PALETTE.length] })))
        setObservations((obs.data ?? []).filter(o => !plantFilter || o.plant_name === plantFilter))
      })
      .finally(() => setLoading(false))
  }, [rangeHours, plantFilter])

  const { mappedSeries, bands } = useMemo(() => {
    const isWater = (measureMode === 'water' || measureMode === 'water_pct') && calibration
    const scale = calibration?.scale ?? 1
    const toV = ml => measureMode === 'water_pct' ? (ml / scale) * 100 : ml

    const ms = series.map(s => ({
      ...s,
      points: s.points.map(p => {
        let v = (measureMode === 'voltage' && p.mv != null) ? p.mv : p.raw
        if (isWater) v = toV(_interp(p.raw, calibration.prior_x, calibration.mean))
        return { t: p.t, v, raw: p.raw }
      }),
    }))

    const bs = isWater ? ms.map(s => ({
      color: s.color,
      points: s.points.map(p => ({
        t: p.t,
        lo: toV(_interp(p.raw, calibration.prior_x, calibration.mean.map((m, i) => Math.max(0, m - 2 * (calibration.std[i] || 0))))),
        hi: toV(_interp(p.raw, calibration.prior_x, calibration.mean.map((m, i) => m + 2 * (calibration.std[i] || 0))))
      }))
    })) : []
    return { mappedSeries: ms, bands: bs }
  }, [series, measureMode, calibration])

  const submitWatering = () => {
    const payload = { note: `WATER manual ml=${pendingMl}`, observed_at: new Date(pendingTime).toISOString() }
    if (pendingPlant) payload.plant_name = pendingPlant
    fetch('/observations', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
      .then(r => r.json()).then(created => { setObservations(prev => [...prev, created]); setPendingTime(null) })
  }

  return (
    <div>
      <div className="controls">
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

      <section className="sensor-pane">
        {loading
          ? <div className="loading">Loading…</div>
          : series.length === 0
            ? <div className="empty">No data in range.</div>
            : <TimeseriesChart
                series={mappedSeries}
                bands={bands}
                observations={observations}
                rangeMs={rangeHours * 3600 * 1000}
                onTimePick={t => { setPendingTime(t); setPendingPlant(plantFilter || ''); setPendingMl('') }}
                pendingTime={pendingTime}
                yLabel={measureMode}
              />
        }
      </section>

      {observations.length > 0 && (
        <div className="table-container">
          <table className="obs-table">
            <thead>
              <tr><th>Time</th><th>Plant</th><th>Note</th><th>Dose (ml)</th><th></th></tr>
            </thead>
            <tbody>
              {[...observations].reverse().map(o => (
                <tr key={o.id}>
                  <td>{new Date(o.observed_at).toLocaleString()}</td>
                  <td>{o.plant_name ?? '—'}</td>
                  <td>{o.note}</td>
                  <td>{o.volume_ml ?? '—'}</td>
                  <td><button onClick={() => {
                    fetch(`/observations/${o.id}`, { method: 'DELETE' }).then(r => r.ok && setObservations(prev => prev.filter(obs => obs.id !== o.id)))
                  }}>Delete</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {pendingTime && (
        <div className="event-panel">
          <span>Watering at {new Date(pendingTime).toLocaleString()}</span>
          <input type="number" placeholder="ml" value={pendingMl} onChange={e => setPendingMl(e.target.value)} />
          <button onClick={submitWatering} disabled={!pendingMl}>Record</button>
          <button onClick={() => setPendingTime(null)}>Cancel</button>
        </div>
      )}
    </div>
  )
}
