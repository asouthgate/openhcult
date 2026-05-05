import { useState, useEffect, useMemo } from 'react'
import { apiJson, apiFetch } from './api'
import { sensorKey, interp, toWater } from './utils'
import { TimeseriesChart } from './TimeseriesChart'
import { PALETTE, OBS_COLOR } from './theme'
import CalibrationParams from './CalibrationParams'

const TIME_RANGES = [
  { label: '6h', hours: 6 },
  { label: '24h', hours: 24 },
  { label: '7d', hours: 24 * 7 },
  { label: '30d', hours: 24 * 30 },
]

const Y_LABELS = { raw: 'Raw', voltage: 'Voltage (mV)', water: 'Water (ml)', water_pct: 'Water (%SC)', rate: 'Rate (ml/day)' }

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
  plantFilter, sensorFilter, sensorAssignedAt, calibration, calibParams, setCalibParam, plantSensors, combinedSwc, calibLoading, calibError, dryingRate, rangeHours, setRangeHours,
}) {
  const [measureMode, setMeasureMode] = useState('voltage')
  const [series, setSeries] = useState([])
  const [observations, setObservations] = useState([])
  const [pendingTime, setPendingTime] = useState(null)
  const [pendingPlant, setPendingPlant] = useState('')
  const [pendingMl, setPendingMl] = useState('')
  const [loading, setLoading] = useState(false)

  const isCombined = sensorFilter === '__combined__'

  useEffect(() => {
    const end = new Date()
    const start = new Date(end - rangeHours * 3600 * 1000)
    const params = new URLSearchParams({ start_utc: start.toISOString(), end_utc: end.toISOString(), limit: '50000' })
    if (plantFilter) params.set('plant', plantFilter)
    if (sensorFilter && !isCombined) {
      const sep = sensorFilter.lastIndexOf(':')
      params.set('sensor', sensorFilter.slice(sep + 1))
      params.set('device_address', sensorFilter.slice(0, sep))
    }

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
          if (sensorFilter && !isCombined && key !== sensorFilter) continue
          if (!grouped[key]) grouped[key] = { label: labelMap[key] ?? key, points: [] }
          grouped[key].points.push({ t: row.adjusted_time_ms, raw: row.measurement, mv: row.voltage_mv })
        }
        setSeries(Object.entries(grouped).map(([, s], i) => ({ ...s, color: PALETTE[i % PALETTE.length] })))
        setObservations((obs.data ?? []).filter(o => !plantFilter || o.plant_name === plantFilter))
      })
      .finally(() => setLoading(false))
  }, [rangeHours, plantFilter, sensorFilter, plantSensors])

  const isRateMode = measureMode === 'rate'
  const needsPlant = (measureMode === 'water' || measureMode === 'water_pct' || measureMode === 'rate') && !calibration && !combinedSwc && !dryingRate

  const rateSeries = useMemo(() => {
    if (!isRateMode || !dryingRate?.times_ms?.length) return null
    const { times_ms, rate_ml_per_day } = dryingRate
    const points = []
    for (let i = 0; i < times_ms.length; i++) {
      points.push({ t: times_ms[i], v: rate_ml_per_day[i], raw: rate_ml_per_day[i] })
    }
    return points.length ? [{ label: 'Drying rate', points, color: '#d0fffc' }] : null
  }, [isRateMode, dryingRate])

  const { mappedSeries, bands, combinedReady } = useMemo(() => {
    const isWaterMode = measureMode === 'water' || measureMode === 'water_pct'
    const isCombinedWater = isCombined && isWaterMode

    if (isWaterMode && !isCombined && !calibration) {
      return { mappedSeries: series, bands: [], combinedReady: true }
    }

    if (isCombinedWater) {
      if (!combinedSwc || !combinedSwc.times_ms?.length) return { mappedSeries: [], bands: [], combinedReady: false }
      const scale = combinedSwc.scale ?? 1
      const toV = ml => toWater(ml, scale, measureMode === 'water_pct')
      const points = combinedSwc.times_ms.map((t, i) => {
        const v = combinedSwc.mean_swc[i]
        return { t, v: v != null ? toV(v) : null, raw: 0 }
      })
      const validPoints = points.filter(p => p.v != null)
      if (!validPoints.length) return { mappedSeries: [], bands: [], combinedReady: false }
      const swcSeries = { label: 'Combined SWC', points: validPoints, color: '#d0fffc' }
      const swcBands = [{
        color: '#d0fffc',
        points: combinedSwc.times_ms.map((t, i) => ({
          t,
          lo: combinedSwc.ci_low[i] != null ? toV(combinedSwc.ci_low[i]) : null,
          hi: combinedSwc.ci_high[i] != null ? toV(combinedSwc.ci_high[i]) : null,
        })).filter(p => p.lo != null && p.hi != null),
      }]
      return { mappedSeries: [swcSeries], bands: swcBands, combinedReady: true }
    }

    const isWater = isWaterMode && calibration
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

    if (!isWater) return { mappedSeries: ms, bands: [], combinedReady: true }

    const loArr = calibration.ci_low ?? calibration.mean.map((m, i) => m - 2 * (calibration.std?.[i] ?? 0))
    const hiArr = calibration.ci_high ?? calibration.mean.map((m, i) => m + 2 * (calibration.std?.[i] ?? 0))
    const bs = ms.map(s => ({
      color: s.color,
      points: s.points.map(p => ({
        t: p.t,
        lo: toV(interp(p.raw, calibration.prior_x, loArr)),
        hi: toV(interp(p.raw, calibration.prior_x, hiArr)),
      })),
    }))
    return { mappedSeries: ms, bands: bs, combinedReady: true }
  }, [series, measureMode, calibration, isCombined, combinedSwc])

  const submitWatering = () => {
    const payload = { note: `WATER manual ml=${pendingMl}`, observed_at: new Date(pendingTime).toISOString() }
    if (pendingPlant) payload.plant_name = pendingPlant
    apiJson('/observations', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
      .then(created => { setObservations(prev => [...prev, created]); setPendingTime(null) })
  }

  const deleteObservation = id =>
    apiFetch(`/observations/${id}`, { method: 'DELETE' }).then(r => r.ok && setObservations(prev => prev.filter(o => o.id !== id)))

  let chartContent
  if (isRateMode) {
    if (!plantFilter) {
      chartContent = <div className="empty">Select a plant to view drying rate</div>
    } else if (calibLoading) {
      chartContent = <div className="loading"><span className="spinner" />Computing drying rate…</div>
    } else if (calibError) {
      chartContent = <div className="full error">{calibError}</div>
    } else if (!rateSeries) {
      chartContent = <div className="empty">No drying rate data available.</div>
    } else {
      const allT = rateSeries.flatMap(s => s.points.map(p => p.t))
      const rangeMs = allT.length > 1 ? Math.max(...allT) - Math.min(...allT) : rangeHours * 3600 * 1000
      chartContent = (
        <TimeseriesChart
          series={rateSeries}
          bands={[]}
          observations={[]}
          rangeMs={rangeMs}
          onTimePick={null}
          pendingTime={null}
          yLabel={Y_LABELS[measureMode]}
          hideObsLegend
        />
      )
    }
  } else if (loading) {
    chartContent = <div className="loading">Loading…</div>
  } else if (calibError && (measureMode === 'water' || measureMode === 'water_pct')) {
    chartContent = <div className="full error">{calibError}</div>
  } else if (needsPlant) {
    chartContent = <div className="empty">Select a plant to show water estimates</div>
  } else if (isCombined && (measureMode === 'water' || measureMode === 'water_pct')) {
    if (calibLoading) chartContent = <div className="loading"><span className="spinner" />Computing combined SWC…</div>
    else if (calibError) chartContent = <div className="full error">{calibError}</div>
    else if (!combinedReady) chartContent = <div className="empty">No combined SWC data available.</div>
    else chartContent = (
      <TimeseriesChart
        series={mappedSeries}
        bands={bands}
        observations={sensorAssignedAt != null ? observations.filter(o => new Date(o.observed_at).getTime() >= sensorAssignedAt) : observations}
        rangeMs={rangeHours * 3600 * 1000}
        onTimePick={t => { setPendingTime(t); setPendingPlant(plantFilter || ''); setPendingMl('') }}
        pendingTime={pendingTime}
        yLabel={Y_LABELS[measureMode]}
        eventWindowOffset={Number(calibParams.offsetMin) * 60 * 1000}
        eventWindowWidth={Number(calibParams.widthMin) * 60 * 1000}
      />
    )
  } else if (series.length === 0) {
    chartContent = <div className="empty">No data in range.</div>
  } else {
    chartContent = (
      <TimeseriesChart
        series={mappedSeries}
        bands={bands}
        observations={sensorAssignedAt != null ? observations.filter(o => new Date(o.observed_at).getTime() >= sensorAssignedAt) : observations}
        rangeMs={rangeHours * 3600 * 1000}
        onTimePick={t => { setPendingTime(t); setPendingPlant(plantFilter || ''); setPendingMl('') }}
        pendingTime={pendingTime}
        yLabel={Y_LABELS[measureMode]}
        eventWindowOffset={Number(calibParams.offsetMin) * 60 * 1000}
        eventWindowWidth={Number(calibParams.widthMin) * 60 * 1000}
      />
    )
  }

  return (
    <div className="pane-grid">
      <div className="full controls">
        <div className="range-btns">
          {TIME_RANGES.map(r => (
            <button key={r.hours} className={rangeHours === r.hours ? 'active' : ''} onClick={() => setRangeHours(r.hours)}>{r.label}</button>
          ))}
        </div>
        <div className="range-btns">
          {[['raw', 'Raw'], ['voltage', 'mV'], ['water', 'Water (ml)'], ['water_pct', 'Water (%SC)'], ['rate', 'Rate']].map(([m, label]) => (
            <button key={m} className={measureMode === m ? 'active' : ''} onClick={() => setMeasureMode(m)}>{label}</button>
          ))}
        </div>
      </div>

      <div className="full">
        {chartContent}
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