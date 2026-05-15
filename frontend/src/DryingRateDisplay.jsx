import { useState, useMemo } from 'react'
import { TimeseriesChart } from './TimeseriesChart'
import { transformRateSeries } from './sensorDataTransforms'

export default function DryingRateDisplay({ dryingRate, dryingRateLoading, plantFilter, rangeHours }) {
  const [logScale, setLogScale] = useState(false)

  const { mappedSeries, bands } = useMemo(() => transformRateSeries(dryingRate), [dryingRate])

  const currentRate = useMemo(() => {
    if (!mappedSeries.length) return null
    const points = mappedSeries[0].points
    if (!points.length) return null
    const now = Math.max(...points.map(p => p.t))
    const windowMs = 30 * 60 * 1000
    const recent = points.filter(p => p.t > now - windowMs)
    const validRecent = dryingRate?.valid
      ? recent.filter((p, i) => {
          const idx = points.indexOf(p)
          return dryingRate.valid[idx]
        })
      : recent
    const values = validRecent.length > 0 ? validRecent.map(p => p.v) : points.map(p => p.v)
    return values.sort((a, b) => a - b)[Math.floor(values.length / 2)]
  }, [mappedSeries, dryingRate])

  if (!plantFilter) {
    return <div className="empty">Select a plant to view drying rate</div>
  }

  if (dryingRateLoading) {
    return <div className="loading"><span className="spinner" />Computing drying rate…</div>
  }

  if (!dryingRate) {
    return <div className="full error">Drying rate computation failed. Check calibration and try again.</div>
  }

  if (!mappedSeries.length) {
    return <div className="empty">No drying rate data available.</div>
  }

  const allT = mappedSeries.flatMap(s => s.points.map(p => p.t))
  const rangeMs = allT.length > 1 ? Math.max(...allT) - Math.min(...allT) : rangeHours * 3600 * 1000

  return (
    <div>
      {currentRate != null && (
        <div className="current-rate-display">
          <span className={`current-rate-value ${currentRate < 0 ? 'drying' : 'watering'}`}>
            {currentRate.toFixed(2)}
          </span>
          <span className="current-rate-unit">ml/day</span>
          <span className="current-rate-label">{currentRate < 0 ? 'Drying' : 'Watering'}</span>
        </div>
      )}
      <div className="rate-controls">
        <button className={logScale ? 'active' : ''} onClick={() => setLogScale(v => !v)}>Log</button>
      </div>
      <TimeseriesChart
        series={mappedSeries}
        bands={bands}
        observations={[]}
        rangeMs={rangeMs}
        onTimePick={null}
        pendingTime={null}
        yLabel="Rate (ml/day)"
        hideObsLegend
        logScale={logScale}
      />
    </div>
  )
}
