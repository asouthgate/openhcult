import { interp, toWater } from './utils'

export function transformRateSeries(dryingRate) {
  if (!dryingRate?.times_ms?.length) return null
  const { times_ms, rate_ml_per_day } = dryingRate
  const points = []
  for (let i = 0; i < times_ms.length; i++) {
    points.push({ t: times_ms[i], v: rate_ml_per_day[i], raw: rate_ml_per_day[i] })
  }
  return points.length ? [{ label: 'Drying rate', points, color: '#d0fffc' }] : null
}

export function transformSeriesToWaterMode(series, calibration, measureMode) {
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
  return { mappedSeries: ms, bands: bs }
}

export function transformCombinedSwc(combinedSwc, measureMode) {
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

export function filterObservationsBySensorAssignment(observations, sensorAssignedAt) {
  return sensorAssignedAt != null
    ? observations.filter(o => new Date(o.observed_at).getTime() >= sensorAssignedAt)
    : observations
}
