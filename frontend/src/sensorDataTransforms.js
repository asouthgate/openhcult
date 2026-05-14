import { interp } from './utils'

export function transformRateSeries(dryingRate) {
  if (!dryingRate?.times_ms?.length) return { mappedSeries: [], bands: [] }
  const { times_ms, rate_ml_per_day, rate_ml_per_day_ci_low, rate_ml_per_day_ci_high } = dryingRate
  const points = []
  const bandPoints = []
  for (let i = 0; i < times_ms.length; i++) {
    const v = rate_ml_per_day[i]
    if (v != null) {
      points.push({ t: times_ms[i], v, raw: v })
    }
    const lo = rate_ml_per_day_ci_low?.[i]
    const hi = rate_ml_per_day_ci_high?.[i]
    if (lo != null && hi != null) {
      bandPoints.push({ t: times_ms[i], lo, hi })
    }
  }
  const series = points.length ? [{ label: 'Drying rate', points, color: '#d0fffc' }] : []
  const bands = bandPoints.length ? [{ color: '#d0fffc', points: bandPoints }] : []
  return { mappedSeries: series, bands }
}

export function transformSeriesToWaterMode(series, calibration, measureMode) {
  const isWater = (measureMode === 'water' || measureMode === 'fractional') && calibration

  const ms = series.map(s => ({
    ...s,
    points: s.points.map(p => {
      let v = (measureMode === 'voltage' && p.mv != null) ? p.mv : p.raw
      if (isWater) {
        v = interp(p.raw, calibration.prior_x, calibration.mean)
      }
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
      lo: interp(p.raw, calibration.prior_x, loArr),
      hi: interp(p.raw, calibration.prior_x, hiArr),
    })),
  }))
  return { mappedSeries: ms, bands: bs }
}

export function transformCombinedSwc(combinedSwc, measureMode) {
  if (!combinedSwc || !combinedSwc.times_ms?.length) return { mappedSeries: [], bands: [], combinedReady: false }
  const isFractional = measureMode === 'fractional'
  const points = combinedSwc.times_ms.map((t, i) => {
    const v = combinedSwc.mean_swc[i]
    return { t, v: v != null ? v : null, raw: 0 }
  })
  const validPoints = points.filter(p => p.v != null)
  if (!validPoints.length) return { mappedSeries: [], bands: [], combinedReady: false }
  const swcSeries = { label: isFractional ? 'Combined fractional' : 'Combined SWC', points: validPoints, color: '#d0fffc' }
  const swcBands = [{
    color: '#d0fffc',
    points: combinedSwc.times_ms.map((t, i) => ({
      t,
      lo: combinedSwc.ci_low[i] != null ? combinedSwc.ci_low[i] : null,
      hi: combinedSwc.ci_high[i] != null ? combinedSwc.ci_high[i] : null,
    })).filter(p => p.lo != null && p.hi != null),
  }]
  return { mappedSeries: [swcSeries], bands: swcBands, combinedReady: true }
}

export function filterObservationsBySensorAssignment(observations, sensorAssignedAt) {
  return sensorAssignedAt != null
    ? observations.filter(o => new Date(o.observed_at).getTime() >= sensorAssignedAt)
    : observations
}
