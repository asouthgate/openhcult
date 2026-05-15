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
