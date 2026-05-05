import { toWater } from './utils'

export function computeSwcStats(calibration, showPct) {
  if (!calibration) return null
  const { mean, ci_low, ci_high, scale } = calibration
  const toV = v => toWater(v, scale, showPct)
  const ref = mean[mean.length - 1]
  const estMin = toV(Math.min(...mean) - ref)
  const estMax = toV(Math.max(...mean) - ref)
  const lo = ci_low ? toV(Math.min(...ci_low) - ref) : null
  const hi = ci_high ? toV(Math.max(...ci_high) - ref) : null
  return { estMin, estMax, lo, hi }
}
