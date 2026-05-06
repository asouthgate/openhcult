export function computeSwcStats(calibration, showFractional = false) {
  if (!calibration) return null
  const { mean, ci_low, ci_high } = calibration
  const ref = showFractional ? 0 : mean[mean.length - 1]
  const estMin = Math.min(...mean) - ref
  const estMax = Math.max(...mean) - ref
  const lo = ci_low ? Math.min(...ci_low) - ref : null
  const hi = ci_high ? Math.max(...ci_high) - ref : null
  return { estMin, estMax, lo, hi }
}
