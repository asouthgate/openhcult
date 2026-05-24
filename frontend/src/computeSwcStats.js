export function computeSwcStats(calibration, showFractional = false) {
  if (!calibration) return null
  const { mean, ci_low, ci_high } = calibration
  const estMin = Math.min(...mean)
  const estMax = Math.max(...mean)
  const lo = ci_low ? Math.min(...ci_low) : null
  const hi = ci_high ? Math.max(...ci_high) : null
  return { estMin, estMax, lo, hi }
}
