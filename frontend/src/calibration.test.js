import { describe, it, expect } from 'vitest'
import { computeSwcStats } from './computeSwcStats'

describe('computeSwcStats', () => {
  it('returns null for null calibration', () => {
    expect(computeSwcStats(null, false)).toBeNull()
  })

  it('computes stats with CI bounds', () => {
    const calib = {
      mean: [10, 20, 30],
      ci_low: [5, 15, 25],
      ci_high: [15, 25, 35],
      scale: 1,
    }
    const stats = computeSwcStats(calib, false)
    expect(stats.estMin).toBe(-20)
    expect(stats.estMax).toBe(0)
    expect(stats.lo).toBe(-25)
    expect(stats.hi).toBe(5)
  })

  it('converts to percentage when showPct is true', () => {
    const calib = {
      mean: [0, 50, 100],
      ci_low: null,
      ci_high: null,
      scale: 100,
    }
    const stats = computeSwcStats(calib, true)
    expect(stats.estMin).toBe(-100)
    expect(stats.estMax).toBe(0)
  })
})
