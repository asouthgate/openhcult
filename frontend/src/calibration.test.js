import { describe, it, expect } from 'vitest'
import { computeSwcStats } from './computeSwcStats'

describe('computeSwcStats', () => {
  it('returns null for null calibration', () => {
    expect(computeSwcStats(null)).toBeNull()
  })

  it('computes stats with CI bounds', () => {
    const calib = {
      mean: [10, 20, 30],
      ci_low: [5, 15, 25],
      ci_high: [15, 25, 35],
    }
    const stats = computeSwcStats(calib)
    expect(stats.estMin).toBe(10)
    expect(stats.estMax).toBe(30)
    expect(stats.lo).toBe(5)
    expect(stats.hi).toBe(35)
  })

  it('computes stats without CI bounds', () => {
    const calib = {
      mean: [0, 50, 100],
      ci_low: null,
      ci_high: null,
    }
    const stats = computeSwcStats(calib)
    expect(stats.estMin).toBe(0)
    expect(stats.estMax).toBe(100)
  })

  it('returns absolute values in fractional mode', () => {
    const calib = {
      mean: [0.1, 0.5, 1.0],
      ci_low: [0.05, 0.45, 0.95],
      ci_high: [0.15, 0.55, 1.05],
    }
    const stats = computeSwcStats(calib, true)
    expect(stats.estMin).toBe(0.1)
    expect(stats.estMax).toBe(1.0)
    expect(stats.lo).toBe(0.05)
    expect(stats.hi).toBe(1.05)
  })
})
