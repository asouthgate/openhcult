import { describe, it, expect } from 'vitest'
import { transformRateSeries } from './sensorDataTransforms'

describe('transformRateSeries', () => {
  it('transforms valid drying rate data', () => {
    const result = transformRateSeries({
      times_ms: [1000, 2000, 3000],
      rate_ml_per_day: [10, 20, 30],
      rate_ml_per_day_ci_low: [8, 18, 28],
      rate_ml_per_day_ci_high: [12, 22, 32],
    })
    expect(result.mappedSeries).toEqual([{
      label: 'Drying rate',
      points: [
        { t: 1000, v: 10, raw: 10 },
        { t: 2000, v: 20, raw: 20 },
        { t: 3000, v: 30, raw: 30 },
      ],
      color: '#d0fffc',
    }])
    expect(result.bands).toEqual([{
      color: '#d0fffc',
      points: [
        { t: 1000, lo: 8, hi: 12 },
        { t: 2000, lo: 18, hi: 22 },
        { t: 3000, lo: 28, hi: 32 },
      ],
    }])
  })
})
