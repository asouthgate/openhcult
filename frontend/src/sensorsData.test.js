import { describe, it, expect } from 'vitest'
import { interp } from './utils'
import {
  transformRateSeries,
  transformSeriesToWaterMode,
  transformCombinedSwc,
  filterObservationsBySensorAssignment,
} from './sensorDataTransforms'

describe('transformRateSeries', () => {
  it('transforms valid drying rate data', () => {
    const result = transformRateSeries({
      times_ms: [1000, 2000, 3000],
      rate_ml_per_day: [10, 20, 30],
    })
    expect(result).toEqual([{
      label: 'Drying rate',
      points: [
        { t: 1000, v: 10, raw: 10 },
        { t: 2000, v: 20, raw: 20 },
        { t: 3000, v: 30, raw: 30 },
      ],
      color: '#d0fffc',
    }])
  })
})

describe('transformSeriesToWaterMode', () => {
  const baseSeries = [{
    label: 'sensor1', color: '#d0fffc',
    points: [
      { t: 1000, raw: 500, mv: 1500 },
      { t: 2000, raw: 600, mv: 1600 },
    ],
  }]

  it('passes through raw values without calibration', () => {
    const { mappedSeries, bands } = transformSeriesToWaterMode(baseSeries, null, 'raw')
    expect(mappedSeries[0].points[0].v).toBe(500)
    expect(bands).toEqual([])
  })

  it('uses voltage when available', () => {
    const { mappedSeries } = transformSeriesToWaterMode(baseSeries, null, 'voltage')
    expect(mappedSeries[0].points[0].v).toBe(1500)
  })

  it('interpolates to water values with calibration', () => {
    const calibration = {
      prior_x: [400, 500, 600, 700],
      mean: [0, 10, 20, 30],
      scale: 1,
    }
    const { mappedSeries, bands } = transformSeriesToWaterMode(baseSeries, calibration, 'water')
    expect(mappedSeries[0].points[0].v).toBe(10)
    expect(bands.length).toBe(1)
  })

  it('computes confidence bands from std when ci not provided', () => {
    const calibration = {
      prior_x: [400, 500, 600, 700],
      mean: [0, 10, 20, 30],
      std: [1, 2, 3, 4],
      scale: 1,
    }
    const { bands } = transformSeriesToWaterMode(baseSeries, calibration, 'water')
    expect(bands[0].points[0].lo).toBeDefined()
    expect(bands[0].points[0].hi).toBeDefined()
  })
})

describe('transformCombinedSwc', () => {
  it('transforms valid combined SWC with bands', () => {
    const result = transformCombinedSwc({
      times_ms: [1000, 2000, 3000],
      mean_swc: [10, 20, 30],
      ci_low: [8, 18, 28],
      ci_high: [12, 22, 32],
      scale: 1,
    }, 'water')
    expect(result.combinedReady).toBe(true)
    expect(result.mappedSeries[0].points.length).toBe(3)
    expect(result.bands[0].points.length).toBe(3)
  })

  it('filters out null mean_swc and incomplete band points', () => {
    const result = transformCombinedSwc({
      times_ms: [1000, 2000, 3000],
      mean_swc: [10, null, 30],
      ci_low: [8, null, 28],
      ci_high: [12, 22, 32],
      scale: 1,
    }, 'water')
    expect(result.mappedSeries[0].points.length).toBe(2)
    expect(result.bands[0].points.length).toBe(2)
  })
})

describe('filterObservationsBySensorAssignment', () => {
  const observations = [
    { id: 1, observed_at: '2024-01-01T00:00:00Z' },
    { id: 2, observed_at: '2024-01-02T00:00:00Z' },
    { id: 3, observed_at: '2024-01-03T00:00:00Z' },
  ]

  it('filters observations before sensor assignment', () => {
    const assignedAt = new Date('2024-01-02T00:00:00Z').getTime()
    const result = filterObservationsBySensorAssignment(observations, assignedAt)
    expect(result.length).toBe(2)
    expect(result[0].id).toBe(2)
  })
})
