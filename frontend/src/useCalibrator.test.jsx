import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useCalibrator } from './useCalibrator'

const mockApiJson = vi.fn()

vi.mock('./api', () => ({
  apiJson: (...args) => mockApiJson(...args),
}))

vi.mock('./paramsBuilder', () => ({
  buildSwcTimeseriesParams: vi.fn(() => new URLSearchParams({ plant: 'plant1' })),
  buildDryingRateParams: vi.fn(() => new URLSearchParams()),
  buildWaterCalibrationParams: vi.fn((plant, sensor, params, returnFractional) => {
    const p = new URLSearchParams({ plant })
    if (sensor) p.set('sensor', sensor)
    if (returnFractional) p.set('return_fractional', 'true')
    return p
  }),
}))

const calcArgs = {
  plantFilter: 'plant1', sensorFilter: '', rangeHours: 48, returnFractional: false,
}

describe('useCalibrator', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockApiJson.mockResolvedValue({ chords_x: [], mean: [] })
  })

  it('sets calibError on swc_timeseries failure', async () => {
    mockApiJson.mockImplementation((url) => {
      if (url.includes('swc_timeseries')) return Promise.reject(new Error('swc failed'))
      if (url.includes('water_calibration')) return Promise.resolve({ chords_x: [], mean: [] })
      if (url.includes('drying_rate')) return Promise.resolve({ times_ms: [], rate_ml_per_day: [], valid: [], scale: 1 })
      return Promise.resolve({})
    })

    const { result } = renderHook(() => useCalibrator())

    await act(async () => {
      result.current.calculate(calcArgs)
    })

    await vi.waitFor(() => expect(result.current.swcLoading).toBe(false), { timeout: 2000 })
    expect(result.current.calibError).toBe('swc failed')
    expect(result.current.mappedSeries).toEqual([])
  })

  it('sets calibError on water_calibration failure', async () => {
    mockApiJson.mockImplementation((url) => {
      if (url.includes('water_calibration')) return Promise.reject(new Error('Invalid system_capacity_mean value'))
      if (url.includes('drying_rate')) return Promise.resolve({ times_ms: [], rate_ml_per_day: [], valid: [], scale: 1 })
      if (url.includes('swc_timeseries')) return Promise.resolve({ times_ms: [], mean_swc: [] })
      return Promise.resolve({})
    })

    const { result } = renderHook(() => useCalibrator())

    await act(async () => {
      result.current.calculate(calcArgs)
    })

    await vi.waitFor(() => expect(result.current.calibLoading).toBe(false), { timeout: 2000 })
    expect(result.current.calibError).toBe('Invalid system_capacity_mean value')
    expect(result.current.calibration).toBeNull()
  })

  it('clears all state when plantFilter is empty', () => {
    const { result } = renderHook(() => useCalibrator())

    act(() => {
      result.current.calculate({ ...calcArgs, plantFilter: '' })
    })

    expect(result.current.calibError).toBeNull()
    expect(result.current.calibration).toBeNull()
    expect(result.current.mappedSeries).toEqual([])
  })

  it('ignores AbortError', async () => {
    const abortError = new Error('Aborted')
    abortError.name = 'AbortError'
    mockApiJson.mockRejectedValue(abortError)

    const { result } = renderHook(() => useCalibrator())

    await act(async () => {
      result.current.calculate(calcArgs)
    })

    await vi.waitFor(() => expect(result.current.calibLoading).toBe(false), { timeout: 2000 })
    expect(result.current.calibError).toBeNull()
  })

  it('sets dryingRate to null on drying_rate failure without affecting calibration', async () => {
    mockApiJson.mockImplementation((url) => {
      if (url.includes('water_calibration')) return Promise.resolve({ chords_x: [], mean: [] })
      if (url.includes('drying_rate')) return Promise.reject(new Error('DR failed'))
      if (url.includes('swc_timeseries')) return Promise.resolve({ times_ms: [], mean_swc: [] })
      return Promise.resolve({})
    })

    const { result } = renderHook(() => useCalibrator())

    await act(async () => {
      result.current.calculate(calcArgs)
    })

    await vi.waitFor(() => expect(result.current.calibLoading).toBe(false), { timeout: 2000 })
    expect(result.current.dryingRate).toBeNull()
    expect(result.current.calibration).toEqual({ chords_x: [], mean: [] })
  })

  it('produces mappedSeries from swc_timeseries response', async () => {
    mockApiJson.mockImplementation((url) => {
      if (url.includes('water_calibration')) return Promise.resolve({ chords_x: [], mean: [] })
      if (url.includes('drying_rate')) return Promise.resolve({ times_ms: [], rate_ml_per_day: [], valid: [], scale: 1 })
      if (url.includes('swc_timeseries')) return Promise.resolve({
        times_ms: [1000, 2000, 3000],
        mean_swc: [10, 20, 30],
        ci_low: [8, 18, 28],
        ci_high: [12, 22, 32],
      })
      return Promise.resolve({})
    })

    const { result } = renderHook(() => useCalibrator())

    await act(async () => {
      result.current.calculate(calcArgs)
    })

    await vi.waitFor(() => expect(result.current.calibLoading).toBe(false), { timeout: 2000 })
    expect(result.current.mappedSeries.length).toBe(1)
    expect(result.current.mappedSeries[0].points.length).toBe(3)
    expect(result.current.mappedBands.length).toBe(1)
  })

  it('populates mappedSeries and dryingRate independently when water_calibration fails', async () => {
    mockApiJson.mockImplementation((url) => {
      if (url.includes('water_calibration')) return Promise.reject(new Error('Calibration failed'))
      if (url.includes('drying_rate')) return Promise.resolve({ times_ms: [100], rate_ml_per_day: [-5], valid: [true], scale: 1 })
      if (url.includes('swc_timeseries')) return Promise.resolve({
        times_ms: [1000, 2000, 3000],
        mean_swc: [10, 20, 30],
        ci_low: [8, 18, 28],
        ci_high: [12, 22, 32],
      })
      return Promise.resolve({})
    })

    const { result } = renderHook(() => useCalibrator())

    await act(async () => {
      result.current.calculate(calcArgs)
    })

    await vi.waitFor(() => expect(result.current.calibLoading).toBe(false), { timeout: 2000 })
    expect(result.current.calibError).toBe('Calibration failed')
    expect(result.current.calibration).toBeNull()
    expect(result.current.mappedSeries.length).toBe(1)
    expect(result.current.mappedSeries[0].points.length).toBe(3)
    expect(result.current.dryingRateLoading).toBe(false)
    expect(result.current.dryingRate).toEqual({ times_ms: [100], rate_ml_per_day: [-5], valid: [true], scale: 1 })
  })

  it('sets dryingRateLoading to false even when water_calibration fails', async () => {
    mockApiJson.mockImplementation((url) => {
      if (url.includes('water_calibration')) return Promise.reject(new Error('Calibration failed'))
      if (url.includes('drying_rate')) return Promise.resolve({ times_ms: [], rate_ml_per_day: [], valid: [], scale: 1 })
      if (url.includes('swc_timeseries')) return Promise.resolve({ times_ms: [], mean_swc: [] })
      return Promise.resolve({})
    })

    const { result } = renderHook(() => useCalibrator())

    await act(async () => {
      result.current.calculate(calcArgs)
    })

    await vi.waitFor(() => expect(result.current.dryingRateLoading).toBe(false), { timeout: 2000 })
    expect(result.current.calibError).toBe('Calibration failed')
    expect(result.current.dryingRate).toEqual({ times_ms: [], rate_ml_per_day: [], valid: [], scale: 1 })
  })

  it('sets swcLoading to true during calculate and false when swc_timeseries completes', async () => {
    let swcResolve
    const swcPromise = new Promise(resolve => { swcResolve = () => resolve({ times_ms: [], mean_swc: [] }) })
    mockApiJson.mockImplementation((url) => {
      if (url.includes('swc_timeseries')) return swcPromise
      if (url.includes('water_calibration')) return Promise.resolve({ chords_x: [], mean: [] })
      if (url.includes('drying_rate')) return Promise.resolve({ times_ms: [], rate_ml_per_day: [], valid: [], scale: 1 })
      return Promise.resolve({})
    })

    const { result } = renderHook(() => useCalibrator())

    await act(async () => {
      result.current.calculate(calcArgs)
    })

    expect(result.current.swcLoading).toBe(true)

    await act(async () => {
      swcResolve()
    })

    await vi.waitFor(() => expect(result.current.swcLoading).toBe(false), { timeout: 2000 })
  })
})
