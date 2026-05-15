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

  it('does not fetch on mount', () => {
    renderHook(() => useCalibrator())
    expect(mockApiJson).not.toHaveBeenCalled()
  })

  it('fetches when calculate is called', async () => {
    mockApiJson.mockResolvedValue({ chords_x: [], mean: [], times_ms: [], mean_swc: [] })

    const { result } = renderHook(() => useCalibrator())

    await act(async () => {
      result.current.calculate(calcArgs)
    })

    expect(mockApiJson).toHaveBeenCalled()
  })

  it('sets calibError only on water_calibration failure, not swc_timeseries failure', async () => {
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

    await vi.waitFor(() => expect(result.current.calibLoading).toBe(false), { timeout: 2000 })
    expect(result.current.calibError).toBeNull()
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

  it('exposes params and setParam', () => {
    const { result } = renderHook(() => useCalibrator())
    expect(result.current.params.offsetMin).toBe('5')

    act(() => {
      result.current.setParam('offsetMin', '10')
    })

    expect(result.current.params.offsetMin).toBe('10')
  })
})
