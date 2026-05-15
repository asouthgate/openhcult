import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useCalibrator } from './useCalibrator'

const defaultParams = {
  offsetMin: '5', widthMin: '50', prior: 'calibrated', priorMin: '867', priorMax: '2009',
  estimator: 'exp_mcmc', priorWeight: '1.0', nBurn: '10', nSteps: '30',
  systemCapacityMean: '', systemCapacityStd: '', emaTauMin: '60',
}

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

  it('fetches only when calculate is called', async () => {
    const { result } = renderHook(() => useCalibrator())

    await act(async () => {
      result.current.calculate(calcArgs)
    })

    expect(mockApiJson).toHaveBeenCalled()
  })

  it('sets calibError on water_calibration failure', async () => {
    mockApiJson.mockImplementation((url) => {
      if (url.includes('water_calibration')) return Promise.reject(new Error('Invalid system_capacity_mean value'))
      if (url.includes('drying_rate')) return Promise.resolve({ times_ms: [], rate_ml_per_day: [], valid: [], scale: 1 })
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

  it('clears combinedSwc on swc_timeseries error', async () => {
    mockApiJson.mockImplementation((url) => {
      if (url.includes('swc_timeseries')) return Promise.reject(new Error('Bad request'))
      if (url.includes('drying_rate')) return Promise.resolve({ times_ms: [], rate_ml_per_day: [], valid: [], scale: 1 })
      return Promise.resolve({})
    })

    const { result } = renderHook(() => useCalibrator())

    await act(async () => {
      result.current.calculate({ ...calcArgs, sensorFilter: '__combined__' })
    })

    await vi.waitFor(() => expect(result.current.calibLoading).toBe(false), { timeout: 2000 })
    expect(result.current.calibError).toBe('Bad request')
    expect(result.current.combinedSwc).toBeNull()
  })

  it('clears all state when plantFilter is empty', () => {
    const { result } = renderHook(() => useCalibrator())

    act(() => {
      result.current.calculate({ ...calcArgs, plantFilter: '' })
    })

    expect(result.current.calibError).toBeNull()
    expect(result.current.calibration).toBeNull()
    expect(result.current.combinedSwc).toBeNull()
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

  it('recalculates with updated params when calculate is called again', async () => {
    mockApiJson.mockResolvedValue({ chords_x: [], mean: [] })

    const { result } = renderHook(() => useCalibrator())

    await act(async () => {
      result.current.calculate(calcArgs)
    })

    await vi.waitFor(() => expect(result.current.calibLoading).toBe(false), { timeout: 2000 })
    expect(result.current.calibError).toBeNull()

    mockApiJson.mockImplementation((url) => {
      if (url.includes('water_calibration')) return Promise.reject(new Error('Bad system_capacity_mean'))
      if (url.includes('swc_timeseries')) return Promise.resolve({ times_ms: [], mean_swc: [] })
      if (url.includes('drying_rate')) return Promise.resolve({ times_ms: [], rate_ml_per_day: [], valid: [], scale: 1 })
      return Promise.resolve({})
    })

    await act(async () => {
      result.current.calculate(calcArgs)
    })

    await vi.waitFor(() => expect(result.current.calibError).toBe('Bad system_capacity_mean'), { timeout: 2000 })
  })

  it('passes returnFractional to the API calls', async () => {
    mockApiJson.mockResolvedValue({ chords_x: [], mean: [] })

    const { result } = renderHook(() => useCalibrator())

    await act(async () => {
      result.current.calculate({ ...calcArgs, returnFractional: true })
    })

    await vi.waitFor(() => expect(result.current.calibLoading).toBe(false), { timeout: 2000 })

    const waterCall = mockApiJson.mock.calls.find(c => c[0].includes('water_calibration'))
    expect(waterCall[0]).toContain('return_fractional=true')
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
