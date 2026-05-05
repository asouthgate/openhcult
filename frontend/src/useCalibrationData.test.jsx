import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { useCalibrationData } from './useCalibrationData'

const defaultCalibParams = {
  offsetMin: '5', widthMin: '50', prior: 'calibrated', priorMin: '867', priorMax: '2009',
  estimator: 'exp_mcmc', priorWeight: '1.0', nBurn: '10', nSteps: '30',
  xminMu: '850', xminSigma: '75', xminHigh: '1100', emaTauMin: '60',
}

const mockApiJson = vi.fn()

vi.mock('./api', () => ({
  apiJson: (...args) => mockApiJson(...args),
}))

vi.mock('./paramsBuilder', () => ({
  buildSwcTimeseriesParams: vi.fn(() => new URLSearchParams({ plant: 'plant1' })),
  buildDryingRateParams: vi.fn(() => new URLSearchParams()),
  buildWaterCalibrationParams: vi.fn((plant, sensor) => {
    const p = new URLSearchParams({ plant })
    if (sensor) p.set('sensor', sensor)
    return p
  }),
}))

describe('useCalibrationData error handling', () => {
  beforeEach(() => vi.clearAllMocks())

  it('sets calibError on water_calibration failure', async () => {
    mockApiJson.mockImplementation((url) =>
      url.includes('water_calibration')
        ? Promise.reject(new Error('Invalid xmin_mu value'))
        : Promise.resolve({})
    )

    const { result } = renderHook(() => useCalibrationData({
      plantFilter: 'plant1', sensorFilter: '',
      calibParams: defaultCalibParams, rangeHours: 48, setCalibParam: vi.fn(),
    }))

    await waitFor(() => expect(result.current.calibLoading).toBe(false))
    expect(result.current.calibError).toBe('Invalid xmin_mu value')
    expect(result.current.calibration).toBeNull()
  })

  it('clears combinedSwc on swc_timeseries error', async () => {
    mockApiJson.mockImplementation((url) =>
      url.includes('swc_timeseries')
        ? Promise.reject(new Error('Bad request'))
        : Promise.resolve({})
    )

    const { result } = renderHook(() => useCalibrationData({
      plantFilter: 'plant1', sensorFilter: '__combined__',
      calibParams: defaultCalibParams, rangeHours: 48, setCalibParam: vi.fn(),
    }))

    await waitFor(() => expect(result.current.calibLoading).toBe(false))
    expect(result.current.calibError).toBe('Bad request')
    expect(result.current.combinedSwc).toBeNull()
  })

  it('clears all state when plantFilter is cleared', async () => {
    mockApiJson.mockResolvedValue({ data: [] })

    const { result, rerender } = renderHook(
      ({ plantFilter }) => useCalibrationData({
        plantFilter, sensorFilter: '',
        calibParams: defaultCalibParams, rangeHours: 48, setCalibParam: vi.fn(),
      }),
      { initialProps: { plantFilter: 'plant1' } },
    )

    await waitFor(() => expect(result.current.calibLoading).toBe(false))

    mockApiJson.mockImplementation((url) =>
      url.includes('water_calibration')
        ? Promise.reject(new Error('Bad param'))
        : Promise.resolve({})
    )
    rerender({ plantFilter: 'plant2' })
    await waitFor(() => expect(result.current.calibLoading).toBe(false))
    expect(result.current.calibError).toBe('Bad param')

    mockApiJson.mockResolvedValue({ data: [] })
    rerender({ plantFilter: '' })
    expect(result.current.calibError).toBeNull()
    expect(result.current.calibration).toBeNull()
  })

  it('ignores AbortError', async () => {
    const abortError = new Error('Aborted')
    abortError.name = 'AbortError'
    mockApiJson.mockRejectedValue(abortError)

    const { result } = renderHook(() => useCalibrationData({
      plantFilter: 'plant1', sensorFilter: '',
      calibParams: defaultCalibParams, rangeHours: 48, setCalibParam: vi.fn(),
    }))

    await waitFor(() => expect(result.current.calibLoading).toBe(false))
    expect(result.current.calibError).toBeNull()
  })

  it('sets dryingRate to null on drying_rate failure without affecting calibration', async () => {
    mockApiJson.mockImplementation((url) => {
      if (url.includes('water_calibration')) return Promise.resolve({ chords_x: [], mean: [] })
      if (url.includes('drying_rate')) return Promise.reject(new Error('DR failed'))
      return Promise.resolve({})
    })

    const { result } = renderHook(() => useCalibrationData({
      plantFilter: 'plant1', sensorFilter: '',
      calibParams: defaultCalibParams, rangeHours: 48, setCalibParam: vi.fn(),
    }))

    await waitFor(() => expect(result.current.calibLoading).toBe(false))
    expect(result.current.dryingRate).toBeNull()
    expect(result.current.calibration).toEqual({ chords_x: [], mean: [] })
  })
})
