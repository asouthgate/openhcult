import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { useSensorData } from './useSensorData'

const mockApiJson = vi.fn()

vi.mock('./api', () => ({
  apiJson: (...args) => mockApiJson(...args),
}))

describe('useSensorData', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('fetches timeseries and observations on mount', async () => {
    mockApiJson.mockResolvedValue({ data: [] })

    const { result } = renderHook(() => useSensorData({
      plantFilter: 'plant1', sensorFilter: '', rangeHours: 48,
    }))

    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(mockApiJson).toHaveBeenCalledTimes(2)
  })

  it('sets error on timeseries failure', async () => {
    mockApiJson.mockImplementation((url) => {
      if (url.includes('timeseries')) return Promise.reject(new Error('Network error'))
      return Promise.resolve({ data: [] })
    })

    const { result } = renderHook(() => useSensorData({
      plantFilter: 'plant1', sensorFilter: '', rangeHours: 48,
    }))

    await waitFor(() => expect(result.current.error).toBe('Network error'), { timeout: 2000 })
    expect(result.current.series).toEqual([])
  })

  it('does not set error on observations failure', async () => {
    mockApiJson.mockImplementation((url) => {
      if (url.includes('timeseries')) return Promise.resolve({ data: [] })
      if (url.includes('observations')) return Promise.reject(new Error('Obs failed'))
      return Promise.resolve({ data: [] })
    })

    const { result } = renderHook(() => useSensorData({
      plantFilter: 'plant1', sensorFilter: '', rangeHours: 48,
    }))

    await waitFor(() => expect(result.current.loading).toBe(false), { timeout: 2000 })
    expect(result.current.error).toBeNull()
  })

  const sensorData = {
    data: [
      { device_address: 'dev1', sensor: 'cap1', adjusted_time_ms: 1000, measurement: 500, voltage_mv: 250 },
      { device_address: 'dev1', sensor: 'cap1', adjusted_time_ms: 2000, measurement: 600, voltage_mv: 300 },
      { device_address: 'dev2', sensor: 'cap2', adjusted_time_ms: 1000, measurement: 450, voltage_mv: 225 },
      { device_address: 'dev2', sensor: 'cap2', adjusted_time_ms: 2000, measurement: 550, voltage_mv: 275 },
    ],
  }

  it('combined mode (sensorFilter="") fetches all sensors and returns all series', async () => {
    mockApiJson.mockImplementation((url) => {
      if (url.includes('timeseries')) return Promise.resolve(sensorData)
      return Promise.resolve({ data: [] })
    })

    const { result } = renderHook(() => useSensorData({
      plantFilter: 'plant1', sensorFilter: '', rangeHours: 48,
    }))

    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.series.length).toBe(2)

    const timeseriesUrl = mockApiJson.mock.calls.find(c => c[0].includes('timeseries'))[0]
    expect(timeseriesUrl).not.toContain('sensor=')
    expect(timeseriesUrl).not.toContain('device_address=')
  })

  it('all sensors mode (sensorFilter="_all_") fetches all data without sensor filter', async () => {
    mockApiJson.mockImplementation((url) => {
      if (url.includes('timeseries')) return Promise.resolve(sensorData)
      return Promise.resolve({ data: [] })
    })

    const { result } = renderHook(() => useSensorData({
      plantFilter: 'plant1', sensorFilter: '_all_', rangeHours: 48,
    }))

    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.series.length).toBe(2)

    const timeseriesUrl = mockApiJson.mock.calls.find(c => c[0].includes('timeseries'))[0]
    expect(timeseriesUrl).not.toContain('sensor=')
    expect(timeseriesUrl).not.toContain('device_address=')
  })

  it('specific sensor mode filters server-side and client-side', async () => {
    mockApiJson.mockImplementation((url) => {
      if (url.includes('timeseries')) return Promise.resolve(sensorData)
      return Promise.resolve({ data: [] })
    })

    const { result } = renderHook(() => useSensorData({
      plantFilter: 'plant1', sensorFilter: 'dev1:cap1', rangeHours: 48,
    }))

    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.series.length).toBe(1)
    expect(result.current.series[0].label).toContain('cap1')

    const timeseriesUrl = mockApiJson.mock.calls.find(c => c[0].includes('timeseries'))[0]
    expect(timeseriesUrl).toContain('sensor=cap1')
    expect(timeseriesUrl).toContain('device_address=dev1')
  })

  it('combined and all sensors produce identical API calls', async () => {
    mockApiJson.mockResolvedValue({ data: [] })

    const { result: result1 } = renderHook(() => useSensorData({
      plantFilter: 'plant1', sensorFilter: '', rangeHours: 48,
    }))
    await waitFor(() => expect(result1.current.loading).toBe(false))

    const call1 = mockApiJson.mock.calls.find(c => c[0].includes('timeseries'))[0][0]

    vi.clearAllMocks()
    mockApiJson.mockResolvedValue({ data: [] })

    const { result: result2 } = renderHook(() => useSensorData({
      plantFilter: 'plant1', sensorFilter: '_all_', rangeHours: 48,
    }))
    await waitFor(() => expect(result2.current.loading).toBe(false))

    const call2 = mockApiJson.mock.calls.find(c => c[0].includes('timeseries'))[0][0]

    const url1 = new URL(call1, 'http://test')
    const url2 = new URL(call2, 'http://test')

    expect(url1.searchParams.get('sensor')).toBeNull()
    expect(url2.searchParams.get('sensor')).toBeNull()
    expect(url1.searchParams.get('device_address')).toBeNull()
    expect(url2.searchParams.get('device_address')).toBeNull()
    expect(url1.pathname).toBe(url2.pathname)
  })
})