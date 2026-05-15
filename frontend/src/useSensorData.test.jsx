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
})
