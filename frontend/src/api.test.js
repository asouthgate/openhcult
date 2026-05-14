import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { getToken, setToken, clearToken, apiFetch, apiJson } from './api'

const TOKEN_KEY = 'hcult_token'

describe('apiFetch', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
    vi.stubGlobal('location', { reload: vi.fn() })
    localStorage.removeItem(TOKEN_KEY)
  })

  afterEach(() => vi.unstubAllGlobals())

  it('adds Authorization header when token exists', async () => {
    setToken('my-token')
    fetch.mockResolvedValue({ status: 200 })
    await apiFetch('/test')
    expect(fetch).toHaveBeenCalledWith('/test', {
      headers: { 'Authorization': 'Bearer my-token' },
    })
    clearToken()
  })

  it('clears token and reloads on 401', async () => {
    setToken('expired-token')
    fetch.mockResolvedValue({ status: 401 })
    await expect(apiFetch('/test')).rejects.toThrow('Unauthorized')
    expect(localStorage.getItem(TOKEN_KEY)).toBeNull()
    expect(location.reload).toHaveBeenCalled()
  })
})

describe('apiJson', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
    vi.stubGlobal('location', { reload: vi.fn() })
    localStorage.removeItem(TOKEN_KEY)
  })

  afterEach(() => vi.unstubAllGlobals())

  it('throws error with detail on failure', async () => {
    fetch.mockResolvedValue({
      ok: false,
      json: () => Promise.resolve({ detail: 'Something went wrong' }),
    })
    await expect(apiJson('/test')).rejects.toThrow('Something went wrong')
  })

  it('throws generic HTTP error when no detail', async () => {
    fetch.mockResolvedValue({ ok: false, status: 500, json: () => Promise.resolve({}) })
    await expect(apiJson('/test')).rejects.toThrow('HTTP 500')
  })

  it('handles malformed error response', async () => {
    fetch.mockResolvedValue({ ok: false, status: 500, json: () => Promise.reject(new Error('bad json')) })
    await expect(apiJson('/test')).rejects.toThrow('HTTP 500')
  })
})
