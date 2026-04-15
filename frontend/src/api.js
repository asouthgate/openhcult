const TOKEN_KEY = 'hcult_token'

export function getToken() {
  return localStorage.getItem(TOKEN_KEY)
}

export function setToken(token) {
  localStorage.setItem(TOKEN_KEY, token)
}

export function clearToken() {
  localStorage.removeItem(TOKEN_KEY)
}

export async function apiFetch(url, options = {}) {
  const token = getToken()
  const headers = { ...options.headers }
  if (token) headers['Authorization'] = `Bearer ${token}`
  const resp = await fetch(url, { ...options, headers })
  if (resp.status === 401) {
    clearToken()
    window.location.reload()
    throw new Error('Unauthorized')
  }
  return resp
}

export async function apiJson(url, options = {}) {
  const resp = await apiFetch(url, options)
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}))
    throw new Error(body.detail ?? `HTTP ${resp.status}`)
  }
  return resp.json()
}
