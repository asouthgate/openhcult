import { useState } from 'react'
import { setToken } from './api'

export default function Setup() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [error, setError] = useState(null)

  const submit = async e => {
    e.preventDefault()
    setError(null)
    if (password !== confirm) { setError('Passwords do not match'); return }
    const resp = await fetch('/auth/setup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    })
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}))
      setError(body.detail ?? 'Setup failed')
      return
    }
    const { access_token } = await resp.json()
    setToken(access_token)
    window.location.reload()
  }

  return (
    <div className="login">
      <h1>Hcult</h1>
      <p>Set up your account to get started.</p>
      <form onSubmit={submit}>
        <input type="text" placeholder="Username" value={username} onChange={e => setUsername(e.target.value)} required autoFocus />
        <input type="password" placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} required />
        <input type="password" placeholder="Confirm password" value={confirm} onChange={e => setConfirm(e.target.value)} required />
        {error && <p className="error">{error}</p>}
        <button type="submit">Create account</button>
      </form>
    </div>
  )
}
