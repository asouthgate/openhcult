import { useState } from 'react'
import { setToken } from './api'

export default function AuthForm({ endpoint, submitLabel, withConfirm, subtitle }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [error, setError] = useState(null)

  const submit = async e => {
    e.preventDefault()
    setError(null)
    if (withConfirm && password !== confirm) { setError('Passwords do not match'); return }
    const resp = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    })
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}))
      setError(body.detail ?? 'Failed')
      return
    }
    const { access_token } = await resp.json()
    setToken(access_token)
    window.location.reload()
  }

  return (
    <div className="login">
      <div className="login-logo"><img src={`${import.meta.env.BASE_URL}teal-no-bg.png`} alt="HCult" /></div>
      {subtitle && <p>{subtitle}</p>}
      <form onSubmit={submit}>
        <input type="text" placeholder="Username" value={username} onChange={e => setUsername(e.target.value)} required autoFocus />
        <input type="password" placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} required />
        {withConfirm && <input type="password" placeholder="Confirm password" value={confirm} onChange={e => setConfirm(e.target.value)} required />}
        {error && <p className="error">{error}</p>}
        <button type="submit">{submitLabel}</button>
      </form>
    </div>
  )
}
