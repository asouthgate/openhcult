import { useState } from 'react'
import { setToken } from './api'

export default function Login() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState(null)

  const submit = async e => {
    e.preventDefault()
    setError(null)
    const resp = await fetch('/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    })
    if (!resp.ok) {
      setError('Invalid credentials')
      return
    }
    const { access_token } = await resp.json()
    setToken(access_token)
    window.location.reload()
  }

  return (
    <div className="login">
      <h1>Hcult</h1>
      <form onSubmit={submit}>
        <input type="text" placeholder="Username" value={username} onChange={e => setUsername(e.target.value)} required autoFocus />
        <input type="password" placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} required />
        {error && <p className="error">{error}</p>}
        <button type="submit">Login</button>
      </form>
    </div>
  )
}
