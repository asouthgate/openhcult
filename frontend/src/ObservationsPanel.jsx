import { useState } from 'react'
import { apiJson, apiFetch } from './api'

function ObservationsTable({ observations, calibration, onDelete }) {
  if (!observations.length) return null

  const estMap = {}
  if (calibration?.chord_times) {
    calibration.chord_times.forEach((t, i) => { estMap[t] = calibration.estimated_chords_dx[i] })
  }

  return (
    <div className="table-container">
      <table className="obs-table">
        <thead>
          <tr><th>Time</th><th>Plant</th><th>Note</th><th>Dose (ml)</th><th>ΔmV (est.)</th><th></th></tr>
        </thead>
        <tbody>
          {[...observations].reverse().map(o => {
            const est = estMap[new Date(o.observed_at).getTime()]
            return (
              <tr key={o.id}>
                <td>{new Date(o.observed_at).toLocaleString()}</td>
                <td>{o.plant_name ?? '—'}</td>
                <td>{o.note}</td>
                <td>{o.volume_ml ?? '—'}</td>
                <td>{est != null ? est.toFixed(1) : '—'}</td>
                <td><button onClick={() => onDelete(o.id)}>Delete</button></td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

export default function ObservationsPanel({ observations, calibration }) {
  const [pendingTime, setPendingTime] = useState(null)
  const [pendingPlant, setPendingPlant] = useState('')
  const [pendingMl, setPendingMl] = useState('')

  const deleteObservation = id =>
    apiFetch(`/observations/${id}`, { method: 'DELETE' }).then(r => r.ok && setObservations(prev => prev.filter(o => o.id !== id)))

  const submitWatering = () => {
    const payload = { note: `WATER manual ml=${pendingMl}`, observed_at: new Date(pendingTime).toISOString() }
    if (pendingPlant) payload.plant_name = pendingPlant
    apiJson('/observations', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
      .then(created => { setObservations(prev => [...prev, created]); setPendingTime(null) })
  }

  return (
    <>
      <ObservationsTable
        observations={observations}
        calibration={calibration}
        onDelete={deleteObservation}
      />

      {pendingTime && (
        <div className="full event-panel">
          <span>Watering at {new Date(pendingTime).toLocaleString()}</span>
          <input type="number" placeholder="ml" value={pendingMl} onChange={e => setPendingMl(e.target.value)} />
          <button onClick={submitWatering} disabled={!pendingMl}>Record</button>
          <button onClick={() => setPendingTime(null)}>Cancel</button>
        </div>
      )}
    </>
  )
}

export function usePendingTime() {
  const [pendingTime, setPendingTime] = useState(null)
  const [pendingPlant, setPendingPlant] = useState('')
  const [pendingMl, setPendingMl] = useState('')

  const pickTime = (t, plantFilter) => {
    setPendingTime(t)
    setPendingPlant(plantFilter || '')
    setPendingMl('')
  }

  const cancel = () => {
    setPendingTime(null)
    setPendingPlant('')
    setPendingMl('')
  }

  return { pendingTime, pendingPlant, pendingMl, pickTime, cancel, setPendingMl }
}
