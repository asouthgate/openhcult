import { useState, useEffect, useRef } from 'react'
import { sensorPart } from './utils'
import CalibrationCurve from './CalibrationCurve'
import ScatterPlot from './ScatterPlot'
import CalibrationParams from './CalibrationParams'
import { computeSwcStats } from './computeSwcStats'

export default function CalibrationPane({ plantFilter, sensorFilter, calibrator }) {
  const [showFractional, setShowFractional] = useState(false)

  const showFractionalRef = useRef(showFractional)
  showFractionalRef.current = showFractional

  useEffect(() => {
    if (!plantFilter || sensorFilter === '__combined__') return
    calibrator.calculate({ plantFilter, sensorFilter, rangeHours: 48, returnFractional: showFractionalRef.current })
  }, [plantFilter, sensorFilter])

  const { calibration, calibError, calibLoading, params } = calibrator
  const hasSystemCapacity = params.systemCapacityMean !== '' && params.systemCapacityStd !== ''

  if (!plantFilter) return <div className="empty">Select a plant to view calibration.</div>

  if (sensorFilter === '__combined__') return <div className="empty">Combined view is available on the Sensors tab.</div>

  const { chords_dx, chords_dy, chord_times, scale, nlml, fractional: isFractional } = calibration ?? {}
  const swcStats = hasSystemCapacity ? computeSwcStats(calibration, showFractional) : null
  const unitLabel = isFractional ? '' : ' ml'

  const sensorLabel = sensorFilter === '__combined__' ? 'Combined' : (sensorFilter ? ` / ${sensorPart(sensorFilter)}` : '')

  const handleRecalculate = () => {
    calibrator.calculate({ plantFilter, sensorFilter, rangeHours: 48, returnFractional: showFractional })
  }

  return (
    <section className="pane-grid">
      <div className="full" style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <h3 style={{ margin: 0 }}>Calibration: {plantFilter}{sensorLabel}</h3>
        {!calibLoading && calibration && hasSystemCapacity && (
          <div className="range-btns">
            <button className={!showFractional ? 'active' : ''} onClick={() => setShowFractional(false)}>ml</button>
            <button className={showFractional ? 'active' : ''} onClick={() => setShowFractional(true)}>Fractional</button>
          </div>
        )}
        {nlml != null && (
          <span style={{ fontSize: 12, opacity: 0.7 }}>NLML: {nlml.toFixed(2)}</span>
        )}
      </div>
      {calibLoading && <div className="full loading"><span className="spinner" />Computing…</div>}
      {calibError && !calibration && <div className="full error">{calibError}</div>}
      {!hasSystemCapacity && calibration && (
        <div className="empty">Enter system capacity params and recalculate to view water calibration</div>
      )}
      {hasSystemCapacity && calibration && <>
        <div className="full">
          <CalibrationCurve calibration={calibration} showFractional={showFractional} />
        </div>
        {chord_times?.length > 0 && (() => {
          const dy = chords_dy
          const stats = [
            ['Events', chord_times.length],
            ['Earliest', new Date(Math.min(...chord_times)).toLocaleString()],
            ['Latest', new Date(Math.max(...chord_times)).toLocaleString()],
            ['Δsensor min', Math.min(...chords_dx).toFixed(1) + ' mV'],
            ['Δsensor max', Math.max(...chords_dx).toFixed(1) + ' mV'],
            ['Dose min', Math.min(...dy).toFixed(1) + unitLabel],
            ['Dose max', Math.max(...dy).toFixed(1) + unitLabel],
            ['Dose mean', (dy.reduce((a, b) => a + b, 0) / dy.length).toFixed(1) + unitLabel],
            ['SWC range', swcStats ? `${swcStats.estMin.toFixed(1)}–${swcStats.estMax.toFixed(1)}${unitLabel}` : '—'],
            ...(swcStats?.lo != null ? [['95% CI', `${swcStats.lo.toFixed(1)}–${swcStats.hi.toFixed(1)}${unitLabel}`]] : []),
          ]
          return (
            <table className="obs-table full">
              <tbody>
                {stats.map(([label, value]) => (
                  <tr key={label}><td style={{ opacity: 0.6 }}>{label}</td><td>{value}</td></tr>
                ))}
              </tbody>
            </table>
          )
        })()}
        {!chord_times?.length && calibration && (() => {
          const stats = [
            ['SWC range', swcStats ? `${swcStats.estMin.toFixed(1)}–${swcStats.estMax.toFixed(1)}${unitLabel}` : '—'],
            ...(swcStats?.lo != null ? [['95% CI', `${swcStats.lo.toFixed(1)}–${swcStats.hi.toFixed(1)}${unitLabel}`]] : []),
            ...(calibration.n_sensors ? [['Sensors combined', calibration.n_sensors]] : []),
          ]
          return (
            <table className="obs-table full">
              <tbody>
                {stats.map(([label, value]) => (
                  <tr key={label}><td style={{ opacity: 0.6 }}>{label}</td><td>{value}</td></tr>
                ))}
              </tbody>
            </table>
          )
        })()}
        {chord_times?.length > 0 && (
          <details className="full">
            <summary style={{ cursor: 'pointer', fontSize: 12, opacity: 0.7 }}>Chord events ({chord_times.length})</summary>
            <table className="obs-table" style={{ marginTop: 8 }}>
              <thead><tr><th>Watering time</th><th>Δsensor (mV)</th><th>Dose (ml)</th></tr></thead>
              <tbody>
                {chord_times.map((t, i) => (
                  <tr key={t}>
                    <td>{new Date(t).toLocaleString()}</td>
                    <td>{chords_dx[i]?.toFixed(1)}</td>
                    <td>{chords_dy[i]?.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        )}
      </>}
      {calibration && (
        <div>
          <ScatterPlot
            dx={chords_dx}
            dy={chords_dy}
            xLabel="Δsensor"
            yLabel={isFractional ? 'Δfractional' : 'Δml'}
          />
        </div>
      )}
      <div>
        <CalibrationParams calibParams={params} setCalibParam={calibrator.setParam} />
        <button className="recalc-btn" onClick={handleRecalculate} disabled={calibLoading || !plantFilter}>
          {calibLoading ? 'Computing…' : 'Recalculate'}
        </button>
      </div>
    </section>
  )
}
