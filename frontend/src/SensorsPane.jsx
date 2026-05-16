import { useState, useMemo, useEffect, useRef } from 'react'
import { useSensorData } from './useSensorData'
import ChartDisplay from './ChartDisplay'
import DryingRateDisplay from './DryingRateDisplay'
import ObservationsPanel, { usePendingTime } from './ObservationsPanel'
import CalibrationControls from './CalibrationControls'

const TIME_RANGES = [
  { label: '6h', hours: 6 },
  { label: '24h', hours: 24 },
  { label: '7d', hours: 24 * 7 },
  { label: '30d', hours: 24 * 30 },
]

const Y_LABELS = { raw: 'Raw', voltage: 'Voltage (mV)', water: 'Water (ml)', fractional: 'Fractional content', rate: 'Rate (ml/day)' }

export default function SensorsPane({ plantFilter, sensorFilter, calibrator, plantSensors }) {
  const [measureMode, setMeasureMode] = useState('voltage')
  const [showFractional, setShowFractional] = useState(false)
  const [rangeHours, setRangeHours] = useState(48)

  const { series, observations, loading: sensorLoading, error: sensorError } = useSensorData({
    plantFilter, sensorFilter, rangeHours,
  })

  const { pendingTime, pendingPlant, pendingMl, pickTime, cancel: cancelPending, setPendingMl } = usePendingTime()

  const isRateMode = measureMode === 'rate'
  const isWaterMode = measureMode === 'water' || measureMode === 'fractional'

  const displaySeries = useMemo(() => {
    if (isWaterMode) return calibrator.mappedSeries
    return series.map(s => ({
      ...s,
      points: s.points.map(p => ({
        ...p,
        v: measureMode === 'voltage' ? (p.mv ?? p.raw) : p.raw,
      })),
    }))
  }, [series, measureMode, isWaterMode, calibrator.mappedSeries])
  const displayBands = isWaterMode ? calibrator.mappedBands : []

  const handleTimePick = t => pickTime(t, plantFilter)

  const chartProps = {
    series: displaySeries,
    bands: displayBands,
    observations,
    rangeMs: rangeHours * 3600 * 1000,
    onTimePick: handleTimePick,
    pendingTime,
    yLabel: Y_LABELS[measureMode],
    eventWindowOffset: Number(calibrator.params.offsetMin) * 60 * 1000,
    eventWindowWidth: Number(calibrator.params.widthMin) * 60 * 1000,
  }

  const showFractionalRef = useRef(showFractional)
  showFractionalRef.current = showFractional

  const handleRecalculate = () => {
    calibrator.calculate({ plantFilter, sensorFilter, rangeHours, returnFractional: showFractionalRef.current, plantSensors })
  }

  useEffect(() => {
    if (!plantFilter) return
    calibrator.calculate({ plantFilter, sensorFilter, rangeHours, returnFractional: showFractionalRef.current, plantSensors })
  }, [plantFilter, sensorFilter, rangeHours])

  return (
    <div className="pane-grid">
      <div className="full controls">
        <div className="range-btns">
          {TIME_RANGES.map(r => (
            <button key={r.hours} className={rangeHours === r.hours ? 'active' : ''} onClick={() => setRangeHours(r.hours)}>{r.label}</button>
          ))}
        </div>
        <div className="range-btns">
          {[['raw', 'Raw'], ['voltage', 'mV'], ['water', 'Water (ml)'], ['fractional', 'Fractional'], ['rate', 'Rate']].map(([m, label]) => (
            <button key={m} className={measureMode === m ? 'active' : ''} onClick={() => {
              setMeasureMode(m)
              setShowFractional(m === 'fractional')
            }}>{label}</button>
          ))}
        </div>
      </div>

      <div className="full">
        {isRateMode
          ? <DryingRateDisplay dryingRate={calibrator.dryingRate} dryingRateLoading={calibrator.dryingRateLoading} plantFilter={plantFilter} rangeHours={rangeHours} />
          : <ChartDisplay
              {...chartProps}
              loading={sensorLoading}
              sensorError={sensorError}
              calibError={calibrator.calibError}
              calibLoading={calibrator.calibLoading || calibrator.swcLoading}
              isWaterMode={isWaterMode}
              hasCalibration={calibrator.mappedSeries.length > 0}
              plantFilter={plantFilter}
            />
        }
      </div>

      <div>
        <CalibrationControls
          calibParams={calibrator.params}
          setCalibParam={calibrator.setParam}
          recalculate={handleRecalculate}
          calibLoading={calibrator.calibLoading || calibrator.swcLoading}
          plantFilter={plantFilter}
        />
      </div>

      <div className="full">
        <ObservationsPanel
          observations={observations}
          calibration={calibrator.calibration}
        />
      </div>
    </div>
  )
}
