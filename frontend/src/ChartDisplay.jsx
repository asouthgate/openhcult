import { TimeseriesChart } from './TimeseriesChart'

export default function ChartDisplay({
  series, bands, observations, rangeMs, onTimePick, pendingTime, yLabel,
  eventWindowOffset, eventWindowWidth, loading, sensorError,
  calibError, calibLoading, isWaterMode, hasCalibration, plantFilter,
}) {
  if (isWaterMode) {
    if (calibLoading) {
      return <div className="loading"><span className="spinner" />Computing calibration…</div>
    }
    if (!plantFilter) {
      return <div className="empty">Select a plant to show water estimates</div>
    }
    if (!hasCalibration) {
      if (calibError) {
        return <div className="full error">Calibration failed: {calibError}</div>
      }
      return <div className="empty">Click Recalculate to compute water estimates</div>
    }
    if (series.length === 0) {
      return <div className="empty">No water data in range.</div>
    }
    return (
      <div>
        {calibError && <div className="full error">Calibration warning: {calibError}</div>}
        <TimeseriesChart
          series={series}
          bands={bands}
          observations={observations}
          rangeMs={rangeMs}
          onTimePick={onTimePick}
          pendingTime={pendingTime}
          yLabel={yLabel}
          eventWindowOffset={eventWindowOffset}
          eventWindowWidth={eventWindowWidth}
        />
      </div>
    )
  }

  if (sensorError) {
    return <div className="full error">Sensor data error: {sensorError}</div>
  }
  if (loading) {
    return <div className="loading">Loading…</div>
  }
  if (series.length === 0) {
    return <div className="empty">No data in range.</div>
  }

  return (
    <TimeseriesChart
      series={series}
      bands={bands}
      observations={observations}
      rangeMs={rangeMs}
      onTimePick={onTimePick}
      pendingTime={pendingTime}
      yLabel={yLabel}
      eventWindowOffset={eventWindowOffset}
      eventWindowWidth={eventWindowWidth}
    />
  )
}
