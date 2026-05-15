import { TimeseriesChart } from './TimeseriesChart'

export default function ChartDisplay({
  series, bands, observations, rangeMs, onTimePick, pendingTime, yLabel,
  eventWindowOffset, eventWindowWidth, loading, calibError, calibLoading,
  isWaterMode, hasCalibration, plantFilter,
}) {
  if (calibError) {
    return <div className="full error">{calibError}</div>
  }

  if (calibLoading) {
    return <div className="loading"><span className="spinner" />Computing calibration…</div>
  }

  if (loading) {
    return <div className="loading">Loading…</div>
  }

  if (isWaterMode && !plantFilter) {
    return <div className="empty">Select a plant to show water estimates</div>
  }

  if (isWaterMode && !hasCalibration) {
    return <div className="empty">Enter system capacity params and recalculate to show water estimates</div>
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
