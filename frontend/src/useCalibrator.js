import { useState, useRef, useCallback } from 'react'
import { apiJson } from './api'
import { buildSwcTimeseriesParams, buildDryingRateParams, buildWaterCalibrationParams } from './paramsBuilder'
import { PALETTE } from './theme'

const DEFAULT_CALIB_PARAMS = {
  offsetMin: '5', widthMin: '50', prior: 'calibrated', priorMin: '867', priorMax: '2009',
  estimator: 'exp_mcmc', priorWeight: '1.0', nBurn: '10', nSteps: '30', emaTauMin: '60',
  systemCapacityMean: '', systemCapacityStd: '',
}

export function useCalibrator() {
  const [params, setParams] = useState(DEFAULT_CALIB_PARAMS)
  const [calibration, setCalibration] = useState(null)
  const [calibError, setCalibError] = useState(null)
  const [calibLoading, setCalibLoading] = useState(false)
  const [dryingRate, setDryingRate] = useState(null)
  const [dryingRateLoading, setDryingRateLoading] = useState(false)
  const [mappedSeries, setMappedSeries] = useState([])
  const [mappedBands, setMappedBands] = useState([])

  const drControllerRef = useRef(null)

  const setParam = useCallback((key, val) => {
    setParams(p => ({ ...p, [key]: val }))
  }, [])

  const calculate = useCallback(({ plantFilter, sensorFilter, rangeHours, returnFractional = false }) => {
    const isCombined = sensorFilter === '__combined__'

    if (!plantFilter) {
      setCalibration(null)
      setCalibError(null)
      setDryingRate(null)
      setDryingRateLoading(false)
      setMappedSeries([])
      setMappedBands([])
      return
    }

    drControllerRef.current?.abort()
    const controller = new AbortController()
    const drController = new AbortController()
    drControllerRef.current = drController
    setCalibLoading(true)
    setCalibError(null)
    setDryingRateLoading(true)
    setMappedSeries([])
    setMappedBands([])

    const autoStdSet = { current: false }

    const endMs = Date.now()
    const startMs = endMs - rangeHours * 3600 * 1000

    const swcParams = buildSwcTimeseriesParams(plantFilter, params, rangeHours, returnFractional)
    if (!isCombined && sensorFilter) {
      const sep = sensorFilter.lastIndexOf(':')
      swcParams.set('sensor', sensorFilter.slice(sep + 1))
      swcParams.set('device_address', sensorFilter.slice(0, sep))
    }
    swcParams.set('start_ms', String(startMs))
    swcParams.set('end_ms', String(endMs))

    apiJson(`/swc_timeseries?${swcParams}`, { signal: controller.signal })
      .then(swc => {
        if (!swc.times_ms?.length) {
          setMappedSeries([])
          setMappedBands([])
          return
        }

        const label = isCombined ? 'Combined SWC' : `${plantFilter} / water`
        const color = PALETTE[0]
        const points = swc.times_ms.map((t, i) => ({ t, v: swc.mean_swc[i], raw: swc.mean_swc[i] })).filter(p => p.v != null)
        const bandPoints = swc.times_ms.map((t, i) => ({ t, lo: swc.ci_low[i], hi: swc.ci_high[i] })).filter(p => p.lo != null && p.hi != null)

        setMappedSeries(points.length ? [{ label, points, color }] : [])
        setMappedBands(bandPoints.length ? [{ color, points: bandPoints }] : [])
      })
      .catch(err => {
        if (err.name !== 'AbortError') {
          console.error(err)
          setCalibError(err.message)
        }
      })

    const waterParams = buildWaterCalibrationParams(plantFilter, sensorFilter, params, returnFractional)

    apiJson(`/water_calibration?${waterParams}`, { signal: controller.signal })
      .then(d => {
        setCalibration(d)

        if (!autoStdSet.current && d.chords_x?.length > 0) {
          const endpoints = d.chords_x.map((x, i) => x + (d.chords_dx?.[i] ?? 0))
          const newMin = String(Math.round(Math.min(...d.chords_x, ...endpoints)))
          if (newMin !== params.priorMin) {
            setParams(p => ({ ...p, priorMin: newMin }))
          }
          autoStdSet.current = true
        }

        const drParams = buildDryingRateParams(plantFilter, sensorFilter, params, rangeHours, isCombined, returnFractional)
        apiJson(`/drying_rate?${drParams}`, { signal: drController.signal })
          .then(dr => setDryingRate(dr))
          .catch(() => setDryingRate(null))
          .finally(() => setDryingRateLoading(false))
      })
      .catch(err => {
        if (err.name !== 'AbortError') {
          console.error(err)
          setCalibration(null)
          setCalibError(err.message)
        }
      })
      .finally(() => setCalibLoading(false))

    return () => { controller.abort(); drController.abort() }
  }, [params])

  return {
    params,
    setParam,
    calibration,
    calibError,
    calibLoading,
    dryingRate,
    dryingRateLoading,
    mappedSeries,
    mappedBands,
    calculate,
  }
}
