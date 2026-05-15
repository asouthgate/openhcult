import { useState, useRef, useCallback } from 'react'
import { apiJson } from './api'
import { buildSwcTimeseriesParams, buildDryingRateParams, buildWaterCalibrationParams } from './paramsBuilder'

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
  const [combinedSwc, setCombinedSwc] = useState(null)

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
      setCombinedSwc(null)
      return
    }

    drControllerRef.current?.abort()
    const controller = new AbortController()
    const drController = new AbortController()
    drControllerRef.current = drController
    setCalibLoading(true)
    setCalibError(null)
    setDryingRateLoading(true)

    const autoStdSet = { current: false }

    if (isCombined) {
      const swcParams = buildSwcTimeseriesParams(plantFilter, params, rangeHours, returnFractional)
      setCalibration(null)
      setCombinedSwc(null)
      apiJson(`/swc_timeseries?${swcParams}`, { signal: controller.signal })
        .then(d => { setCombinedSwc(d); setCalibLoading(false) })
        .catch(err => {
          if (err.name !== 'AbortError') {
            console.error(err)
            setCalibError(err.message)
            setCombinedSwc(null)
            setCalibLoading(false)
          }
        })

      const drParams = buildDryingRateParams(plantFilter, '', params, rangeHours, true, returnFractional)
      apiJson(`/drying_rate?${drParams}`, { signal: drController.signal })
        .then(dr => setDryingRate(dr))
        .catch(() => setDryingRate(null))
        .finally(() => setDryingRateLoading(false))

      return () => { controller.abort(); drController.abort() }
    }

    setCombinedSwc(null)
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

        const drParams = buildDryingRateParams(plantFilter, sensorFilter, params, rangeHours, false, returnFractional)
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
    combinedSwc,
    calculate,
  }
}
