import { useState, useEffect, useRef, useCallback } from 'react'
import { apiJson } from './api'
import { buildSwcTimeseriesParams, buildDryingRateParams, buildWaterCalibrationParams } from './paramsBuilder'

export function useCalibrationData({ plantFilter, sensorFilter, calibParams, rangeHours, setCalibParam, returnFractional = false }) {
  const [calibration, setCalibration] = useState(null)
  const [calibError, setCalibError] = useState(null)
  const [calibLoading, setCalibLoading] = useState(false)
  const [dryingRate, setDryingRate] = useState(null)
  const [combinedSwc, setCombinedSwc] = useState(null)

  const calibParamsRef = useRef(calibParams)
  calibParamsRef.current = calibParams

  const setCalibParamRef = useRef(setCalibParam)
  setCalibParamRef.current = setCalibParam

  const autoStdSet = useRef(false)
  const drControllerRef = useRef(null)

  const isCombined = sensorFilter === '__combined__'

  const doFetch = useCallback(() => {
    if (!plantFilter) {
      setCalibration(null)
      setCalibError(null)
      setDryingRate(null)
      setCombinedSwc(null)
      drControllerRef.current?.abort()
      return
    }

    drControllerRef.current?.abort()
    const controller = new AbortController()
    const drController = new AbortController()
    drControllerRef.current = drController
    setCalibLoading(true)
    setCalibError(null)

    const params = calibParamsRef.current
    const hours = rangeHours

    if (isCombined) {
      const swcParams = buildSwcTimeseriesParams(plantFilter, params, hours, returnFractional)
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

      const drParams = buildDryingRateParams(plantFilter, '', params, hours, true, returnFractional)
      apiJson(`/drying_rate?${drParams}`, { signal: drController.signal })
        .then(dr => setDryingRate(dr))
        .catch(() => setDryingRate(null))

      return () => {
        controller.abort()
        drController.abort()
      }
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
            setCalibParamRef.current('priorMin', newMin)
          }
          autoStdSet.current = true
        }

        const drParams = buildDryingRateParams(plantFilter, sensorFilter, calibParamsRef.current, hours, false, returnFractional)
        apiJson(`/drying_rate?${drParams}`, { signal: drController.signal })
          .then(dr => setDryingRate(dr))
          .catch(() => setDryingRate(null))
      })
      .catch(err => {
        if (err.name !== 'AbortError') {
          console.error(err)
          setCalibration(null)
          setCalibError(err.message)
        }
      })
      .finally(() => setCalibLoading(false))

    return () => {
      controller.abort()
      drController.abort()
    }
  }, [plantFilter, sensorFilter, rangeHours, isCombined, returnFractional])

  useEffect(() => {
    autoStdSet.current = false
    const cleanup = doFetch()
    return cleanup
  }, [doFetch])

  const recalculate = useCallback(() => {
    autoStdSet.current = false
    const cleanup = doFetch()
    return cleanup
  }, [doFetch])

  return {
    calibration,
    calibError,
    calibLoading,
    dryingRate,
    combinedSwc,
    recalculate,
  }
}
