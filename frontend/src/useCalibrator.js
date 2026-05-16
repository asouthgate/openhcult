import { useState, useRef, useCallback } from 'react'
import { apiJson } from './api'
import { buildSwcTimeseriesParams, buildDryingRateParams, buildWaterCalibrationParams } from './paramsBuilder'
import { PALETTE } from './theme'

const DEBUG = true

const DEFAULT_CALIB_PARAMS = {
  offsetMin: '5', widthMin: '50', estimator: 'exp_mcmc', priorWeight: '1.0', nBurn: '10', nSteps: '30', emaTauMin: '60',
  systemCapacityMean: '', systemCapacityStd: '',
}

function parseSensorKey(key) {
  const sep = key.lastIndexOf(':')
  return { deviceAddress: key.slice(0, sep), sensor: key.slice(sep + 1) }
}

export function useCalibrator() {
  const [params, setParams] = useState(DEFAULT_CALIB_PARAMS)
  const [calibration, setCalibration] = useState(null)
  const [calibError, setCalibError] = useState(null)
  const [calibLoading, setCalibLoading] = useState(false)
  const [swcLoading, setSwcLoading] = useState(false)
  const [dryingRate, setDryingRate] = useState(null)
  const [dryingRateLoading, setDryingRateLoading] = useState(false)
  const [mappedSeries, setMappedSeries] = useState([])
  const [mappedBands, setMappedBands] = useState([])

  const drControllerRef = useRef(null)

  const setParam = useCallback((key, val) => {
    setParams(p => ({ ...p, [key]: val }))
  }, [])

  const calculate = useCallback(({ plantFilter, sensorFilter, rangeHours, returnFractional = false, plantSensors = [] }) => {
    const isAll = sensorFilter === '_all_'
    const isSingle = !isAll && !!sensorFilter

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
    const drController = new AbortController()
    drControllerRef.current = drController
    setCalibLoading(true)
    setSwcLoading(true)
    setCalibError(null)
    setDryingRateLoading(true)
    setMappedSeries([])
    setMappedBands([])

    const autoStdSet = { current: false }

    const endMs = Date.now()
    const startMs = endMs - rangeHours * 3600 * 1000

    function swcUrlForSensor(sensorSpec) {
      const swcParams = buildSwcTimeseriesParams(plantFilter, params, rangeHours, returnFractional)
      if (sensorSpec) {
        swcParams.set('sensor', sensorSpec.sensor)
        swcParams.set('device_address', sensorSpec.deviceAddress)
      }
      swcParams.set('start_ms', String(startMs))
      swcParams.set('end_ms', String(endMs))
      return `/swc_timeseries?${swcParams}`
    }

    function toSwcSeries(swc, label, color) {
      if (!swc.times_ms?.length) return null
      const points = swc.times_ms.map((t, i) => ({ t, v: swc.mean_swc[i], raw: swc.mean_swc[i] })).filter(p => p.v != null)
      const bandPoints = swc.times_ms.map((t, i) => ({ t, lo: swc.ci_low[i], hi: swc.ci_high[i] })).filter(p => p.lo != null && p.hi != null)
      if (DEBUG) {
        const nTotal = swc.times_ms.length
        const nValid = swc.mean_swc?.filter(v => v != null).length ?? 0
        console.log(`[swc_timeseries] label=${label} n_times=${nTotal} n_valid=${nValid} points=${points.length}`)
      }
      return {
        series: points.length ? [{ label, points, color }] : [],
        bands: bandPoints.length ? [{ color, points: bandPoints }] : [],
      }
    }

    if (isAll) {
      const sensorSpecs = plantSensors
        .filter(ps => ps.plant_name === plantFilter)
        .map(ps => ({ deviceAddress: ps.device_address, sensor: ps.sensor }))

      const swcController = new AbortController()
      const allSeries = []
      const allBands = []
      let pendingSwc = sensorSpecs.length + 1

      const combinedUrl = swcUrlForSensor(null)
      apiJson(combinedUrl, { signal: swcController.signal })
        .then(swc => {
          const result = toSwcSeries(swc, 'Combined SWC', PALETTE[0])
          if (result) {
            allSeries.push(...result.series)
            allBands.push(...result.bands)
          }
        })
        .catch(err => {
          if (err.name !== 'AbortError') {
            console.error('swc_timeseries (combined) error:', err)
            setCalibError(err.message)
          }
        })
        .finally(() => {
          pendingSwc--
          if (pendingSwc === 0) {
            setMappedSeries(allSeries)
            setMappedBands(allBands)
            setSwcLoading(false)
          }
        })

      for (let i = 0; i < sensorSpecs.length; i++) {
        const spec = sensorSpecs[i]
        const label = `${spec.deviceAddress} / ${spec.sensor}`
        const color = PALETTE[(i + 1) % PALETTE.length]
        apiJson(swcUrlForSensor(spec), { signal: swcController.signal })
          .then(swc => {
            const result = toSwcSeries(swc, label, color)
            if (result) {
              allSeries.push(...result.series)
              allBands.push(...result.bands)
            }
          })
          .catch(err => {
            if (err.name !== 'AbortError') {
              console.error(`swc_timeseries (${label}) error:`, err)
            }
          })
          .finally(() => {
            pendingSwc--
            if (pendingSwc === 0) {
              setMappedSeries([...allSeries])
              setMappedBands([...allBands])
              setSwcLoading(false)
            }
          })
      }

      if (sensorSpecs.length === 0) {
        setMappedSeries([])
        setMappedBands([])
        setSwcLoading(false)
      }
    } else {
      const swcParams = buildSwcTimeseriesParams(plantFilter, params, rangeHours, returnFractional)
      if (isSingle) {
        const { deviceAddress, sensor } = parseSensorKey(sensorFilter)
        swcParams.set('sensor', sensor)
        swcParams.set('device_address', deviceAddress)
      }
      swcParams.set('start_ms', String(startMs))
      swcParams.set('end_ms', String(endMs))

      const swcController = new AbortController()

      apiJson(`/swc_timeseries?${swcParams}`, { signal: swcController.signal })
        .then(swc => {
          if (!swc.times_ms?.length) {
            if (DEBUG) console.log('[swc_timeseries] empty times_ms')
            setMappedSeries([])
            setMappedBands([])
            return
          }

          const nTotal = swc.times_ms.length
          const nValid = swc.mean_swc?.filter(v => v != null).length ?? 0
          if (DEBUG) {
            console.log(
              `[swc_timeseries] n_times=${nTotal} n_valid_swc=${nValid} n_null_swc=${nTotal - nValid}`,
              'first_values:', swc.mean_swc?.slice(0, 5),
              'last_values:', swc.mean_swc?.slice(-5),
            )
          }

          const label = isSingle ? `${plantFilter} / ${parseSensorKey(sensorFilter).sensor}` : 'Combined SWC'
          const color = PALETTE[0]
          const points = swc.times_ms.map((t, i) => ({ t, v: swc.mean_swc[i], raw: swc.mean_swc[i] })).filter(p => p.v != null)
          const bandPoints = swc.times_ms.map((t, i) => ({ t, lo: swc.ci_low[i], hi: swc.ci_high[i] })).filter(p => p.lo != null && p.hi != null)

          if (DEBUG) console.log(`[swc_timeseries] after filter: points=${points.length} bandPoints=${bandPoints.length}`)
          setMappedSeries(points.length ? [{ label, points, color }] : [])
          setMappedBands(bandPoints.length ? [{ color, points: bandPoints }] : [])
        })
        .catch(err => {
          if (err.name !== 'AbortError') {
            console.error('swc_timeseries error:', err)
            setCalibError(err.message)
          }
          setMappedSeries([])
          setMappedBands([])
        })
        .finally(() => setSwcLoading(false))
    }

    const effectiveSensor = isAll ? '' : sensorFilter
    const waterParams = buildWaterCalibrationParams(plantFilter, effectiveSensor, params, returnFractional)
    const waterController = new AbortController()

    apiJson(`/water_calibration?${waterParams}`, { signal: waterController.signal })
      .then(d => {
        setCalibration(d)
      })
      .catch(err => {
        if (err.name !== 'AbortError') {
          setCalibError(err.message)
        }
        setCalibration(null)
      })
      .finally(() => setCalibLoading(false))

    const drParams = buildDryingRateParams(plantFilter, effectiveSensor, params, rangeHours, isAll, returnFractional)
    apiJson(`/drying_rate?${drParams}`, { signal: drController.signal })
      .then(dr => setDryingRate(dr))
      .catch(() => setDryingRate(null))
      .finally(() => setDryingRateLoading(false))

    return () => { drController.abort() }
  }, [params])

  return {
    params,
    setParam,
    calibration,
    calibError,
    calibLoading,
    swcLoading,
    dryingRate,
    dryingRateLoading,
    mappedSeries,
    mappedBands,
    calculate,
  }
}