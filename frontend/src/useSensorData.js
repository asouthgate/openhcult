import { useState, useEffect, useCallback } from 'react'
import { apiJson } from './api'
import { sensorKey, parseSensorKey } from './utils'
import { PALETTE } from './theme'

export function useSensorData({ plantFilter, sensorFilter, rangeHours }) {
  const [series, setSeries] = useState([])
  const [observations, setObservations] = useState([])
  const [loading, setLoading] = useState(false)

  const fetch = useCallback(() => {
    const end = new Date()
    const start = new Date(end - rangeHours * 3600 * 1000)
    const params = new URLSearchParams({ start_utc: start.toISOString(), end_utc: end.toISOString(), limit: '50000' })
    if (plantFilter) params.set('plant', plantFilter)
    const isCombined = sensorFilter === '__combined__'
    if (sensorFilter && !isCombined) {
      const { deviceAddress, sensor } = parseSensorKey(sensorFilter)
      params.set('sensor', sensor)
      params.set('device_address', deviceAddress)
    }

    setLoading(true)

    return apiJson(`/timeseries?${params}`)
      .then(ts => {
        const grouped = {}
        for (const row of ts.data ?? []) {
          const key = sensorKey(row.device_address, row.sensor)
          if (sensorFilter && !isCombined && key !== sensorFilter) continue
          const label = plantFilter
            ? `${plantFilter} / ${row.device_address} / ${row.sensor}`
            : key
          if (!grouped[key]) grouped[key] = { label, points: [] }
          grouped[key].points.push({ t: row.adjusted_time_ms, raw: row.measurement, mv: row.voltage_mv })
        }
        setSeries(Object.entries(grouped).map(([, s], i) => ({ ...s, color: PALETTE[i % PALETTE.length] })))
      })
      .finally(() => setLoading(false))
  }, [plantFilter, sensorFilter, rangeHours])

  useEffect(() => {
    setSeries([])
    fetch()
  }, [fetch])

  const fetchObservations = useCallback(() => {
    const end = new Date()
    const start = new Date(end - rangeHours * 3600 * 1000)
    const params = new URLSearchParams({ start_utc: start.toISOString(), end_utc: end.toISOString(), limit: '10000' })
    return apiJson(`/observations?${params}`)
      .then(obs => {
        const filtered = (obs.data ?? []).filter(o => !plantFilter || o.plant_name === plantFilter)
        setObservations(filtered)
      })
  }, [plantFilter, rangeHours])

  useEffect(() => {
    fetchObservations()
  }, [fetchObservations])

  return { series, observations, loading }
}
