import { useState, useEffect } from 'react'
import { apiJson, clearToken } from './api'
import SensorsPane from './SensorsPane'
import CalibrationPane from './CalibrationPane'
import { useCalibrator } from './useCalibrator'

export default function Sensors() {
  const [view, setView] = useState('sensors')
  const [plantFilter, setPlantFilter] = useState('')
  const [plants, setPlants] = useState([])
  const [plantSensors, setPlantSensors] = useState([])
  const [sensorFilter, setSensorFilter] = useState('')

  const calibrator = useCalibrator()

  useEffect(() => {
    apiJson('/plants?limit=1000').then(d => setPlants(d.data ?? []))
    apiJson('/plant_sensors').then(d => setPlantSensors(d.data ?? []))
  }, [])

  function handlePlantChange(plant) {
    setPlantFilter(plant)
    setSensorFilter('')
    if (plant) {
      const p = plants.find(p => p.plant_name === plant)
      if (p?.soil_volume != null) {
        const mean = Math.round(p.soil_volume)
        calibrator.setParam('systemCapacityMean', String(mean))
        calibrator.setParam('systemCapacityStd', String(Math.round(mean * 0.1)))
      } else {
        calibrator.setParam('systemCapacityMean', '')
        calibrator.setParam('systemCapacityStd', '')
      }
    } else {
      calibrator.setParam('systemCapacityMean', '')
      calibrator.setParam('systemCapacityStd', '')
    }
  }

  return (
    <div className="app">
      <div className="app-header">
        <div className="logo"><img src={`${import.meta.env.BASE_URL}teal-no-bg.png`} alt="HCult" /><h1>HCult</h1></div>
        <div className="controls">
          <div className="range-btns">
            <button className={view === 'sensors' ? 'active' : ''} onClick={() => setView('sensors')}>Sensors</button>
            <button className={view === 'calibration' ? 'active' : ''} onClick={() => setView('calibration')}>Calibration</button>
          </div>
          <select value={plantFilter} onChange={e => handlePlantChange(e.target.value)}>
            <option value="">All plants</option>
            {plants.map(p => <option key={p.plant_name} value={p.plant_name}>{p.plant_name}</option>)}
          </select>
          {plantSensors.filter(ps => ps.plant_name === plantFilter).length > 0 && (
            <select value={sensorFilter} onChange={e => setSensorFilter(e.target.value)}>
              <option value="">All sensors</option>
              <option value="__combined__">Combined</option>
              {plantSensors.filter(ps => ps.plant_name === plantFilter).map(ps => (
                <option key={`${ps.device_address}:${ps.sensor}`} value={`${ps.device_address}:${ps.sensor}`}>
                  {ps.device_address} / {ps.sensor}
                </option>
              ))}
            </select>
          )}
          <button className="logout-btn" onClick={() => { clearToken(); window.location.reload() }}>Logout</button>
        </div>
      </div>

      {view === 'sensors'
        ? <SensorsPane plantFilter={plantFilter} sensorFilter={sensorFilter} calibrator={calibrator} />
        : <CalibrationPane plantFilter={plantFilter} sensorFilter={sensorFilter} calibrator={calibrator} />
      }
    </div>
  )
}
