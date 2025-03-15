import type { EnergyDataPoint, ForecastDataPoint, OptimizationResult, Device, EnergySystemConfig } from "./types"
import fs from "fs"
import path from "path"

// Define paths for data storage
const DATA_DIR = path.join(process.cwd(), "data", "storage")
const ENERGY_DATA_FILE = path.join(DATA_DIR, "energy_data.json")
const FORECAST_DATA_FILE = path.join(DATA_DIR, "forecast_data.json")
const OPTIMIZATION_RESULTS_FILE = path.join(DATA_DIR, "optimization_results.json")
const DEVICES_FILE = path.join(DATA_DIR, "devices.json")
const SYSTEM_CONFIG_FILE = path.join(DATA_DIR, "system_config.json")

// Ensure data directory exists
if (!fs.existsSync(DATA_DIR)) {
  fs.mkdirSync(DATA_DIR, { recursive: true })
}

// Initialize files if they don't exist
function initializeFiles() {
  if (!fs.existsSync(ENERGY_DATA_FILE)) {
    fs.writeFileSync(ENERGY_DATA_FILE, JSON.stringify([]))
  }
  if (!fs.existsSync(FORECAST_DATA_FILE)) {
    fs.writeFileSync(FORECAST_DATA_FILE, JSON.stringify([]))
  }
  if (!fs.existsSync(OPTIMIZATION_RESULTS_FILE)) {
    fs.writeFileSync(OPTIMIZATION_RESULTS_FILE, JSON.stringify([]))
  }
  if (!fs.existsSync(DEVICES_FILE)) {
    fs.writeFileSync(DEVICES_FILE, JSON.stringify([]))
  }
  if (!fs.existsSync(SYSTEM_CONFIG_FILE)) {
    // Default system configuration
    const defaultConfig: EnergySystemConfig = {
      battery_capacity: 10, // kWh
      battery_max_power: 5, // kW
      battery_efficiency: 0.95, // 95%
      pv_capacity: 8, // kW
      grid_connection_capacity: 10, // kW
    }
    fs.writeFileSync(SYSTEM_CONFIG_FILE, JSON.stringify(defaultConfig))
  }
}

// Initialize files on module load
initializeFiles()

// Energy data functions
export async function saveEnergyData(data: EnergyDataPoint[]) {
  fs.writeFileSync(ENERGY_DATA_FILE, JSON.stringify(data))
}

export async function getEnergyData(): Promise<EnergyDataPoint[]> {
  const data = fs.readFileSync(ENERGY_DATA_FILE, "utf8")
  return JSON.parse(data)
}

export async function appendEnergyData(dataPoint: EnergyDataPoint) {
  const data = await getEnergyData()
  data.push(dataPoint)

  // Keep only the last 7 days of data
  const sevenDaysAgo = new Date()
  sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7)
  const filteredData = data.filter((point) => new Date(point.timestamp) >= sevenDaysAgo)

  await saveEnergyData(filteredData)
}

// Forecast data functions
export async function saveForecastData(data: ForecastDataPoint[]) {
  fs.writeFileSync(FORECAST_DATA_FILE, JSON.stringify(data))
}

export async function getForecastData(): Promise<ForecastDataPoint[]> {
  const data = fs.readFileSync(FORECAST_DATA_FILE, "utf8")
  return JSON.parse(data)
}

// Optimization results functions
export async function saveOptimizationResult(result: OptimizationResult) {
  const results = await getOptimizationResults()
  results.push(result)

  // Keep only the last 30 optimization results
  const limitedResults = results.slice(-30)

  fs.writeFileSync(OPTIMIZATION_RESULTS_FILE, JSON.stringify(limitedResults))
  return result
}

export async function getOptimizationResults(): Promise<OptimizationResult[]> {
  const data = fs.readFileSync(OPTIMIZATION_RESULTS_FILE, "utf8")
  return JSON.parse(data)
}

export async function getOptimizationResultById(id: string): Promise<OptimizationResult | null> {
  const results = await getOptimizationResults()
  return results.find((result) => result.id === id) || null
}

// Device functions
export async function saveDevices(devices: Device[]) {
  fs.writeFileSync(DEVICES_FILE, JSON.stringify(devices))
}

export async function getDevices(): Promise<Device[]> {
  const data = fs.readFileSync(DEVICES_FILE, "utf8")
  return JSON.parse(data)
}

export async function updateDevice(id: string, updates: Partial<Device>): Promise<Device | null> {
  const devices = await getDevices()
  const deviceIndex = devices.findIndex((device) => device.id === id)

  if (deviceIndex === -1) {
    return null
  }

  devices[deviceIndex] = { ...devices[deviceIndex], ...updates }
  await saveDevices(devices)

  return devices[deviceIndex]
}

// System configuration functions
export async function getSystemConfig(): Promise<EnergySystemConfig> {
  const data = fs.readFileSync(SYSTEM_CONFIG_FILE, "utf8")
  return JSON.parse(data)
}

export async function updateSystemConfig(updates: Partial<EnergySystemConfig>): Promise<EnergySystemConfig> {
  const config = await getSystemConfig()
  const updatedConfig = { ...config, ...updates }

  fs.writeFileSync(SYSTEM_CONFIG_FILE, JSON.stringify(updatedConfig))

  return updatedConfig
}

