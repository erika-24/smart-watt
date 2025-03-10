// API endpoints for fetching data
const API_BASE_URL = "/api"

// Energy data fetching
export async function fetchCurrentEnergyData() {
  try {
    const response = await fetch(`${API_BASE_URL}/energy/current`)
    if (!response.ok) throw new Error("Failed to fetch current energy data")
    return await response.json()
  } catch (error) {
    console.error("Error fetching current energy data:", error)
    throw error
  }
}

export async function fetchForecastData(date: string) {
  try {
    const response = await fetch(`${API_BASE_URL}/energy/forecast?date=${date}`)
    if (!response.ok) throw new Error("Failed to fetch forecast data")
    return await response.json()
  } catch (error) {
    console.error("Error fetching forecast data:", error)
    throw error
  }
}

export async function fetchHistoricalData(timeRange: string, date: string) {
  try {
    const response = await fetch(`${API_BASE_URL}/energy/history?timeRange=${timeRange}&date=${date}`)
    if (!response.ok) throw new Error("Failed to fetch historical data")
    return await response.json()
  } catch (error) {
    console.error("Error fetching historical data:", error)
    throw error
  }
}

export async function fetchEnergyBreakdown() {
  try {
    const response = await fetch(`${API_BASE_URL}/energy/breakdown`)
    if (!response.ok) throw new Error("Failed to fetch energy breakdown")
    return await response.json()
  } catch (error) {
    console.error("Error fetching energy breakdown:", error)
    throw error
  }
}

// Device data fetching
export async function fetchDevices() {
  try {
    const response = await fetch(`${API_BASE_URL}/devices`)
    if (!response.ok) throw new Error("Failed to fetch devices")
    return await response.json()
  } catch (error) {
    console.error("Error fetching devices:", error)
    throw error
  }
}

export async function toggleDevice(id: string, isOn: boolean) {
  try {
    const response = await fetch(`${API_BASE_URL}/devices/${id}/toggle`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ isOn }),
    })
    if (!response.ok) throw new Error("Failed to toggle device")
    return await response.json()
  } catch (error) {
    console.error("Error toggling device:", error)
    throw error
  }
}

// Optimization data fetching
export async function runOptimization(params: OptimizationParams) {
  try {
    const response = await fetch(`${API_BASE_URL}/optimization/run`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(params),
    })
    if (!response.ok) throw new Error("Failed to run optimization")
    return await response.json()
  } catch (error) {
    console.error("Error running optimization:", error)
    throw error
  }
}

export async function fetchOptimizationResults(id: string) {
  try {
    const response = await fetch(`${API_BASE_URL}/optimization/results/${id}`)
    if (!response.ok) throw new Error("Failed to fetch optimization results")
    return await response.json()
  } catch (error) {
    console.error("Error fetching optimization results:", error)
    throw error
  }
}

export async function applyOptimizationSchedule(id: string) {
  try {
    const response = await fetch(`${API_BASE_URL}/optimization/apply/${id}`, {
      method: "POST",
    })
    if (!response.ok) throw new Error("Failed to apply optimization schedule")
    return await response.json()
  } catch (error) {
    console.error("Error applying optimization schedule:", error)
    throw error
  }
}

// Types
export interface OptimizationParams {
  optimizationMode: string
  timeHorizon: string
  batteryConstraints: {
    enabled: boolean
    minSoc: number
    maxCycles: number
  }
  gridConstraints: {
    enabled: boolean
    maxPower: number
  }
}

export interface Device {
  id: string
  name: string
  type: string
  status: "online" | "offline"
  power: number
  isOn: boolean
  icon: string
  schedule?: {
    start: string
    end: string
    power: number
  }[]
}

export interface EnergyData {
  time: string
  consumption: number
  solar: number
  battery: number
  grid: number
}

export interface ForecastData {
  time: string
  load: number
  solar: number
  optimizedLoad: number
}

export interface OptimizationResult {
  id: string
  cost: number
  selfConsumption: number
  peakGridPower: number
  batteryCycles: number
  timestamp: string
  scheduleData: {
    time: string
    load: number
    solar: number
    battery: number
    grid: number
    optimizedLoad: number
  }[]
  deviceScheduleData: {
    id: string
    name: string
    type: string
    schedule: {
      start: string
      end: string
      power: number
    }[]
  }[]
}

