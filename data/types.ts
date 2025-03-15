// Types for energy data and optimization

export interface EnergyDataPoint {
  timestamp: string
  load_power: number // kW
  pv_power: number // kW
  battery_power: number // kW (positive = charging, negative = discharging)
  battery_soc: number // % (0-100)
  grid_power: number // kW (positive = import, negative = export)
}

export interface ForecastDataPoint {
  timestamp: string
  load_power: number // kW
  pv_power: number // kW
}

export interface OptimizationResult {
  id: string
  timestamp: string
  cost: number
  self_consumption: number
  peak_grid_power: number
  battery_cycles: number
  schedule: OptimizationSchedulePoint[]
  device_schedules: DeviceSchedule[]
}

export interface OptimizationSchedulePoint {
  timestamp: string
  load_power: number
  pv_power: number
  battery_power: number
  grid_power: number
  optimized_load: number
}

export interface DeviceSchedule {
  id: string
  name: string
  type: string
  schedule: DeviceScheduleSlot[]
}

export interface DeviceScheduleSlot {
  start: string
  end: string
  power: number
}

export interface Device {
  id: string
  name: string
  type: string
  status: "online" | "offline"
  power: number
  isOn: boolean
  icon: string
  shiftable: boolean
  min_power?: number
  max_power?: number
  schedule?: DeviceScheduleSlot[]
}

export interface OptimizationParams {
  objective: "cost" | "self_consumption" | "grid_independence" | "battery_life"
  time_horizon: number // hours
  battery_constraints: {
    enabled: boolean
    min_soc: number
    max_cycles: number
  }
  grid_constraints: {
    enabled: boolean
    max_power: number
  }
  price_data?: {
    import: number[]
    export: number[]
  }
}

export interface EnergySystemConfig {
  battery_capacity: number // kWh
  battery_max_power: number // kW
  battery_efficiency: number // 0-1
  pv_capacity: number // kW
  grid_connection_capacity: number // kW
}

