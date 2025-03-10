import type {
  OptimizationParams,
  EnergySystemConfig,
  ForecastDataPoint,
  OptimizationResult,
  OptimizationSchedulePoint,
  Device,
  DeviceSchedule,
} from "@/data/types"
import * as glpk from "glpk.js"

// Initialize GLPK (GNU Linear Programming Kit)
let glpkInstance: any = null

async function initGlpk() {
  if (!glpkInstance) {
    glpkInstance = await glpk()
  }
  return glpkInstance
}

export async function runOptimization(
  params: OptimizationParams,
  forecast: ForecastDataPoint[],
  systemConfig: EnergySystemConfig,
  devices: Device[],
): Promise<OptimizationResult> {
  // Initialize GLPK
  const GLPK = await initGlpk()

  // Prepare optimization problem
  const problem = {
    name: "EnergyOptimization",
    objective: {
      direction: GLPK.GLP_MIN,
      name: "obj",
      vars: [] as any[],
    },
    subjectTo: [] as any[],
    bounds: [] as any[],
    binaries: [] as string[],
  }

  const timeSteps = forecast.length
  const timeHorizon = params.time_horizon

  // Only use forecast data up to the time horizon
  const forecastData = forecast.slice(0, timeHorizon)

  // Create variables for each time step
  // Grid power (import/export)
  for (let t = 0; t < timeHorizon; t++) {
    problem.bounds.push({
      name: `grid_import_${t}`,
      type: GLPK.GLP_LO,
      ub: params.grid_constraints.enabled ? params.grid_constraints.max_power : systemConfig.grid_connection_capacity,
      lb: 0,
    })

    problem.bounds.push({
      name: `grid_export_${t}`,
      type: GLPK.GLP_LO,
      ub: systemConfig.grid_connection_capacity,
      lb: 0,
    })
  }

  // Battery power (charge/discharge)
  for (let t = 0; t < timeHorizon; t++) {
    problem.bounds.push({
      name: `battery_charge_${t}`,
      type: GLPK.GLP_LO,
      ub: systemConfig.battery_max_power,
      lb: 0,
    })

    problem.bounds.push({
      name: `battery_discharge_${t}`,
      type: GLPK.GLP_LO,
      ub: systemConfig.battery_max_power,
      lb: 0,
    })
  }

  // Battery state of charge
  for (let t = 0; t <= timeHorizon; t++) {
    problem.bounds.push({
      name: `battery_soc_${t}`,
      type: GLPK.GLP_DB,
      ub: systemConfig.battery_capacity,
      lb: params.battery_constraints.enabled
        ? (params.battery_constraints.min_soc / 100) * systemConfig.battery_capacity
        : 0,
    })
  }

  // Device power
  const shiftableDevices = devices.filter((device) => device.shiftable)

  for (const device of shiftableDevices) {
    for (let t = 0; t < timeHorizon; t++) {
      problem.bounds.push({
        name: `device_${device.id}_${t}`,
        type: GLPK.GLP_DB,
        ub: device.max_power || device.power,
        lb: device.min_power || 0,
      })

      // Binary variable for device on/off status
      problem.bounds.push({
        name: `device_${device.id}_on_${t}`,
        type: GLPK.GLP_DB,
        ub: 1,
        lb: 0,
      })

      problem.binaries.push(`device_${device.id}_on_${t}`)
    }
  }

  // Set up objective function based on optimization mode
  switch (params.objective) {
    case "cost":
      setupCostObjective(problem, timeHorizon, params)
      break
    case "self_consumption":
      setupSelfConsumptionObjective(problem, timeHorizon, forecastData)
      break
    case "grid_independence":
      setupGridIndependenceObjective(problem, timeHorizon)
      break
    case "battery_life":
      setupBatteryLifeObjective(problem, timeHorizon, params)
      break
  }

  // Add power balance constraints for each time step
  for (let t = 0; t < timeHorizon; t++) {
    const loadPower = forecastData[t].load_power
    const pvPower = forecastData[t].pv_power

    // Power balance: PV + Grid Import + Battery Discharge = Load + Grid Export + Battery Charge
    problem.subjectTo.push({
      name: `power_balance_${t}`,
      vars: [
        { name: `grid_import_${t}`, coef: 1 },
        { name: `grid_export_${t}`, coef: -1 },
        { name: `battery_charge_${t}`, coef: -1 },
        { name: `battery_discharge_${t}`, coef: 1 },
      ],
      bnds: { type: GLPK.GLP_FX, ub: loadPower - pvPower, lb: loadPower - pvPower },
    })
  }

  // Add battery state of charge constraints
  for (let t = 0; t < timeHorizon; t++) {
    // SOC(t+1) = SOC(t) + charge_efficiency * charge - discharge / discharge_efficiency
    problem.subjectTo.push({
      name: `battery_soc_balance_${t}`,
      vars: [
        { name: `battery_soc_${t}`, coef: -1 },
        { name: `battery_soc_${t + 1}`, coef: 1 },
        { name: `battery_charge_${t}`, coef: -systemConfig.battery_efficiency },
        { name: `battery_discharge_${t}`, coef: 1 / systemConfig.battery_efficiency },
      ],
      bnds: { type: GLPK.GLP_FX, ub: 0, lb: 0 },
    })
  }

  // Add device constraints
  for (const device of shiftableDevices) {
    for (let t = 0; t < timeHorizon; t++) {
      // Device power must be between min and max when on, or 0 when off
      problem.subjectTo.push({
        name: `device_power_${device.id}_${t}`,
        vars: [
          { name: `device_${device.id}_${t}`, coef: 1 },
          { name: `device_${device.id}_on_${t}`, coef: -device.max_power || -device.power },
        ],
        bnds: { type: GLPK.GLP_UP, ub: 0, lb: 0 },
      })

      if (device.min_power && device.min_power > 0) {
        problem.subjectTo.push({
          name: `device_min_power_${device.id}_${t}`,
          vars: [
            { name: `device_${device.id}_${t}`, coef: 1 },
            { name: `device_${device.id}_on_${t}`, coef: -device.min_power },
          ],
          bnds: { type: GLPK.GLP_LO, ub: 0, lb: 0 },
        })
      }
    }
  }

  // Add battery cycle constraint if enabled
  if (params.battery_constraints.enabled && params.battery_constraints.max_cycles > 0) {
    const maxEnergyThroughput = params.battery_constraints.max_cycles * systemConfig.battery_capacity

    problem.subjectTo.push({
      name: "battery_cycles",
      vars: Array.from({ length: timeHorizon }, (_, t) => ({
        name: `battery_charge_${t}`,
        coef: 1,
      })),
      bnds: { type: GLPK.GLP_UP, ub: maxEnergyThroughput, lb: 0 },
    })
  }

  // Solve the optimization problem
  const result = GLPK.solve(problem)
  console.log("Optimization result:", result)

  if (result.result.status !== GLPK.GLP_OPT) {
    throw new Error(`Optimization failed with status: ${result.result.status}`)
  }

  // Extract results
  const schedule: OptimizationSchedulePoint[] = []
  const deviceSchedules: DeviceSchedule[] = []

  // Process optimization results
  for (let t = 0; t < timeHorizon; t++) {
    const timestamp = new Date(new Date(forecastData[t].timestamp).getTime() + t * 3600 * 1000).toISOString()
    const gridImport = result.result.vars[`grid_import_${t}`] || 0
    const gridExport = result.result.vars[`grid_export_${t}`] || 0
    const batteryCharge = result.result.vars[`battery_charge_${t}`] || 0
    const batteryDischarge = result.result.vars[`battery_discharge_${t}`] || 0

    // Calculate optimized load
    let optimizedLoad = forecastData[t].load_power

    // Adjust for shiftable devices
    for (const device of shiftableDevices) {
      const devicePower = result.result.vars[`device_${device.id}_${t}`] || 0
      optimizedLoad += devicePower
    }

    schedule.push({
      timestamp,
      load_power: forecastData[t].load_power,
      pv_power: forecastData[t].pv_power,
      battery_power: batteryCharge - batteryDischarge,
      grid_power: gridImport - gridExport,
      optimized_load: optimizedLoad,
    })
  }

  // Create device schedules
  for (const device of shiftableDevices) {
    const deviceSchedule: DeviceSchedule = {
      id: device.id,
      name: device.name,
      type: device.type,
      schedule: [],
    }

    let currentSlot: { start: string; end: string; power: number } | null = null

    for (let t = 0; t < timeHorizon; t++) {
      const timestamp = new Date(new Date(forecastData[t].timestamp).getTime() + t * 3600 * 1000)
      const deviceOn = result.result.vars[`device_${device.id}_on_${t}`] > 0.5
      const devicePower = result.result.vars[`device_${device.id}_${t}`] || 0

      if (deviceOn) {
        if (!currentSlot) {
          // Start a new slot
          currentSlot = {
            start: timestamp.toISOString().substring(11, 16), // HH:MM format
            end: new Date(timestamp.getTime() + 3600 * 1000).toISOString().substring(11, 16),
            power: devicePower,
          }
        } else {
          // Extend the current slot
          currentSlot.end = new Date(timestamp.getTime() + 3600 * 1000).toISOString().substring(11, 16)
        }
      } else if (currentSlot) {
        // End the current slot
        deviceSchedule.schedule.push(currentSlot)
        currentSlot = null
      }
    }

    // Add the last slot if it exists
    if (currentSlot) {
      deviceSchedule.schedule.push(currentSlot)
    }

    if (deviceSchedule.schedule.length > 0) {
      deviceSchedules.push(deviceSchedule)
    }
  }

  // Calculate metrics
  const totalLoad = schedule.reduce((sum, point) => sum + point.load_power, 0)
  const totalPv = schedule.reduce((sum, point) => sum + point.pv_power, 0)
  const totalGridImport = schedule.reduce((sum, point) => sum + Math.max(0, point.grid_power), 0)
  const totalGridExport = schedule.reduce((sum, point) => sum + Math.abs(Math.min(0, point.grid_power)), 0)
  const peakGridPower = Math.max(...schedule.map((point) => Math.abs(point.grid_power)))

  const selfConsumption = totalPv > 0 ? ((totalPv - totalGridExport) / totalPv) * 100 : 0

  const totalBatteryCharge = schedule.reduce((sum, point) => sum + Math.max(0, point.battery_power), 0)
  const batteryCycles = totalBatteryCharge / systemConfig.battery_capacity

  // Calculate cost if price data is available
  let cost = 0
  if (params.price_data) {
    for (let t = 0; t < timeHorizon; t++) {
      const gridImport = Math.max(0, schedule[t].grid_power)
      const gridExport = Math.abs(Math.min(0, schedule[t].grid_power))

      cost += gridImport * (params.price_data.import[t % params.price_data.import.length] || 0.15)
      cost -= gridExport * (params.price_data.export[t % params.price_data.export.length] || 0.05)
    }
  } else {
    // Default pricing
    cost = totalGridImport * 0.15 - totalGridExport * 0.05
  }

  return {
    id: `opt-${Date.now()}`,
    timestamp: new Date().toISOString(),
    cost,
    self_consumption: selfConsumption,
    peak_grid_power: peakGridPower,
    battery_cycles: batteryCycles,
    schedule,
    device_schedules: deviceSchedules,
  }
}

// Helper functions for setting up different objective functions

function setupCostObjective(problem: any, timeHorizon: number, params: OptimizationParams) {
  const priceData = params.price_data || {
    import: Array(24).fill(0.15), // Default import price: $0.15/kWh
    export: Array(24).fill(0.05), // Default export price: $0.05/kWh
  }

  for (let t = 0; t < timeHorizon; t++) {
    const importPrice = priceData.import[t % priceData.import.length]
    const exportPrice = priceData.export[t % priceData.export.length]

    problem.objective.vars.push({ name: `grid_import_${t}`, coef: importPrice })
    problem.objective.vars.push({ name: `grid_export_${t}`, coef: -exportPrice })
  }
}

function setupSelfConsumptionObjective(problem: any, timeHorizon: number, forecastData: ForecastDataPoint[]) {
  for (let t = 0; t < timeHorizon; t++) {
    // Minimize grid export (maximize self-consumption)
    problem.objective.vars.push({ name: `grid_export_${t}`, coef: 1 })
  }
}

function setupGridIndependenceObjective(problem: any, timeHorizon: number) {
  for (let t = 0; t < timeHorizon; t++) {
    // Minimize grid import and export (maximize grid independence)
    problem.objective.vars.push({ name: `grid_import_${t}`, coef: 1 })
    problem.objective.vars.push({ name: `grid_export_${t}`, coef: 1 })
  }
}

function setupBatteryLifeObjective(problem: any, timeHorizon: number, params: OptimizationParams) {
  // Minimize battery throughput while still meeting other constraints
  for (let t = 0; t < timeHorizon; t++) {
    problem.objective.vars.push({ name: `battery_charge_${t}`, coef: 1 })
    problem.objective.vars.push({ name: `battery_discharge_${t}`, coef: 1 })

    // Still consider cost as a secondary objective
    problem.objective.vars.push({ name: `grid_import_${t}`, coef: 0.1 })
  }
}

