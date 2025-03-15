import { NextResponse } from "next/server"
import { LinearProgrammingOptimizer, type OptimizationParams, type TimeSeriesData } from "../../../../lib/optimizer"
import { SolcastClient } from "../../../../lib/solcast-client"
import { HomeAssistantClient } from "../../../../lib/home-assistant-client"

// This API route runs the optimization algorithm
export async function POST(request: Request) {
  const body = await request.json()
  const {
    timeHorizon = 24,
    timeStep = 15,
    batteryCapacity = 10,
    batteryMaxPower = 5,
    batteryEfficiencyCharge = 95,
    batteryEfficiencyDischarge = 95,
    minSOC = 20,
    maxSOC = 95,
    maxGridPower = 5,
    gridImportCost = 0.25,
    gridExportPrice = 0.08,
    batteryDegradationCost = 0.05,
    pvCapacity = 5,
    solcastResourceId,
    optimizedDevices = [],
  } = body

  try {
    // Create optimization parameters
    const params: OptimizationParams = {
      timeHorizon,
      timeStep,
      batteryCapacity,
      batteryMaxPower,
      batteryEfficiencyCharge,
      batteryEfficiencyDischarge,
      minSOC,
      maxSOC,
      maxGridPower,
      gridImportCost,
      gridExportPrice,
      batteryDegradationCost,
      pvCapacity,
    }

    // Get PV forecast from Solcast
    let pvForecast: number[] = []
    if (solcastResourceId) {
      const solcastClient = new SolcastClient(solcastResourceId)
      pvForecast = await solcastClient.getForecastForOptimization(timeHorizon)
    } else {
      // Generate mock PV forecast if no Solcast resource ID is provided
      pvForecast = generateMockPVForecast(timeHorizon, timeStep, pvCapacity)
    }

    // Get current load data from Home Assistant or generate mock data
    let loadForecast: number[] = []
    try {
      const haClient = new HomeAssistantClient()
      const sensorData = await haClient.getSensorData()

      // Use current load as a basis for forecast
      const currentLoad = sensorData.gridPower + sensorData.solarPower - sensorData.batteryPower

      // Generate load forecast based on current load
      loadForecast = generateLoadForecast(currentLoad, timeHorizon, timeStep)
    } catch (error) {
      console.error("Error getting load data from Home Assistant:", error)
      // Generate mock load forecast if Home Assistant data is not available
      loadForecast = generateMockLoadForecast(timeHorizon, timeStep)
    }

    // Add optimized devices to load forecast
    if (optimizedDevices.length > 0) {
      // In a real implementation, you would incorporate device schedules into the load forecast
      // This is a simplified example
      console.log(`Incorporating ${optimizedDevices.length} devices into optimization`)
    }

    // Create time series data for optimization
    const timeSeriesData: TimeSeriesData = {
      loadForecast,
      pvForecast,
    }

    // Run optimization
    const optimizer = new LinearProgrammingOptimizer(params)
    const results = optimizer.optimize(timeSeriesData)

    return NextResponse.json(results)
  } catch (error) {
    console.error("Error running optimization:", error)
    return NextResponse.json({ error: "Failed to run optimization" }, { status: 500 })
  }
}

// Helper function to generate mock PV forecast
function generateMockPVForecast(timeHorizon: number, timeStep: number, pvCapacity: number): number[] {
  const intervals = (timeHorizon * 60) / timeStep
  const forecast: number[] = []

  // Get current hour to align the forecast with the time of day
  const currentHour = new Date().getHours()

  for (let i = 0; i < intervals; i++) {
    const intervalHour = (currentHour + (i * timeStep) / 60) % 24

    // Create a solar curve that peaks at noon
    let pvEstimate = 0
    if (intervalHour >= 6 && intervalHour <= 18) {
      // Simple bell curve centered at noon (hour 12)
      pvEstimate = pvCapacity * Math.sin((Math.PI * (intervalHour - 6)) / 12)

      // Add some randomness
      pvEstimate *= 0.8 + Math.random() * 0.4

      // Round to 2 decimal places
      pvEstimate = Math.round(pvEstimate * 100) / 100
    }

    forecast.push(pvEstimate)
  }

  return forecast
}

// Helper function to generate mock load forecast
function generateMockLoadForecast(timeHorizon: number, timeStep: number): number[] {
  const intervals = (timeHorizon * 60) / timeStep
  const forecast: number[] = []

  // Get current hour to align the forecast with the time of day
  const currentHour = new Date().getHours()

  for (let i = 0; i < intervals; i++) {
    const intervalHour = (currentHour + (i * timeStep) / 60) % 24

    // Create a load curve with morning and evening peaks
    let load = 1.0 // Base load

    // Morning peak (7-9 AM)
    if (intervalHour >= 7 && intervalHour <= 9) {
      load += 1.0 * Math.sin((Math.PI * (intervalHour - 7)) / 2)
    }

    // Evening peak (18-22 PM)
    if (intervalHour >= 18 && intervalHour <= 22) {
      load += 1.5 * Math.sin((Math.PI * (intervalHour - 18)) / 4)
    }

    // Add some randomness
    load *= 0.9 + Math.random() * 0.2

    // Round to 2 decimal places
    load = Math.round(load * 100) / 100

    forecast.push(load)
  }

  return forecast
}

// Helper function to generate load forecast based on current load
function generateLoadForecast(currentLoad: number, timeHorizon: number, timeStep: number): number[] {
  const intervals = (timeHorizon * 60) / timeStep
  const forecast: number[] = []

  // Get current hour to align the forecast with the time of day
  const currentHour = new Date().getHours()

  for (let i = 0; i < intervals; i++) {
    const intervalHour = (currentHour + (i * timeStep) / 60) % 24

    // Create a load curve with morning and evening peaks
    let loadFactor = 1.0 // Base load factor

    // Morning peak (7-9 AM)
    if (intervalHour >= 7 && intervalHour <= 9) {
      loadFactor += 0.5 * Math.sin((Math.PI * (intervalHour - 7)) / 2)
    }

    // Evening peak (18-22 PM)
    if (intervalHour >= 18 && intervalHour <= 22) {
      loadFactor += 0.8 * Math.sin((Math.PI * (intervalHour - 18)) / 4)
    }

    // Add some randomness
    loadFactor *= 0.9 + Math.random() * 0.2

    // Apply the factor to the current load
    let load = currentLoad * loadFactor

    // Round to 2 decimal places
    load = Math.round(load * 100) / 100

    forecast.push(load)
  }

  return forecast
}

