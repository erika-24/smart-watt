import { NextResponse } from "next/server"

export async function POST(request: Request) {
  // Simulate API response delay and optimization processing time
  await new Promise((resolve) => setTimeout(resolve, 2000))

  const params = await request.json()

  // Generate optimization results based on the parameters
  const optimizationId = `opt-${Date.now()}`
  const timestamp = new Date().toISOString()

  // Calculate metrics based on optimization parameters
  const cost = Number.parseFloat((Math.random() * 2 + 2).toFixed(2))
  const selfConsumption = Math.floor(Math.random() * 20 + 70)
  const peakGridPower = Number.parseFloat((Math.random() * 1 + 1).toFixed(1))
  const batteryCycles = Number.parseFloat((Math.random() * 0.5 + 0.5).toFixed(1))

  // Generate schedule data
  const scheduleData = Array.from({ length: 24 }, (_, i) => {
    const hour = i.toString().padStart(2, "0") + ":00"
    const load = Math.random() * 1.5 + 0.5
    const solar = i >= 6 && i <= 18 ? Math.random() * 2.0 * Math.sin(((i - 6) * Math.PI) / 12) : 0

    // Adjust battery and grid usage based on optimization parameters
    let battery = 0
    let grid = 0
    let optimizedLoad = load

    if (params.batteryConstraints.enabled) {
      // Battery discharges at night, charges during solar production
      battery = i >= 18 || i <= 6 ? -Math.min(load * 0.8, 1.0) : Math.min(solar - load * 0.8, 1.0)

      // Respect battery constraints
      battery = Math.max(battery, -1.5) // Limit discharge rate
      battery = Math.min(battery, 1.5) // Limit charge rate
    }

    if (params.optimizationMode === "cost") {
      // Reduce load during peak hours (assuming peak is 17:00-21:00)
      optimizedLoad = i >= 17 && i <= 21 ? load * 0.8 : load
    } else if (params.optimizationMode === "self_consumption") {
      // Try to match load with solar production
      optimizedLoad = Math.min(load, solar + Math.abs(battery))
    }

    // Calculate grid usage
    grid = optimizedLoad - solar - battery

    // Apply grid constraints if enabled
    if (params.gridConstraints.enabled && grid > params.gridConstraints.maxPower) {
      grid = params.gridConstraints.maxPower
      optimizedLoad = grid + solar + battery
    }

    return {
      time: hour,
      load: Number.parseFloat(load.toFixed(1)),
      solar: Number.parseFloat(solar.toFixed(1)),
      battery: Number.parseFloat(battery.toFixed(1)),
      grid: Number.parseFloat(grid.toFixed(1)),
      optimizedLoad: Number.parseFloat(optimizedLoad.toFixed(1)),
    }
  })

  // Generate device schedules based on optimization
  const deviceScheduleData = [
    {
      id: "1",
      name: "EV Charger",
      type: "charger",
      schedule: [
        {
          start: params.optimizationMode === "cost" ? "01:00" : "22:00",
          end: params.optimizationMode === "cost" ? "05:00" : "02:00",
          power: 7.2,
        },
      ],
    },
    {
      id: "2",
      name: "Washing Machine",
      type: "appliance",
      schedule: [
        {
          start: params.optimizationMode === "self_consumption" ? "12:00" : "14:00",
          end: params.optimizationMode === "self_consumption" ? "13:30" : "15:30",
          power: 0.8,
        },
      ],
    },
    {
      id: "3",
      name: "Dishwasher",
      type: "appliance",
      schedule: [
        {
          start: params.optimizationMode === "cost" ? "22:00" : "13:00",
          end: params.optimizationMode === "cost" ? "23:30" : "14:30",
          power: 0.7,
        },
      ],
    },
    {
      id: "4",
      name: "Water Heater",
      type: "heating",
      schedule: [
        {
          start: params.optimizationMode === "self_consumption" ? "10:00" : "06:00",
          end: params.optimizationMode === "self_consumption" ? "11:00" : "07:00",
          power: 1.2,
        },
        {
          start: params.optimizationMode === "self_consumption" ? "14:00" : "18:00",
          end: params.optimizationMode === "self_consumption" ? "15:00" : "19:00",
          power: 1.2,
        },
      ],
    },
  ]

  return NextResponse.json({
    id: optimizationId,
    cost,
    selfConsumption,
    peakGridPower,
    batteryCycles,
    timestamp,
    scheduleData,
    deviceScheduleData,
  })
}

