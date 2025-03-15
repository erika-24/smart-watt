import type { Device, EnergyDataPoint } from "./types"

// Home Assistant API integration
export async function fetchHomeAssistantData(): Promise<{
  energyData: EnergyDataPoint
  devices: Device[]
}> {
  try {
    const haUrl = process.env.HOME_ASSISTANT_URL
    const haToken = process.env.HOME_ASSISTANT_TOKEN

    if (!haUrl || !haToken) {
      throw new Error("HOME_ASSISTANT_URL or HOME_ASSISTANT_TOKEN environment variables are not set")
    }

    // Fetch current energy data
    const energyResponse = await fetch(`${haUrl}/api/states`, {
      headers: {
        Authorization: `Bearer ${haToken}`,
        "Content-Type": "application/json",
      },
    })

    if (!energyResponse.ok) {
      throw new Error(`Home Assistant API error: ${energyResponse.status} ${energyResponse.statusText}`)
    }

    const states = await energyResponse.json()

    // Extract energy data from states
    const loadPower = findEntityState(states, "sensor.home_load_power")
    const pvPower = findEntityState(states, "sensor.solar_power")
    const batteryPower = findEntityState(states, "sensor.battery_power")
    const batterySoc = findEntityState(states, "sensor.battery_soc")
    const gridPower = findEntityState(states, "sensor.grid_power")

    const energyData: EnergyDataPoint = {
      timestamp: new Date().toISOString(),
      load_power: Number.parseFloat(loadPower?.state || "0"),
      pv_power: Number.parseFloat(pvPower?.state || "0"),
      battery_power: Number.parseFloat(batteryPower?.state || "0"),
      battery_soc: Number.parseFloat(batterySoc?.state || "0"),
      grid_power: Number.parseFloat(gridPower?.state || "0"),
    }

    // Fetch devices
    const devicesResponse = await fetch(`${haUrl}/api/states`, {
      headers: {
        Authorization: `Bearer ${haToken}`,
        "Content-Type": "application/json",
      },
    })

    if (!devicesResponse.ok) {
      throw new Error(`Home Assistant API error: ${devicesResponse.status} ${devicesResponse.statusText}`)
    }

    const deviceStates = await devicesResponse.json()

    // Extract device data
    const devices: Device[] = extractDevices(deviceStates)

    return { energyData, devices }
  } catch (error) {
    console.error("Error fetching Home Assistant data:", error)
    return {
      energyData: createDefaultEnergyData(),
      devices: [],
    }
  }
}

// Helper function to find entity state
function findEntityState(states: any[], entityId: string) {
  return states.find((state) => state.entity_id === entityId)
}

// Helper function to extract devices from states
function extractDevices(states: any[]): Device[] {
  const devices: Device[] = []
  const switchEntities = states.filter(
    (state) =>
      state.entity_id.startsWith("switch.") ||
      state.entity_id.startsWith("light.") ||
      state.entity_id.startsWith("climate."),
  )

  for (const entity of switchEntities) {
    const id = entity.entity_id
    const name = entity.attributes.friendly_name || id
    const isOn = entity.state === "on"

    let type = "other"
    let icon = "plug"
    let power = 0

    if (id.startsWith("light.")) {
      type = "lighting"
      icon = "lightbulb"
      power = isOn ? 60 : 0 // Estimate
    } else if (id.startsWith("climate.")) {
      type = "hvac"
      icon = "thermometer"
      power = isOn ? 1000 : 0 // Estimate
    } else if (id.includes("washer") || id.includes("dryer") || id.includes("dishwasher")) {
      type = "appliance"
      icon = "washing-machine"
      power = isOn ? 800 : 0 // Estimate
    } else if (id.includes("charger") || id.includes("ev")) {
      type = "charger"
      icon = "car"
      power = isOn ? 7200 : 0 // Estimate
    }

    devices.push({
      id,
      name,
      type,
      status: "online",
      power,
      isOn,
      icon,
      shiftable: ["appliance", "charger"].includes(type),
    })
  }

  return devices
}

// Create default energy data when API fails
function createDefaultEnergyData(): EnergyDataPoint {
  return {
    timestamp: new Date().toISOString(),
    load_power: 1.5,
    pv_power: 2.0,
    battery_power: 0.5,
    battery_soc: 50,
    grid_power: -1.0,
  }
}

