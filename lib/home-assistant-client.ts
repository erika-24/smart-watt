// Home Assistant API client for communicating with Home Assistant

export interface HomeAssistantEntity {
  entity_id: string
  state: string
  attributes: Record<string, any>
  last_changed: string
  last_updated: string
}

export class HomeAssistantClient {
  private baseUrl: string

  constructor(baseUrl = "/api/homeassistant") {
    this.baseUrl = baseUrl
  }

  // Get all entities
  async getEntities(): Promise<HomeAssistantEntity[]> {
    try {
      const response = await fetch(`${this.baseUrl}`)

      if (!response.ok) {
        throw new Error(`Failed to get entities: ${response.statusText}`)
      }

      return response.json()
    } catch (error) {
      console.error("Error getting entities:", error)
      throw error
    }
  }

  // Get entity state
  async getEntityState(entityId: string): Promise<HomeAssistantEntity> {
    try {
      const response = await fetch(`${this.baseUrl}?entityId=${entityId}`)

      if (!response.ok) {
        throw new Error(`Failed to get entity state: ${response.statusText}`)
      }

      return response.json()
    } catch (error) {
      console.error("Error getting entity state:", error)
      throw error
    }
  }

  // Call service
  async callService(entityId: string, service: string, serviceData?: any): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          entityId,
          service,
          serviceData,
        }),
      })

      if (!response.ok) {
        throw new Error(`Failed to call service: ${response.statusText}`)
      }

      return response.json()
    } catch (error) {
      console.error("Error calling service:", error)
      throw error
    }
  }

  // Get sensor data for optimization
  async getSensorData(): Promise<Record<string, number>> {
    try {
      // Get relevant sensor data for optimization
      const [gridPower, solarPower, batteryPower, batterySOC] = await Promise.all([
        this.getEntityState("sensor.grid_power"),
        this.getEntityState("sensor.solar_power"),
        this.getEntityState("sensor.battery_power"),
        this.getEntityState("sensor.battery_soc"),
      ])

      return {
        gridPower: Number.parseFloat(gridPower.state),
        solarPower: Number.parseFloat(solarPower.state),
        batteryPower: Number.parseFloat(batteryPower.state),
        batterySOC: Number.parseFloat(batterySOC.state),
      }
    } catch (error) {
      console.error("Error getting sensor data:", error)
      throw error
    }
  }
}

