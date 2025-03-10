// ESPHome API client for communicating with ESPHome devices

export interface ESPHomeDevice {
  id: string
  name: string
  type: string
  state: boolean
  value?: number
}

export class ESPHomeClient {
  private baseUrl: string

  constructor(baseUrl = "/api/esphome") {
    this.baseUrl = baseUrl
  }

  // Get device status
  async getDeviceStatus(deviceId: string): Promise<ESPHomeDevice> {
    try {
      const response = await fetch(`${this.baseUrl}?deviceId=${deviceId}`)

      if (!response.ok) {
        throw new Error(`Failed to get device status: ${response.statusText}`)
      }

      return response.json()
    } catch (error) {
      console.error("Error getting device status:", error)
      throw error
    }
  }

  // Control device
  async controlDevice(deviceId: string, state: boolean, value?: number): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          deviceId,
          state,
          value,
        }),
      })

      if (!response.ok) {
        throw new Error(`Failed to control device: ${response.statusText}`)
      }

      return response.json()
    } catch (error) {
      console.error("Error controlling device:", error)
      throw error
    }
  }
}

