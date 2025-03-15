import { NextResponse } from "next/server"
import { ESPHomeClient } from "@/lib/esphome-client"
import { HomeAssistantClient } from "@/lib/home-assistant-client"

// This API route executes the optimized schedule by controlling devices
export async function POST(request: Request) {
  const body = await request.json()
  const { schedule, devices } = body

  if (!schedule || !devices || !Array.isArray(devices)) {
    return NextResponse.json({ error: "Schedule and devices are required" }, { status: 400 })
  }

  try {
    const esphomeClient = new ESPHomeClient()
    const haClient = new HomeAssistantClient()

    // Execute the schedule by controlling devices
    const results = await Promise.all(
      devices.map(async (device: any) => {
        const { id, type, state, value } = device

        if (type === "esphome") {
          // Control ESPHome device
          return esphomeClient.controlDevice(id, state, value)
        } else if (type === "homeassistant") {
          // Control Home Assistant entity
          const service = state ? "switch.turn_on" : "switch.turn_off"
          return haClient.callService(id, service, value ? { brightness: value } : undefined)
        } else {
          throw new Error(`Unknown device type: ${type}`)
        }
      }),
    )

    return NextResponse.json({
      success: true,
      message: `Successfully executed schedule for ${results.length} devices`,
      results,
    })
  } catch (error) {
    console.error("Error executing schedule:", error)
    return NextResponse.json({ error: "Failed to execute schedule" }, { status: 500 })
  }
}

