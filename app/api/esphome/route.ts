import { NextResponse } from "next/server"

// This API route handles communication with ESPHome devices
export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const deviceId = searchParams.get("deviceId")

  if (!deviceId) {
    return NextResponse.json({ error: "Device ID is required" }, { status: 400 })
  }

  try {
    // Fetch device status from ESPHome
    const deviceStatus = await fetchESPHomeDeviceStatus(deviceId)
    return NextResponse.json(deviceStatus)
  } catch (error) {
    console.error("Error fetching ESPHome device status:", error)
    return NextResponse.json({ error: "Failed to fetch device status" }, { status: 500 })
  }
}

export async function POST(request: Request) {
  const body = await request.json()
  const { deviceId, state, value } = body

  if (!deviceId || state === undefined) {
    return NextResponse.json({ error: "Device ID and state are required" }, { status: 400 })
  }

  try {
    // Control ESPHome device
    const result = await controlESPHomeDevice(deviceId, state, value)
    return NextResponse.json(result)
  } catch (error) {
    console.error("Error controlling ESPHome device:", error)
    return NextResponse.json({ error: "Failed to control device" }, { status: 500 })
  }
}

// Helper function to fetch ESPHome device status
async function fetchESPHomeDeviceStatus(deviceId: string) {
  // In a real implementation, you would use the ESPHome API to fetch device status
  // This is a simplified example
  const esphomeApiUrl = process.env.ESPHOME_API_URL

  if (!esphomeApiUrl) {
    throw new Error("ESPHOME_API_URL environment variable is not set")
  }

  const response = await fetch(`${esphomeApiUrl}/device/${deviceId}`, {
    headers: {
      Accept: "application/json",
    },
  })

  if (!response.ok) {
    throw new Error(`Failed to fetch device status: ${response.statusText}`)
  }

  return response.json()
}

// Helper function to control ESPHome device
async function controlESPHomeDevice(deviceId: string, state: boolean, value?: number) {
  // In a real implementation, you would use the ESPHome API to control the device
  // This is a simplified example
  const esphomeApiUrl = process.env.ESPHOME_API_URL

  if (!esphomeApiUrl) {
    throw new Error("ESPHOME_API_URL environment variable is not set")
  }

  const payload = {
    state,
    value,
  }

  const response = await fetch(`${esphomeApiUrl}/device/${deviceId}/control`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify(payload),
  })

  if (!response.ok) {
    throw new Error(`Failed to control device: ${response.statusText}`)
  }

  return response.json()
}

