import { NextResponse } from "next/server"

export async function GET() {
  // Simulate API response delay
  await new Promise((resolve) => setTimeout(resolve, 500))

  // Generate 24 hours of data
  const data = Array.from({ length: 24 }, (_, i) => {
    const hour = i.toString().padStart(2, "0") + ":00"
    const consumption = Math.random() * 1.5 + 0.5 // Between 0.5 and 2.0
    const solar = i >= 6 && i <= 18 ? Math.random() * 2.0 * Math.sin(((i - 6) * Math.PI) / 12) : 0
    const battery = i >= 18 || i <= 6 ? -Math.min(consumption, 1.0) : Math.min(solar - consumption, 1.0)
    const grid = consumption - solar - battery

    return {
      time: hour,
      consumption: Number.parseFloat(consumption.toFixed(1)),
      solar: Number.parseFloat(solar.toFixed(1)),
      battery: Number.parseFloat(battery.toFixed(1)),
      grid: Number.parseFloat(grid.toFixed(1)),
    }
  })

  return NextResponse.json(data)
}

