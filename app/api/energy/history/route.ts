import { NextResponse } from "next/server"

export async function GET(request: Request) {
  // Simulate API response delay
  await new Promise((resolve) => setTimeout(resolve, 500))

  // Get parameters from query
  const { searchParams } = new URL(request.url)
  const timeRange = searchParams.get("timeRange") || "day"
  const date = searchParams.get("date") || new Date().toISOString().split("T")[0]

  let data = []

  if (timeRange === "day") {
    // Generate hourly data for a day
    data = Array.from({ length: 24 }, (_, i) => {
      const hour = i.toString().padStart(2, "0") + ":00"
      const consumption = Math.random() * 1.5 + 0.5
      const solar = i >= 6 && i <= 18 ? Math.random() * 2.0 * Math.sin(((i - 6) * Math.PI) / 12) : 0
      const grid = Math.max(consumption - solar, 0)

      return {
        time: hour,
        consumption: Number.parseFloat(consumption.toFixed(1)),
        solar: Number.parseFloat(solar.toFixed(1)),
        grid: Number.parseFloat(grid.toFixed(1)),
      }
    })
  } else if (timeRange === "week") {
    // Generate daily data for a week
    const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    data = days.map((day) => {
      const consumption = Math.random() * 10 + 20
      const solar = Math.random() * 5 + 10
      const grid = consumption - solar

      return {
        date: day,
        consumption: Number.parseFloat(consumption.toFixed(1)),
        solar: Number.parseFloat(solar.toFixed(1)),
        grid: Number.parseFloat(grid.toFixed(1)),
      }
    })
  } else if (timeRange === "month") {
    // Generate weekly data for a month
    data = Array.from({ length: 4 }, (_, i) => {
      const weekNum = i + 1
      const consumption = Math.random() * 50 + 100
      const solar = Math.random() * 30 + 50
      const grid = consumption - solar

      return {
        date: `Week ${weekNum}`,
        consumption: Number.parseFloat(consumption.toFixed(1)),
        solar: Number.parseFloat(solar.toFixed(1)),
        grid: Number.parseFloat(grid.toFixed(1)),
      }
    })
  }

  return NextResponse.json(data)
}

