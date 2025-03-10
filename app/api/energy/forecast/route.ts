import { NextResponse } from "next/server"

export async function GET(request: Request) {
  // Simulate API response delay
  await new Promise((resolve) => setTimeout(resolve, 500))

  // Get date from query params
  const { searchParams } = new URL(request.url)
  const date = searchParams.get("date") || new Date().toISOString().split("T")[0]

  // Generate 24 hours of forecast data
  const data = Array.from({ length: 24 }, (_, i) => {
    const hour = i.toString().padStart(2, "0") + ":00"
    const load = Math.random() * 1.5 + 0.5 // Between 0.5 and 2.0
    const solar = i >= 6 && i <= 18 ? Math.random() * 2.0 * Math.sin(((i - 6) * Math.PI) / 12) : 0
    const optimizedLoad = Math.max(load * (1 - Math.random() * 0.2), 0.3) // Reduce load by up to 20%

    return {
      time: hour,
      load: Number.parseFloat(load.toFixed(1)),
      solar: Number.parseFloat(solar.toFixed(1)),
      optimizedLoad: Number.parseFloat(optimizedLoad.toFixed(1)),
    }
  })

  return NextResponse.json(data)
}

