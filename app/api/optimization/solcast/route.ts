import { NextResponse } from "next/server"

// This API route handles communication with Solcast API for PV forecasting
export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const resourceId = searchParams.get("resourceId")
  const period = searchParams.get("period") || "PT30M" // Default to 30-minute intervals

  if (!resourceId) {
    return NextResponse.json({ error: "Resource ID is required" }, { status: 400 })
  }

  try {
    // Fetch PV forecast from Solcast
    const forecast = await fetchSolcastForecast(resourceId, period)
    return NextResponse.json(forecast)
  } catch (error) {
    console.error("Error fetching Solcast forecast:", error)
    return NextResponse.json({ error: "Failed to fetch forecast" }, { status: 500 })
  }
}

// Helper function to fetch Solcast PV forecast
async function fetchSolcastForecast(resourceId: string, period: string) {
  const solcastApiKey = process.env.SOLCAST_API_KEY

  if (!solcastApiKey) {
    throw new Error("SOLCAST_API_KEY environment variable is not set")
  }

  const url = new URL(`https://api.solcast.com.au/rooftop_sites/${resourceId}/forecasts`)
  url.searchParams.append("format", "json")
  url.searchParams.append("period", period)
  url.searchParams.append("api_key", solcastApiKey)

  const response = await fetch(url.toString(), {
    headers: {
      Accept: "application/json",
    },
  })

  if (!response.ok) {
    throw new Error(`Failed to fetch Solcast forecast: ${response.statusText}`)
  }

  const data = await response.json()

  // Transform the data into a more usable format for our application
  return {
    forecasts: data.forecasts.map((forecast: any) => ({
      period_end: forecast.period_end,
      period: forecast.period,
      pv_estimate: forecast.pv_estimate,
      pv_estimate10: forecast.pv_estimate10,
      pv_estimate90: forecast.pv_estimate90,
    })),
    metadata: data.metadata,
  }
}

