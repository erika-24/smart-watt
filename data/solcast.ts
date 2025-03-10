import type { ForecastDataPoint } from "./types"

// Solcast API integration for solar forecasts
export async function fetchSolarForecast(): Promise<ForecastDataPoint[]> {
  try {
    const apiKey = process.env.SOLCAST_API_KEY

    if (!apiKey) {
      throw new Error("SOLCAST_API_KEY environment variable is not set")
    }

    // Fetch solar forecast from Solcast API
    const response = await fetch(
      `https://api.solcast.com.au/rooftop_sites/your-site-id/forecasts?format=json&api_key=${apiKey}`,
    )

    if (!response.ok) {
      throw new Error(`Solcast API error: ${response.status} ${response.statusText}`)
    }

    const data = await response.json()

    // Transform Solcast data to our format
    return data.forecasts.map((forecast: any) => ({
      timestamp: forecast.period_end,
      pv_power: forecast.pv_estimate, // kW
      load_power: 0, // We'll fill this in later with load forecast
    }))
  } catch (error) {
    console.error("Error fetching solar forecast:", error)
    return []
  }
}

