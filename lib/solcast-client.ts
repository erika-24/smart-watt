// Solcast API client for fetching PV forecasts

export interface SolcastForecast {
  period_end: string
  period: string
  pv_estimate: number
  pv_estimate10: number
  pv_estimate90: number
}

export interface SolcastResponse {
  forecasts: SolcastForecast[]
  metadata: Record<string, any>
}

export class SolcastClient {
  private baseUrl: string
  private resourceId: string

  constructor(resourceId: string, baseUrl = "/api/solcast") {
    this.baseUrl = baseUrl
    this.resourceId = resourceId
  }

  // Get PV forecast
  async getForecast(period = "PT30M"): Promise<SolcastResponse> {
    try {
      const response = await fetch(`${this.baseUrl}?resourceId=${this.resourceId}&period=${period}`)

      if (!response.ok) {
        throw new Error(`Failed to get forecast: ${response.statusText}`)
      }

      return response.json()
    } catch (error) {
      console.error("Error getting forecast:", error)
      throw error
    }
  }

  // Get forecast data formatted for optimization
  async getForecastForOptimization(hours = 24): Promise<number[]> {
    try {
      const { forecasts } = await this.getForecast()

      // Calculate how many intervals we need based on the period
      // Assuming 30-minute intervals (PT30M)
      const intervals = hours * 2

      // Extract PV estimates for the specified number of hours
      return forecasts.slice(0, intervals).map((forecast) => forecast.pv_estimate)
    } catch (error) {
      console.error("Error getting forecast for optimization:", error)
      throw error
    }
  }
}

