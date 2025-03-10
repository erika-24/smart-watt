"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Battery, Home, Sun, Zap, ArrowRight, ArrowDown, ArrowLeft, RefreshCw, AlertCircle } from "lucide-react"
import { fetchCurrentEnergyData } from "@/lib/api"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

export default function EnergyFlowDiagram() {
  const [energyFlow, setEnergyFlow] = useState<{
    solar: number
    battery: {
      charging: number
      discharging: number
      soc: number
    }
    grid: {
      import: number
      export: number
    }
    load: number
  } | null>(null)

  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Fetch energy flow data
  useEffect(() => {
    async function loadEnergyFlowData() {
      try {
        setIsLoading(true)
        const data = await fetchCurrentEnergyData()

        // Transform the data for the energy flow diagram
        const transformedData = {
          solar: data.reduce((max: number, item: any) => Math.max(max, item.solar), 0),
          battery: {
            charging:
              data.reduce((sum: number, item: any) => sum + (item.battery > 0 ? item.battery : 0), 0) / data.length,
            discharging:
              Math.abs(data.reduce((sum: number, item: any) => sum + (item.battery < 0 ? item.battery : 0), 0)) /
              data.length,
            soc: 85, // This would come from a separate API call in a real implementation
          },
          grid: {
            import: data.reduce((sum: number, item: any) => sum + (item.grid > 0 ? item.grid : 0), 0) / data.length,
            export:
              Math.abs(data.reduce((sum: number, item: any) => sum + (item.grid < 0 ? item.grid : 0), 0)) / data.length,
          },
          load: data.reduce((sum: number, item: any) => sum + item.consumption, 0) / data.length,
        }

        setEnergyFlow(transformedData)
        setError(null)
      } catch (err) {
        setError("Failed to load energy flow data")
        console.error(err)
      } finally {
        setIsLoading(false)
      }
    }

    loadEnergyFlowData()

    // Set up polling for real-time updates
    const interval = setInterval(loadEnergyFlowData, 30000) // Update every 30 seconds
    return () => clearInterval(interval)
  }, [])

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Energy Flow Diagram</CardTitle>
        <CardDescription>Real-time visualization of energy flows in your system</CardDescription>
      </CardHeader>
      <CardContent>
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {isLoading ? (
          <div className="h-[500px] relative bg-muted/20 rounded-lg p-4 flex items-center justify-center">
            <div className="flex flex-col items-center gap-2">
              <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
              <p className="text-sm text-muted-foreground">Loading energy flow data...</p>
            </div>
          </div>
        ) : !energyFlow ? (
          <div className="h-[500px] relative bg-muted/20 rounded-lg p-4 flex items-center justify-center">
            <p className="text-muted-foreground">No energy flow data available</p>
          </div>
        ) : (
          <div className="h-[500px] relative bg-muted/20 rounded-lg p-4 flex items-center justify-center">
            {/* Solar Panel */}
            <div className="absolute top-10 left-1/2 transform -translate-x-1/2 flex flex-col items-center">
              <div className="bg-blue-100 dark:bg-blue-950 p-4 rounded-lg border border-blue-200 dark:border-blue-800 flex items-center justify-center">
                <Sun className="h-12 w-12 text-yellow-500" />
              </div>
              <div className="text-center mt-2">
                <div className="font-medium">Solar PV</div>
                <div className="text-2xl font-bold">{energyFlow.solar.toFixed(1)} kW</div>
              </div>
              {energyFlow.solar > 0 && <ArrowDown className="h-8 w-8 text-blue-500 mt-2" />}
            </div>

            {/* Battery */}
            <div className="absolute left-10 top-1/2 transform -translate-y-1/2 flex flex-col items-center">
              <div className="bg-green-100 dark:bg-green-950 p-4 rounded-lg border border-green-200 dark:border-green-800 flex items-center justify-center">
                <Battery className="h-12 w-12 text-green-500" />
              </div>
              <div className="text-center mt-2">
                <div className="font-medium">Battery</div>
                <div className="text-2xl font-bold">{energyFlow.battery.soc}%</div>
                {energyFlow.battery.charging > 0 && (
                  <div className="text-sm text-green-500">+{energyFlow.battery.charging.toFixed(1)} kW</div>
                )}
                {energyFlow.battery.discharging > 0 && (
                  <div className="text-sm text-orange-500">-{energyFlow.battery.discharging.toFixed(1)} kW</div>
                )}
              </div>
              {energyFlow.battery.discharging > 0 && <ArrowRight className="h-8 w-8 text-orange-500 mt-2" />}
            </div>

            {/* Grid */}
            <div className="absolute right-10 top-1/2 transform -translate-y-1/2 flex flex-col items-center">
              <div className="bg-purple-100 dark:bg-purple-950 p-4 rounded-lg border border-purple-200 dark:border-purple-800 flex items-center justify-center">
                <Zap className="h-12 w-12 text-purple-500" />
              </div>
              <div className="text-center mt-2">
                <div className="font-medium">Grid</div>
                {energyFlow.grid.import > 0 && (
                  <div className="text-xl font-bold text-purple-500">+{energyFlow.grid.import.toFixed(1)} kW</div>
                )}
                {energyFlow.grid.export > 0 && (
                  <div className="text-xl font-bold text-blue-500">-{energyFlow.grid.export.toFixed(1)} kW</div>
                )}
              </div>
              {energyFlow.grid.import > 0 && <ArrowLeft className="h-8 w-8 text-purple-500 mt-2" />}
            </div>

            {/* Home/Load */}
            <div className="absolute bottom-10 left-1/2 transform -translate-x-1/2 flex flex-col items-center">
              <div className="bg-orange-100 dark:bg-orange-950 p-4 rounded-lg border border-orange-200 dark:border-orange-800 flex items-center justify-center">
                <Home className="h-12 w-12 text-orange-500" />
              </div>
              <div className="text-center mt-2">
                <div className="font-medium">Home Load</div>
                <div className="text-2xl font-bold">{energyFlow.load.toFixed(1)} kW</div>
              </div>
            </div>

            {/* Center Hub */}
            <div className="bg-gray-100 dark:bg-gray-800 p-6 rounded-full border border-gray-200 dark:border-gray-700 flex items-center justify-center z-10">
              <Zap className="h-8 w-8 text-primary" />
            </div>

            {/* Connection Lines - would be better with SVG in a real implementation */}
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className="w-[80%] h-[80%] border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-full opacity-30"></div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

