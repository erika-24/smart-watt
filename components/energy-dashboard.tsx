"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Calendar } from "@/components/ui/calendar"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Line,
  BarChart,
  Bar,
  ComposedChart,
} from "recharts"
import { CalendarDays, TrendingUp, Sun, Zap, DollarSign, RefreshCw } from "lucide-react"
import { cn } from "@/lib/utils"
import { format } from "date-fns"
import {
  fetchCurrentEnergyData,
  fetchForecastData,
  fetchHistoricalData,
  fetchEnergyBreakdown,
  type EnergyData,
  type ForecastData,
} from "@/lib/python-api-adapter"
import { Skeleton } from "@/components/ui/skeleton"

export default function EnergyDashboard() {
  const [date, setDate] = useState<Date | undefined>(new Date())
  const [timeRange, setTimeRange] = useState("day")

  const [currentData, setCurrentData] = useState<EnergyData[]>([])
  const [forecastData, setForecastData] = useState<ForecastData[]>([])
  const [historicalData, setHistoricalData] = useState<any[]>([])
  const [energySources, setEnergySources] = useState<any[]>([])
  const [energyConsumption, setEnergyConsumption] = useState<any[]>([])

  const [isLoadingCurrent, setIsLoadingCurrent] = useState(true)
  const [isLoadingForecast, setIsLoadingForecast] = useState(true)
  const [isLoadingHistorical, setIsLoadingHistorical] = useState(true)
  const [isLoadingBreakdown, setIsLoadingBreakdown] = useState(true)

  const [error, setError] = useState<string | null>(null)

  // Series visibility toggles
  const [visibleSeries, setVisibleSeries] = useState({
    forecastedLoad: true,
    optimizedLoad: true,
    solar: true,
    battery: true,
    grid: true,
  })

  // Toggle a specific series visibility
  const toggleSeries = (series: keyof typeof visibleSeries) => {
    setVisibleSeries((prev) => ({
      ...prev,
      [series]: !prev[series],
    }))
  }

  // Fetch current energy data
  useEffect(() => {
    async function loadCurrentData() {
      try {
        setIsLoadingCurrent(true)
        const data = await fetchCurrentEnergyData()
        setCurrentData(data)
        setError(null)
      } catch (err) {
        setError("Failed to load current energy data")
        console.error(err)
      } finally {
        setIsLoadingCurrent(false)
      }
    }

    loadCurrentData()
    // Set up polling for real-time updates
    const interval = setInterval(loadCurrentData, 60000) // Update every minute
    return () => clearInterval(interval)
  }, [])

  // Fetch forecast data when date changes
  useEffect(() => {
    async function loadForecastData() {
      if (!date) return

      try {
        setIsLoadingForecast(true)
        const formattedDate = format(date, "yyyy-MM-dd")
        const data = await fetchForecastData(formattedDate)
        setForecastData(data)
        setError(null)
      } catch (err) {
        setError("Failed to load forecast data")
        console.error(err)
      } finally {
        setIsLoadingForecast(false)
      }
    }

    loadForecastData()
  }, [date])

  // Fetch historical data when date or time range changes
  useEffect(() => {
    async function loadHistoricalData() {
      if (!date) return

      try {
        setIsLoadingHistorical(true)
        const formattedDate = format(date, "yyyy-MM-dd")
        const data = await fetchHistoricalData(timeRange, formattedDate)
        setHistoricalData(data)
        setError(null)
      } catch (err) {
        setError("Failed to load historical data")
        console.error(err)
      } finally {
        setIsLoadingHistorical(false)
      }
    }

    loadHistoricalData()
  }, [date, timeRange])

  // Fetch energy breakdown data
  useEffect(() => {
    async function loadBreakdownData() {
      try {
        setIsLoadingBreakdown(true)
        const data = await fetchEnergyBreakdown()
        setEnergySources(data.sources)
        setEnergyConsumption(data.consumption)
        setError(null)
      } catch (err) {
        setError("Failed to load energy breakdown data")
        console.error(err)
      } finally {
        setIsLoadingBreakdown(false)
      }
    }

    loadBreakdownData()
  }, [])

  // Combine current data with forecast data for the current chart
  const combinedCurrentData = currentData.map((item) => {
    const forecastItem = forecastData.find((f) => f.time === item.time)
    return {
      ...item,
      forecastedLoad: forecastItem?.load || null,
      optimizedLoad: forecastItem?.optimizedLoad || null,
    }
  })

  // Calculate summary metrics from forecast data
  const forecastSummary =
    forecastData.length > 0
      ? {
          totalLoad: forecastData.reduce((sum, item) => sum + item.load, 0).toFixed(1),
          totalSolar: forecastData.reduce((sum, item) => sum + item.solar, 0).toFixed(1),
          totalOptimizedLoad: forecastData.reduce((sum, item) => sum + item.optimizedLoad, 0).toFixed(1),
          savingsPercent: (
            (1 -
              forecastData.reduce((sum, item) => sum + item.optimizedLoad, 0) /
                forecastData.reduce((sum, item) => sum + item.load, 0)) *
            100
          ).toFixed(0),
          estimatedSavings: (
            (forecastData.reduce((sum, item) => sum + item.load, 0) -
              forecastData.reduce((sum, item) => sum + item.optimizedLoad, 0)) *
            0.15
          ).toFixed(2),
        }
      : null

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Energy Consumption</h2>
        <div className="flex items-center gap-2">
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-[150px]">
              <SelectValue placeholder="Select time range" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="day">Day</SelectItem>
              <SelectItem value="week">Week</SelectItem>
              <SelectItem value="month">Month</SelectItem>
            </SelectContent>
          </Select>
          <Popover>
            <PopoverTrigger asChild>
              <Button
                variant={"outline"}
                className={cn("w-[180px] justify-start text-left font-normal", !date && "text-muted-foreground")}
              >
                <CalendarDays className="mr-2 h-4 w-4" />
                {date ? format(date, "PPP") : "Select date"}
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-auto p-0" align="end" sideOffset={12}>
              <Calendar
                mode="single"
                selected={date}
                onSelect={setDate}
                disabled={(date) => date > new Date() || date < new Date("2020-01-01")}
                initialFocus
              />
            </PopoverContent>
          </Popover>
        </div>
      </div>

      {error && <div className="bg-destructive/15 text-destructive p-3 rounded-md">{error}</div>}

      <Tabs defaultValue="current">
        <TabsList>
          <TabsTrigger value="current">Current Usage</TabsTrigger>
          <TabsTrigger value="forecast">Forecast</TabsTrigger>
          <TabsTrigger value="history">Historical</TabsTrigger>
        </TabsList>

        <TabsContent value="current" className="space-y-4 mt-4">
          <Card>
            <CardHeader className="flex flex-row items-start justify-between">
              <div>
                <CardTitle>Current Energy Flow</CardTitle>
                <CardDescription>Real-time energy usage and production</CardDescription>
              </div>
              <div className="flex flex-col gap-2 mt-1">
                <div className="text-sm font-medium mb-1">Toggle Series:</div>
                <div className="flex flex-wrap gap-3">
                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="toggle-forecasted"
                      checked={visibleSeries.forecastedLoad}
                      onCheckedChange={() => toggleSeries("forecastedLoad")}
                      className="data-[state=checked]:bg-[#2196F3] data-[state=checked]:text-primary-foreground border-[#2196F3]"
                    />
                    <label
                      htmlFor="toggle-forecasted"
                      className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                    >
                      Forecasted Load
                    </label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="toggle-optimized"
                      checked={visibleSeries.optimizedLoad}
                      onCheckedChange={() => toggleSeries("optimizedLoad")}
                      className="data-[state=checked]:bg-[#8BC34A] data-[state=checked]:text-primary-foreground border-[#8BC34A]"
                    />
                    <label
                      htmlFor="toggle-optimized"
                      className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                    >
                      Optimized Load
                    </label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="toggle-solar"
                      checked={visibleSeries.solar}
                      onCheckedChange={() => toggleSeries("solar")}
                      className="data-[state=checked]:bg-[#FFD700] data-[state=checked]:text-primary-foreground border-[#FFD700]"
                    />
                    <label
                      htmlFor="toggle-solar"
                      className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                    >
                      Solar
                    </label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="toggle-battery"
                      checked={visibleSeries.battery}
                      onCheckedChange={() => toggleSeries("battery")}
                      className="data-[state=checked]:bg-[#4CAF50] data-[state=checked]:text-primary-foreground border-[#4CAF50]"
                    />
                    <label
                      htmlFor="toggle-battery"
                      className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                    >
                      Battery
                    </label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="toggle-grid"
                      checked={visibleSeries.grid}
                      onCheckedChange={() => toggleSeries("grid")}
                      className="data-[state=checked]:bg-[#9C27B0] data-[state=checked]:text-primary-foreground border-[#9C27B0]"
                    />
                    <label
                      htmlFor="toggle-grid"
                      className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                    >
                      Grid
                    </label>
                  </div>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {isLoadingCurrent ? (
                <div className="h-[400px] flex items-center justify-center">
                  <div className="flex flex-col items-center gap-2">
                    <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
                    <p className="text-sm text-muted-foreground">Loading energy data...</p>
                  </div>
                </div>
              ) : combinedCurrentData.length === 0 ? (
                <div className="h-[400px] flex items-center justify-center">
                  <p className="text-muted-foreground">No energy data available</p>
                </div>
              ) : (
                <div className="h-[400px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={combinedCurrentData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis />
                      <Tooltip />
                      <Legend />

                      {visibleSeries.solar && (
                        <Area
                          type="monotone"
                          dataKey="solar"
                          stackId="2"
                          stroke="#FFD700"
                          fill="#FFD700"
                          name="Solar"
                        />
                      )}

                      {visibleSeries.battery && (
                        <Area
                          type="monotone"
                          dataKey="battery"
                          stackId="3"
                          stroke="#4CAF50"
                          fill="#4CAF50"
                          name="Battery"
                        />
                      )}

                      {visibleSeries.grid && (
                        <Area type="monotone" dataKey="grid" stackId="4" stroke="#9C27B0" fill="#9C27B0" name="Grid" />
                      )}

                      {visibleSeries.forecastedLoad && (
                        <Line
                          type="monotone"
                          dataKey="forecastedLoad"
                          stroke="#2196F3"
                          name="Forecasted Load"
                          dot={false}
                        />
                      )}

                      {visibleSeries.optimizedLoad && (
                        <Line
                          type="monotone"
                          dataKey="optimizedLoad"
                          stroke="#8BC34A"
                          strokeDasharray="5 5"
                          name="Optimized Load"
                          dot={false}
                        />
                      )}
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              )}
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Energy Sources</CardTitle>
                <CardDescription>Breakdown of energy sources</CardDescription>
              </CardHeader>
              <CardContent>
                {isLoadingBreakdown ? (
                  <div className="h-[300px] flex items-center justify-center">
                    <div className="flex flex-col items-center gap-2">
                      <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
                      <p className="text-sm text-muted-foreground">Loading breakdown data...</p>
                    </div>
                  </div>
                ) : energySources.length === 0 ? (
                  <div className="h-[300px] flex items-center justify-center">
                    <p className="text-muted-foreground">No energy source data available</p>
                  </div>
                ) : (
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={energySources}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="value" fill="#8884d8" name="kWh" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Energy Consumption</CardTitle>
                <CardDescription>Breakdown by category</CardDescription>
              </CardHeader>
              <CardContent>
                {isLoadingBreakdown ? (
                  <div className="h-[300px] flex items-center justify-center">
                    <div className="flex flex-col items-center gap-2">
                      <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
                      <p className="text-sm text-muted-foreground">Loading consumption data...</p>
                    </div>
                  </div>
                ) : energyConsumption.length === 0 ? (
                  <div className="h-[300px] flex items-center justify-center">
                    <p className="text-muted-foreground">No consumption data available</p>
                  </div>
                ) : (
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={energyConsumption}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="value" fill="#82ca9d" name="kWh" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="forecast" className="space-y-4 mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Energy Forecast</CardTitle>
              <CardDescription>Predicted energy production and consumption</CardDescription>
            </CardHeader>
            <CardContent>
              {isLoadingForecast ? (
                <div className="h-[400px] flex items-center justify-center">
                  <div className="flex flex-col items-center gap-2">
                    <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
                    <p className="text-sm text-muted-foreground">Loading forecast data...</p>
                  </div>
                </div>
              ) : forecastData.length === 0 ? (
                <div className="h-[400px] flex items-center justify-center">
                  <p className="text-muted-foreground">No forecast data available</p>
                </div>
              ) : (
                <div className="h-[400px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={forecastData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Area type="monotone" dataKey="solar" fill="#FFD700" stroke="#FFD700" name="Forecasted Solar" />
                      <Line type="monotone" dataKey="load" stroke="#2196F3" name="Forecasted Load" />
                      <Line
                        type="monotone"
                        dataKey="optimizedLoad"
                        stroke="#8BC34A"
                        strokeDasharray="5 5"
                        name="Optimized Load"
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              )}
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Forecasted Load</CardTitle>
                <TrendingUp className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                {isLoadingForecast ? (
                  <Skeleton className="h-8 w-24" />
                ) : forecastSummary ? (
                  <>
                    <div className="text-2xl font-bold">{forecastSummary.totalLoad} kWh</div>
                    <p className="text-xs text-muted-foreground">For {date ? format(date, "PPP") : "today"}</p>
                  </>
                ) : (
                  <p className="text-muted-foreground">No data available</p>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Forecasted Solar</CardTitle>
                <Sun className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                {isLoadingForecast ? (
                  <Skeleton className="h-8 w-24" />
                ) : forecastSummary ? (
                  <>
                    <div className="text-2xl font-bold">{forecastSummary.totalSolar} kWh</div>
                    <p className="text-xs text-muted-foreground">Based on weather forecast</p>
                  </>
                ) : (
                  <p className="text-muted-foreground">No data available</p>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Optimized Load</CardTitle>
                <Zap className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                {isLoadingForecast ? (
                  <Skeleton className="h-8 w-24" />
                ) : forecastSummary ? (
                  <>
                    <div className="text-2xl font-bold">{forecastSummary.totalOptimizedLoad} kWh</div>
                    <p className="text-xs text-muted-foreground">{forecastSummary.savingsPercent}% reduction</p>
                  </>
                ) : (
                  <p className="text-muted-foreground">No data available</p>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Estimated Savings</CardTitle>
                <DollarSign className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                {isLoadingForecast ? (
                  <Skeleton className="h-8 w-24" />
                ) : forecastSummary ? (
                  <>
                    <div className="text-2xl font-bold">${forecastSummary.estimatedSavings}</div>
                    <p className="text-xs text-muted-foreground">Based on current rates</p>
                  </>
                ) : (
                  <p className="text-muted-foreground">No data available</p>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="history" className="space-y-4 mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Historical Energy Usage</CardTitle>
              <CardDescription>
                {timeRange === "day" ? "Hourly" : timeRange === "week" ? "Daily" : "Weekly"} energy consumption and
                production
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isLoadingHistorical ? (
                <div className="h-[400px] flex items-center justify-center">
                  <div className="flex flex-col items-center gap-2">
                    <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
                    <p className="text-sm text-muted-foreground">Loading historical data...</p>
                  </div>
                </div>
              ) : historicalData.length === 0 ? (
                <div className="h-[400px] flex items-center justify-center">
                  <p className="text-muted-foreground">No historical data available</p>
                </div>
              ) : (
                <div className="h-[400px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={historicalData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey={timeRange === "day" ? "time" : "date"} />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="consumption" fill="#2196F3" name="Consumption" />
                      <Bar dataKey="solar" fill="#FFD700" name="Solar" />
                      <Bar dataKey="grid" fill="#9C27B0" name="Grid" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

