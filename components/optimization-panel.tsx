"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Bar, ComposedChart } from "recharts"
import {
  LineChart,
  Battery,
  Sun,
  Home,
  DollarSign,
  Zap,
  Clock,
  Calendar,
  Play,
  Save,
  RefreshCw,
  AlertCircle,
} from "lucide-react"
import { runOptimization, applyOptimizationSchedule, type OptimizationResult } from "@/lib/api"
import { Skeleton } from "@/components/ui/skeleton"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

export default function OptimizationPanel() {
  const [optimizationMode, setOptimizationMode] = useState("cost")
  const [timeHorizon, setTimeHorizon] = useState("24")
  const [batteryUsage, setBatteryUsage] = useState(true)
  const [batteryMinSoc, setBatteryMinSoc] = useState([20])
  const [batteryMaxCycles, setBatteryMaxCycles] = useState([1])
  const [gridLimits, setGridLimits] = useState(true)
  const [maxGridPower, setMaxGridPower] = useState([5])

  const [isOptimizing, setIsOptimizing] = useState(false)
  const [isApplying, setIsApplying] = useState(false)
  const [optimizationResults, setOptimizationResults] = useState<OptimizationResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)

  const runOptimizationProcess = async () => {
    setIsOptimizing(true)
    setError(null)

    try {
      const params = {
        optimizationMode,
        timeHorizon,
        batteryConstraints: {
          enabled: batteryUsage,
          minSoc: batteryMinSoc[0],
          maxCycles: batteryMaxCycles[0],
        },
        gridConstraints: {
          enabled: gridLimits,
          maxPower: maxGridPower[0],
        },
      }

      const result = await runOptimization(params)
      setOptimizationResults(result)
      setSuccessMessage("Optimization completed successfully")

      // Auto-hide success message after 3 seconds
      setTimeout(() => setSuccessMessage(null), 3000)
    } catch (err) {
      setError("Failed to run optimization. Please try again.")
      console.error(err)
    } finally {
      setIsOptimizing(false)
    }
  }

  const applySchedule = async () => {
    if (!optimizationResults) return

    setIsApplying(true)
    setError(null)

    try {
      await applyOptimizationSchedule(optimizationResults.id)
      setSuccessMessage("Schedule applied successfully")

      // Auto-hide success message after 3 seconds
      setTimeout(() => setSuccessMessage(null), 3000)
    } catch (err) {
      setError("Failed to apply schedule. Please try again.")
      console.error(err)
    } finally {
      setIsApplying(false)
    }
  }

  const resetForm = () => {
    setOptimizationMode("cost")
    setTimeHorizon("24")
    setBatteryUsage(true)
    setBatteryMinSoc([20])
    setBatteryMaxCycles([1])
    setGridLimits(true)
    setMaxGridPower([5])
  }

  return (
    <div className="space-y-6">
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {successMessage && (
        <Alert variant="default" className="bg-green-50 text-green-800 border-green-200">
          <AlertDescription>{successMessage}</AlertDescription>
        </Alert>
      )}

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Optimization Settings</CardTitle>
            <CardDescription>Configure your energy optimization parameters</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-4">
              <div>
                <Label htmlFor="optimization-mode">Optimization Objective</Label>
                <Select value={optimizationMode} onValueChange={setOptimizationMode}>
                  <SelectTrigger id="optimization-mode">
                    <SelectValue placeholder="Select optimization objective" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="cost">Minimize Cost</SelectItem>
                    <SelectItem value="self_consumption">Maximize Self-Consumption</SelectItem>
                    <SelectItem value="grid_independence">Maximize Grid Independence</SelectItem>
                    <SelectItem value="battery_life">Maximize Battery Life</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label htmlFor="time-horizon">Time Horizon (hours)</Label>
                <Select value={timeHorizon} onValueChange={setTimeHorizon}>
                  <SelectTrigger id="time-horizon">
                    <SelectValue placeholder="Select time horizon" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="12">12 Hours</SelectItem>
                    <SelectItem value="24">24 Hours</SelectItem>
                    <SelectItem value="48">48 Hours</SelectItem>
                    <SelectItem value="168">7 Days</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="battery-usage">Battery Usage</Label>
                  <Switch id="battery-usage" checked={batteryUsage} onCheckedChange={setBatteryUsage} />
                </div>
                {batteryUsage && (
                  <div className="space-y-4 pt-2">
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <Label htmlFor="battery-min-soc">Minimum State of Charge (%)</Label>
                        <span>{batteryMinSoc}%</span>
                      </div>
                      <Slider
                        id="battery-min-soc"
                        min={0}
                        max={50}
                        step={5}
                        value={batteryMinSoc}
                        onValueChange={setBatteryMinSoc}
                      />
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <Label htmlFor="battery-max-cycles">Maximum Daily Cycles</Label>
                        <span>{batteryMaxCycles}</span>
                      </div>
                      <Slider
                        id="battery-max-cycles"
                        min={0.5}
                        max={2}
                        step={0.1}
                        value={batteryMaxCycles}
                        onValueChange={setBatteryMaxCycles}
                      />
                    </div>
                  </div>
                )}
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="grid-limits">Grid Power Limits</Label>
                  <Switch id="grid-limits" checked={gridLimits} onCheckedChange={setGridLimits} />
                </div>
                {gridLimits && (
                  <div className="space-y-2 pt-2">
                    <div className="flex justify-between">
                      <Label htmlFor="max-grid-power">Maximum Grid Power (kW)</Label>
                      <span>{maxGridPower} kW</span>
                    </div>
                    <Slider
                      id="max-grid-power"
                      min={1}
                      max={10}
                      step={0.5}
                      value={maxGridPower}
                      onValueChange={setMaxGridPower}
                    />
                  </div>
                )}
              </div>
            </div>

            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={resetForm}>
                Reset
              </Button>
              <Button onClick={runOptimizationProcess} disabled={isOptimizing} className="gap-2">
                {isOptimizing ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    Optimizing...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4" />
                    Run Optimization
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Optimization Results</CardTitle>
            <CardDescription>
              {optimizationResults
                ? `Last run: ${new Date(optimizationResults.timestamp).toLocaleString()}`
                : "Run optimization to see results"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isOptimizing ? (
              <div className="space-y-6">
                <div className="grid grid-cols-2 gap-4">
                  {Array.from({ length: 4 }).map((_, i) => (
                    <Card key={i}>
                      <CardContent className="p-4">
                        <Skeleton className="h-8 w-8 rounded-full mb-2" />
                        <Skeleton className="h-4 w-24 mb-1" />
                        <Skeleton className="h-6 w-16" />
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            ) : optimizationResults ? (
              <div className="space-y-6">
                <div className="grid grid-cols-2 gap-4">
                  <Card>
                    <CardContent className="p-4 flex flex-col items-center justify-center">
                      <DollarSign className="h-8 w-8 text-primary mb-2" />
                      <div className="text-sm text-muted-foreground">Optimized Cost</div>
                      <div className="text-2xl font-bold">${optimizationResults.cost.toFixed(2)}</div>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="p-4 flex flex-col items-center justify-center">
                      <Sun className="h-8 w-8 text-primary mb-2" />
                      <div className="text-sm text-muted-foreground">Self-Consumption</div>
                      <div className="text-2xl font-bold">{optimizationResults.selfConsumption}%</div>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="p-4 flex flex-col items-center justify-center">
                      <Home className="h-8 w-8 text-primary mb-2" />
                      <div className="text-sm text-muted-foreground">Peak Grid Power</div>
                      <div className="text-2xl font-bold">{optimizationResults.peakGridPower} kW</div>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="p-4 flex flex-col items-center justify-center">
                      <Battery className="h-8 w-8 text-primary mb-2" />
                      <div className="text-sm text-muted-foreground">Battery Cycles</div>
                      <div className="text-2xl font-bold">{optimizationResults.batteryCycles}</div>
                    </CardContent>
                  </Card>
                </div>

                <div className="flex justify-end gap-2">
                  <Button variant="outline" className="gap-2">
                    <Save className="h-4 w-4" />
                    Save Results
                  </Button>
                  <Button className="gap-2" onClick={applySchedule} disabled={isApplying}>
                    {isApplying ? (
                      <>
                        <RefreshCw className="h-4 w-4 animate-spin" />
                        Applying...
                      </>
                    ) : (
                      <>
                        <Zap className="h-4 w-4" />
                        Apply Schedule
                      </>
                    )}
                  </Button>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <LineChart className="h-12 w-12 text-muted-foreground mb-4" />
                <h3 className="text-lg font-medium mb-2">No Optimization Results</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Configure your settings and run the optimization to see results here.
                </p>
                <Button onClick={runOptimizationProcess} disabled={isOptimizing} className="gap-2">
                  {isOptimizing ? (
                    <>
                      <RefreshCw className="h-4 w-4 animate-spin" />
                      Optimizing...
                    </>
                  ) : (
                    <>
                      <Play className="h-4 w-4" />
                      Run Optimization
                    </>
                  )}
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Optimization Schedule</CardTitle>
          <CardDescription>Optimized energy flow schedule</CardDescription>
        </CardHeader>
        <CardContent>
          {isOptimizing ? (
            <div className="h-[400px] flex items-center justify-center">
              <div className="flex flex-col items-center gap-2">
                <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
                <p className="text-sm text-muted-foreground">Generating optimization schedule...</p>
              </div>
            </div>
          ) : optimizationResults ? (
            <div className="space-y-6">
              <Tabs defaultValue="chart">
                <TabsList>
                  <TabsTrigger value="chart">Energy Flow Chart</TabsTrigger>
                  <TabsTrigger value="devices">Device Schedule</TabsTrigger>
                </TabsList>
                <TabsContent value="chart" className="pt-4">
                  <div className="h-[400px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart
                        data={optimizationResults.scheduleData}
                        margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="time" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Area
                          type="monotone"
                          dataKey="solar"
                          stackId="1"
                          fill="#FFD700"
                          stroke="#FFD700"
                          name="Solar"
                        />
                        <Area
                          type="monotone"
                          dataKey="battery"
                          stackId="2"
                          fill="#4CAF50"
                          stroke="#4CAF50"
                          name="Battery"
                        />
                        <Area type="monotone" dataKey="grid" stackId="3" fill="#9C27B0" stroke="#9C27B0" name="Grid" />
                        <Bar dataKey="load" barSize={20} fill="#FF5722" name="Original Load" />
                        <Bar dataKey="optimizedLoad" barSize={20} fill="#2196F3" name="Optimized Load" />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </TabsContent>
                <TabsContent value="devices" className="pt-4">
                  <div className="space-y-4">
                    {optimizationResults.deviceScheduleData.length === 0 ? (
                      <div className="flex flex-col items-center justify-center py-12 text-center">
                        <p className="text-muted-foreground">No device schedules available</p>
                      </div>
                    ) : (
                      optimizationResults.deviceScheduleData.map((device) => (
                        <Card key={device.id}>
                          <CardContent className="p-4">
                            <div className="flex flex-col">
                              <div className="flex items-center justify-between mb-2">
                                <h3 className="font-medium">{device.name}</h3>
                                <span className="text-sm text-muted-foreground capitalize">{device.type}</span>
                              </div>
                              <div className="space-y-2">
                                {device.schedule.map((slot, index) => (
                                  <div
                                    key={index}
                                    className="flex items-center justify-between bg-muted p-2 rounded-md"
                                  >
                                    <div className="flex items-center gap-2">
                                      <Clock className="h-4 w-4 text-muted-foreground" />
                                      <span>
                                        {slot.start} - {slot.end}
                                      </span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                      <Zap className="h-4 w-4 text-muted-foreground" />
                                      <span>{slot.power} kW</span>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))
                    )}
                  </div>
                </TabsContent>
              </Tabs>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <Calendar className="h-12 w-12 text-muted-foreground mb-4" />
              <h3 className="text-lg font-medium mb-2">No Schedule Available</h3>
              <p className="text-sm text-muted-foreground">Run the optimization to generate a schedule.</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

