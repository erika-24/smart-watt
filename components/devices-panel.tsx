"use client"

import { useState, useEffect } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Badge } from "@/components/ui/badge"
import {
  Lightbulb,
  Thermometer,
  Plug,
  Car,
  WashingMachine,
  Refrigerator,
  Plus,
  Settings,
  RefreshCw,
  AlertCircle,
} from "lucide-react"
import { fetchDevices, toggleDevice, type Device } from "@/lib/api"
import { Skeleton } from "@/components/ui/skeleton"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

export default function DevicesPanel() {
  const [devices, setDevices] = useState<Device[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeFilter, setActiveFilter] = useState("all")
  const [togglingDevices, setTogglingDevices] = useState<Record<string, boolean>>({})

  // Map device types to icons
  const getDeviceIcon = (type: string, isOn: boolean) => {
    const iconProps = { className: "h-5 w-5" }

    switch (type) {
      case "lighting":
        return <Lightbulb {...iconProps} />
      case "hvac":
        return <Thermometer {...iconProps} />
      case "charger":
        return <Car {...iconProps} />
      // case "appliance":
      //   return device.subtype === "refrigerator" ? <Refrigerator {...iconProps} /> : <WashingMachine {...iconProps} />
      default:
        return <Plug {...iconProps} />
    }
  }

  // Fetch devices on component mount
  useEffect(() => {
    async function loadDevices() {
      try {
        setIsLoading(true)
        const data = await fetchDevices()
        setDevices(data)
        setError(null)
      } catch (err) {
        setError("Failed to load devices. Please try again.")
        console.error(err)
      } finally {
        setIsLoading(false)
      }
    }

    loadDevices()

    // Set up polling for real-time updates
    const interval = setInterval(loadDevices, 30000) // Update every 30 seconds
    return () => clearInterval(interval)
  }, [])

  // Toggle device state
  const handleToggleDevice = async (id: string, currentState: boolean) => {
    setTogglingDevices((prev) => ({ ...prev, [id]: true }))

    try {
      const updatedDevice = await toggleDevice(id, !currentState)

      // Update the device in the local state
      setDevices(devices.map((device) => (device.id === id ? updatedDevice : device)))

      setError(null)
    } catch (err) {
      setError(`Failed to toggle device ${id}. Please try again.`)
      console.error(err)
    } finally {
      setTogglingDevices((prev) => ({ ...prev, [id]: false }))
    }
  }

  // Filter devices based on active tab
  const filteredDevices = activeFilter === "all" ? devices : devices.filter((device) => device.type === activeFilter)

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Connected Devices</h2>
        <Button>
          <Plus className="h-4 w-4 mr-2" />
          Add Device
        </Button>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Tabs defaultValue="all" onValueChange={setActiveFilter}>
        <TabsList>
          <TabsTrigger value="all">All Devices</TabsTrigger>
          <TabsTrigger value="lighting">Lighting</TabsTrigger>
          <TabsTrigger value="appliance">Appliances</TabsTrigger>
          <TabsTrigger value="hvac">HVAC</TabsTrigger>
          <TabsTrigger value="charger">Chargers</TabsTrigger>
        </TabsList>

        <TabsContent value={activeFilter} className="space-y-4 mt-4">
          {isLoading ? (
            // Loading skeleton
            Array.from({ length: 5 }).map((_, i) => (
              <Card key={i}>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <Skeleton className="h-10 w-10 rounded-full" />
                      <div>
                        <Skeleton className="h-5 w-40 mb-1" />
                        <Skeleton className="h-4 w-20" />
                      </div>
                    </div>
                    <Skeleton className="h-6 w-12" />
                  </div>
                </CardContent>
              </Card>
            ))
          ) : filteredDevices.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <Plug className="h-12 w-12 text-muted-foreground mb-4" />
              <h3 className="text-lg font-medium mb-2">No Devices Found</h3>
              <p className="text-sm text-muted-foreground">
                {activeFilter === "all"
                  ? "You don't have any connected devices yet."
                  : `You don't have any ${activeFilter} devices connected.`}
              </p>
            </div>
          ) : (
            filteredDevices.map((device) => (
              <Card key={device.id}>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-full ${device.isOn ? "bg-primary/10" : "bg-muted"}`}>
                        {getDeviceIcon(device.type, device.isOn)}
                      </div>
                      <div>
                        <div className="font-medium flex items-center gap-2">
                          {device.name}
                          <Badge variant={device.status === "online" ? "default" : "secondary"} className="ml-2">
                            {device.status}
                          </Badge>
                        </div>
                        <div className="text-sm text-muted-foreground">{device.isOn ? `${device.power}W` : "Off"}</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      {device.schedule && device.schedule.length > 0 && (
                        <div className="text-sm text-muted-foreground">
                          Scheduled: {device.schedule[0].start} - {device.schedule[0].end}
                        </div>
                      )}
                      <div className="flex items-center gap-2">
                        {togglingDevices[device.id] ? (
                          <RefreshCw className="h-4 w-4 animate-spin" />
                        ) : (
                          <Switch
                            checked={device.isOn}
                            onCheckedChange={() => handleToggleDevice(device.id, device.isOn)}
                            aria-label={`Toggle ${device.name}`}
                            disabled={device.status === "offline"}
                          />
                        )}
                        <Button variant="ghost" size="icon">
                          <Settings className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}

