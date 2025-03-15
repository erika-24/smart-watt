import Link from "next/link"
import { Battery, Home, Settings, Sun, Zap } from "lucide-react"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import EnergyDashboard from "@/components/energy-dashboard"
import OptimizationPanel from "@/components/optimization-panel"
import EnergyFlowDiagram from "@/components/energy-flow-diagram"
import DevicesPanel from "@/components/devices-panel"

export default function HomePage() {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-10 border-b bg-background/95 backdrop-blur">
        <div className="container flex h-16 items-center justify-between py-4">
          <div className="flex items-center gap-2">
            <Zap className="h-6 w-6 text-primary" />
            <h1 className="text-xl font-bold">EnergyOptimize</h1>
          </div>
          <nav className="flex items-center gap-6">
            <Link href="#" className="text-sm font-medium">
              Dashboard
            </Link>
            <Link href="#" className="text-sm font-medium text-muted-foreground hover:text-foreground">
              Devices
            </Link>
            <Link href="#" className="text-sm font-medium text-muted-foreground hover:text-foreground">
              History
            </Link>
            <Link href="#" className="text-sm font-medium text-muted-foreground hover:text-foreground">
              Settings
            </Link>
          </nav>
        </div>
      </header>
      <main className="flex-1">
        <div className="container py-6">
          <div className="mb-8 flex items-center justify-between">
            <div>
              <h2 className="text-3xl font-bold tracking-tight">Energy Dashboard</h2>
              <p className="text-muted-foreground">
                Monitor and optimize your home energy usage with linear programming techniques.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Link
                href="#"
                className="inline-flex h-9 items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground shadow hover:bg-primary/90"
              >
                <Settings className="mr-2 h-4 w-4" />
                Configure
              </Link>
            </div>
          </div>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Current Power</CardTitle>
                <Zap className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">2.4 kW</div>
                <p className="text-xs text-muted-foreground">+5% from average</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Solar Production</CardTitle>
                <Sun className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">1.8 kW</div>
                <p className="text-xs text-muted-foreground">75% of current usage</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Battery Status</CardTitle>
                <Battery className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">85%</div>
                <p className="text-xs text-muted-foreground">8.5 kWh available</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Grid Usage</CardTitle>
                <Home className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">0.6 kW</div>
                <p className="text-xs text-muted-foreground">25% of current usage</p>
              </CardContent>
            </Card>
          </div>
          <Tabs defaultValue="dashboard" className="mt-6">
            <TabsList>
              <TabsTrigger value="dashboard">Dashboard</TabsTrigger>
              <TabsTrigger value="optimization">Optimization</TabsTrigger>
              <TabsTrigger value="energy-flow">Energy Flow</TabsTrigger>
              <TabsTrigger value="devices">Devices</TabsTrigger>
            </TabsList>
            <TabsContent value="dashboard" className="space-y-4">
              <EnergyDashboard />
            </TabsContent>
            <TabsContent value="optimization" className="space-y-4">
              <OptimizationPanel />
            </TabsContent>
            <TabsContent value="energy-flow" className="space-y-4">
              <EnergyFlowDiagram />
            </TabsContent>
            <TabsContent value="devices" className="space-y-4">
              <DevicesPanel />
            </TabsContent>
          </Tabs>
        </div>
      </main>
    </div>
  )
}

