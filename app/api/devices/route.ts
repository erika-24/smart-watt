import { NextResponse } from "next/server"

export async function GET() {
  // Simulate API response delay
  await new Promise((resolve) => setTimeout(resolve, 500))

  // Generate device data
  const devices = [
    {
      id: "1",
      name: "Living Room Lights",
      type: "lighting",
      status: "online",
      power: 60,
      isOn: true,
      icon: "lightbulb",
      schedule: [{ start: "18:00", end: "23:00", power: 0.06 }],
    },
    {
      id: "2",
      name: "Kitchen Refrigerator",
      type: "appliance",
      status: "online",
      power: 120,
      isOn: true,
      icon: "refrigerator",
    },
    {
      id: "3",
      name: "EV Charger",
      type: "charger",
      status: "offline",
      power: 0,
      isOn: false,
      icon: "car",
      schedule: [{ start: "01:00", end: "05:00", power: 7.2 }],
    },
    {
      id: "4",
      name: "Washing Machine",
      type: "appliance",
      status: "online",
      power: 0,
      isOn: false,
      icon: "washing-machine",
      schedule: [{ start: "14:00", end: "15:30", power: 0.8 }],
    },
    {
      id: "5",
      name: "Smart Thermostat",
      type: "hvac",
      status: "online",
      power: 500,
      isOn: true,
      icon: "thermometer",
    },
  ]

  return NextResponse.json(devices)
}

