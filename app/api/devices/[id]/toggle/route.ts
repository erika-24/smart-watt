import { NextResponse } from "next/server"

export async function POST(request: Request, { params }: { params: { id: string } }) {
  // Simulate API response delay
  await new Promise((resolve) => setTimeout(resolve, 500))

  const { id } = params
  const { isOn } = await request.json()

  // Simulate device toggling
  // In a real app, this would communicate with the actual device

  // Generate a response with the updated device state
  const updatedDevice = {
    id,
    name:
      id === "1"
        ? "Living Room Lights"
        : id === "2"
          ? "Kitchen Refrigerator"
          : id === "3"
            ? "EV Charger"
            : id === "4"
              ? "Washing Machine"
              : "Smart Thermostat",
    type: id === "1" ? "lighting" : id === "2" || id === "4" ? "appliance" : id === "3" ? "charger" : "hvac",
    status: "online",
    power: isOn ? (id === "1" ? 60 : id === "2" ? 120 : id === "3" ? 7200 : id === "4" ? 800 : 500) : 0,
    isOn,
    icon:
      id === "1"
        ? "lightbulb"
        : id === "2"
          ? "refrigerator"
          : id === "3"
            ? "car"
            : id === "4"
              ? "washing-machine"
              : "thermometer",
    schedule:
      id === "1"
        ? [{ start: "18:00", end: "23:00", power: 0.06 }]
        : id === "3"
          ? [{ start: "01:00", end: "05:00", power: 7.2 }]
          : id === "4"
            ? [{ start: "14:00", end: "15:30", power: 0.8 }]
            : undefined,
  }

  return NextResponse.json(updatedDevice)
}

