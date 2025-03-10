import { NextResponse } from "next/server"

export async function GET() {
  // Simulate API response delay
  await new Promise((resolve) => setTimeout(resolve, 500))

  // Generate energy source breakdown
  const sources = [
    { name: "Solar", value: Number.parseFloat((Math.random() * 5 + 10).toFixed(1)) },
    { name: "Battery", value: Number.parseFloat((Math.random() * 3 + 4).toFixed(1)) },
    { name: "Grid", value: Number.parseFloat((Math.random() * 4 + 5).toFixed(1)) },
  ]

  // Generate energy consumption breakdown
  const consumption = [
    { name: "HVAC", value: Number.parseFloat((Math.random() * 3 + 7).toFixed(1)) },
    { name: "Appliances", value: Number.parseFloat((Math.random() * 2 + 5).toFixed(1)) },
    { name: "Lighting", value: Number.parseFloat((Math.random() * 1 + 2).toFixed(1)) },
    { name: "EV Charging", value: Number.parseFloat((Math.random() * 2 + 3).toFixed(1)) },
    { name: "Other", value: Number.parseFloat((Math.random() * 1 + 2).toFixed(1)) },
  ]

  return NextResponse.json({ sources, consumption })
}

