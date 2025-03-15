import { NextResponse } from "next/server"
import { generateText } from "ai"
import { openai } from "@ai-sdk/openai"

// This API route handles chat interactions
export async function POST(request: Request) {
  try {
    const body = await request.json()
    const { message, systemData } = body

    if (!message) {
      return NextResponse.json({ error: "Message is required" }, { status: 400 })
    }

    // Prepare context from system data
    let context = "Current Energy System Status:\n"

    if (systemData?.sensorData) {
      const { loadPower, pvPower, batteryPower, batterySoc, gridPower } = systemData.sensorData
      context += `- Load: ${loadPower?.toFixed(2) || "N/A"} kW\n`
      context += `- Solar: ${pvPower?.toFixed(2) || "N/A"} kW\n`
      context += `- Battery: ${batteryPower?.toFixed(2) || "N/A"} kW (${batterySoc?.toFixed(1) || "N/A"}%)\n`
      context += `- Grid: ${gridPower?.toFixed(2) || "N/A"} kW\n`
    }

    if (systemData?.optimizationResults) {
      const { cost, selfConsumption, peakGridPower, batteryCycles } = systemData.optimizationResults
      context += "\nOptimization Results:\n"
      context += `- Cost: $${cost?.toFixed(2) || "N/A"}\n`
      context += `- Self-Consumption: ${selfConsumption?.toFixed(1) || "N/A"}%\n`
      context += `- Peak Grid Power: ${peakGridPower?.toFixed(2) || "N/A"} kW\n`
      context += `- Battery Cycles: ${batteryCycles?.toFixed(2) || "N/A"}\n`
    }

    // Generate response using AI
    const prompt = `
You are an AI assistant for a home energy management system with solar panels and battery storage.
The user has asked: "${message}"

Here is the current system data:
${context}

Provide a helpful, informative response that addresses the user's question.
Your response should be conversational but precise, and include specific data from the context when relevant.
If you don't know the answer or don't have enough information, say so clearly and suggest what additional information might be needed.
`

    const { text } = await generateText({
      model: openai("gpt-4o"),
      prompt: prompt,
      maxTokens: 500,
    })

    return NextResponse.json({
      response: text,
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    console.error("Error in chat API:", error)
    return NextResponse.json({ error: "Failed to process chat message" }, { status: 500 })
  }
}

