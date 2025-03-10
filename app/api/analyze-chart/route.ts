import { NextResponse } from "next/server"
import { generateText } from "ai"
import { openai } from "@ai-sdk/openai"

// This API route handles chart analysis
export async function POST(request: Request) {
  try {
    const body = await request.json()
    const { chartData, chartType, timePeriod } = body

    if (!chartData || !chartType) {
      return NextResponse.json({ error: "Chart data and type are required" }, { status: 400 })
    }

    // Convert chart data to string representation
    const chartDataString = JSON.stringify(chartData)

    // Generate analysis using AI
    const prompt = `
You are an expert energy analyst specializing in analyzing energy charts and data visualizations.
Analyze the following chart data and provide detailed insights.

Chart Data:
${chartDataString}

Chart Type: ${chartType}
Time Period: ${timePeriod || "Not specified"}

Please provide a comprehensive analysis of this chart, including:
1. Key trends and patterns visible in the chart
2. Notable peaks, valleys, or anomalies
3. Correlations between different data series (if multiple series are present)
4. Potential causes for the observed patterns
5. Optimization opportunities based on the chart data
6. Recommendations for improving energy efficiency or cost savings

Format your response as a JSON object with the following structure:
{
  "trends": ["list of identified trends"],
  "anomalies": ["list of identified anomalies"],
  "correlations": ["list of identified correlations"],
  "causes": ["list of potential causes"],
  "opportunities": ["list of optimization opportunities"],
  "recommendations": ["list of specific recommendations"]
}

Ensure your insights are specific, actionable, and based on the chart data provided.
`

    const { text } = await generateText({
      model: openai("gpt-4o"),
      prompt: prompt,
      maxTokens: 1000,
    })

    // Parse the response as JSON
    let analysis
    try {
      analysis = JSON.parse(text)
    } catch (e) {
      // If parsing fails, return the raw text
      analysis = { general: text }
    }

    return NextResponse.json({
      analysis,
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    console.error("Error in chart analysis API:", error)
    return NextResponse.json({ error: "Failed to analyze chart" }, { status: 500 })
  }
}

