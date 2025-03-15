"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Loader2, TrendingUp, AlertTriangle, GitCompare, HelpCircle, Lightbulb, ListChecks } from "lucide-react"

interface ChartAnalysisProps {
  chartData: any
  chartType: string
  timePeriod?: string
  onClose?: () => void
}

export default function ChartAnalysis({ chartData, chartType, timePeriod, onClose }: ChartAnalysisProps) {
  const [analysis, setAnalysis] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const analyzeChart = async () => {
    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch("/api/analyze-chart", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          chartData,
          chartType,
          timePeriod,
        }),
      })

      if (!response.ok) {
        throw new Error("Failed to analyze chart")
      }

      const data = await response.json()
      setAnalysis(data.analysis)
    } catch (error) {
      console.error("Error analyzing chart:", error)
      setError("Failed to analyze chart. Please try again.")
    } finally {
      setIsLoading(false)
    }
  }

  // Analyze chart on component mount
  useState(() => {
    analyzeChart()
  })

  return (
    <Card className="w-full">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-xl font-bold">Chart Analysis</CardTitle>
        {onClose && (
          <Button variant="ghost" size="sm" onClick={onClose}>
            Close
          </Button>
        )}
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="flex flex-col items-center justify-center py-8">
            <Loader2 className="h-8 w-8 animate-spin text-primary mb-4" />
            <p className="text-muted-foreground">Analyzing chart data...</p>
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center py-8 text-destructive">
            <AlertTriangle className="h-8 w-8 mb-4" />
            <p>{error}</p>
            <Button variant="outline" className="mt-4" onClick={analyzeChart}>
              Try Again
            </Button>
          </div>
        ) : analysis ? (
          <Tabs defaultValue="trends">
            <TabsList className="grid grid-cols-6 mb-4">
              <TabsTrigger value="trends">
                <TrendingUp className="h-4 w-4 mr-2" />
                Trends
              </TabsTrigger>
              <TabsTrigger value="anomalies">
                <AlertTriangle className="h-4 w-4 mr-2" />
                Anomalies
              </TabsTrigger>
              <TabsTrigger value="correlations">
                <GitCompare className="h-4 w-4 mr-2" />
                Correlations
              </TabsTrigger>
              <TabsTrigger value="causes">
                <HelpCircle className="h-4 w-4 mr-2" />
                Causes
              </TabsTrigger>
              <TabsTrigger value="opportunities">
                <Lightbulb className="h-4 w-4 mr-2" />
                Opportunities
              </TabsTrigger>
              <TabsTrigger value="recommendations">
                <ListChecks className="h-4 w-4 mr-2" />
                Recommendations
              </TabsTrigger>
            </TabsList>

            <TabsContent value="trends" className="space-y-4">
              <h3 className="text-lg font-semibold">Key Trends</h3>
              <ul className="space-y-2">
                {analysis.trends && analysis.trends.length > 0 ? (
                  analysis.trends.map((trend: string, index: number) => (
                    <li key={index} className="flex items-start gap-2">
                      <TrendingUp className="h-5 w-5 text-primary mt-0.5 flex-shrink-0" />
                      <span>{trend}</span>
                    </li>
                  ))
                ) : (
                  <li className="text-muted-foreground">No significant trends identified</li>
                )}
              </ul>
            </TabsContent>

            <TabsContent value="anomalies" className="space-y-4">
              <h3 className="text-lg font-semibold">Anomalies</h3>
              <ul className="space-y-2">
                {analysis.anomalies && analysis.anomalies.length > 0 ? (
                  analysis.anomalies.map((anomaly: string, index: number) => (
                    <li key={index} className="flex items-start gap-2">
                      <AlertTriangle className="h-5 w-5 text-warning mt-0.5 flex-shrink-0" />
                      <span>{anomaly}</span>
                    </li>
                  ))
                ) : (
                  <li className="text-muted-foreground">No anomalies detected</li>
                )}
              </ul>
            </TabsContent>

            <TabsContent value="correlations" className="space-y-4">
              <h3 className="text-lg font-semibold">Correlations</h3>
              <ul className="space-y-2">
                {analysis.correlations && analysis.correlations.length > 0 ? (
                  analysis.correlations.map((correlation: string, index: number) => (
                    <li key={index} className="flex items-start gap-2">
                      <GitCompare className="h-5 w-5 text-primary mt-0.5 flex-shrink-0" />
                      <span>{correlation}</span>
                    </li>
                  ))
                ) : (
                  <li className="text-muted-foreground">No significant correlations identified</li>
                )}
              </ul>
            </TabsContent>

            <TabsContent value="causes" className="space-y-4">
              <h3 className="text-lg font-semibold">Potential Causes</h3>
              <ul className="space-y-2">
                {analysis.causes && analysis.causes.length > 0 ? (
                  analysis.causes.map((cause: string, index: number) => (
                    <li key={index} className="flex items-start gap-2">
                      <HelpCircle className="h-5 w-5 text-primary mt-0.5 flex-shrink-0" />
                      <span>{cause}</span>
                    </li>
                  ))
                ) : (
                  <li className="text-muted-foreground">No specific causes identified</li>
                )}
              </ul>
            </TabsContent>

            <TabsContent value="opportunities" className="space-y-4">
              <h3 className="text-lg font-semibold">Optimization Opportunities</h3>
              <ul className="space-y-2">
                {analysis.opportunities && analysis.opportunities.length > 0 ? (
                  analysis.opportunities.map((opportunity: string, index: number) => (
                    <li key={index} className="flex items-start gap-2">
                      <Lightbulb className="h-5 w-5 text-warning mt-0.5 flex-shrink-0" />
                      <span>{opportunity}</span>
                    </li>
                  ))
                ) : (
                  <li className="text-muted-foreground">No specific optimization opportunities identified</li>
                )}
              </ul>
            </TabsContent>

            <TabsContent value="recommendations" className="space-y-4">
              <h3 className="text-lg font-semibold">Recommendations</h3>
              <ul className="space-y-2">
                {analysis.recommendations && analysis.recommendations.length > 0 ? (
                  analysis.recommendations.map((recommendation: string, index: number) => (
                    <li key={index} className="flex items-start gap-2">
                      <ListChecks className="h-5 w-5 text-success mt-0.5 flex-shrink-0" />
                      <span>{recommendation}</span>
                    </li>
                  ))
                ) : (
                  <li className="text-muted-foreground">No specific recommendations available</li>
                )}
              </ul>
            </TabsContent>
          </Tabs>
        ) : (
          <div className="flex flex-col items-center justify-center py-8">
            <Button variant="primary" onClick={analyzeChart}>
              Analyze Chart
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

