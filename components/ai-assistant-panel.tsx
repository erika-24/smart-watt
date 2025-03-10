"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Bot, LineChart, Lightbulb } from "lucide-react"
import ChatInterface from "@/components/chat-interface"
import ChartAnalysis from "@/components/chart-analysis"

interface AIAssistantPanelProps {
  systemData?: any
  chartData?: any
  chartType?: string
  timePeriod?: string
}

export default function AIAssistantPanel({
  systemData,
  chartData,
  chartType = "line",
  timePeriod = "Last 24 hours",
}: AIAssistantPanelProps) {
  const [activeTab, setActiveTab] = useState("chat")
  const [showChartAnalysis, setShowChartAnalysis] = useState(false)

  const handleAnalysisRequest = () => {
    if (chartData) {
      setActiveTab("analysis")
      setShowChartAnalysis(true)
    }
  }

  return (
    <Card className="w-full">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2">
          <Bot className="h-5 w-5 text-primary" />
          AI Energy Assistant
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid grid-cols-2 mx-4 mt-2">
            <TabsTrigger value="chat" className="flex items-center gap-2">
              <Bot className="h-4 w-4" />
              Chat
            </TabsTrigger>
            <TabsTrigger
              value="analysis"
              className="flex items-center gap-2"
              disabled={!chartData || !showChartAnalysis}
            >
              <LineChart className="h-4 w-4" />
              Chart Analysis
            </TabsTrigger>
          </TabsList>

          <TabsContent value="chat" className="p-4 pt-2">
            <ChatInterface systemData={systemData} onAnalysisRequest={handleAnalysisRequest} />
          </TabsContent>

          <TabsContent value="analysis" className="p-4 pt-2">
            {showChartAnalysis && chartData ? (
              <ChartAnalysis chartData={chartData} chartType={chartType} timePeriod={timePeriod} />
            ) : (
              <div className="flex flex-col items-center justify-center py-8">
                <Lightbulb className="h-12 w-12 text-muted-foreground mb-4" />
                <p className="text-muted-foreground text-center mb-4">
                  Ask the assistant to analyze your energy data to see AI-powered insights here.
                </p>
                <Button variant="outline" onClick={() => setActiveTab("chat")} className="mt-2">
                  Go to Chat
                </Button>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}

