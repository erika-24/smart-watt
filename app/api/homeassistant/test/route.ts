import { NextResponse } from "next/server"

// This API route handles communication with Home Assistant
export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const entityId = searchParams.get("entityId")

  if (!entityId) {
    // If no entity ID is provided, fetch all entities
    try {
      const entities = await fetchHomeAssistantEntities()
      return NextResponse.json(entities)
    } catch (error) {
      console.error("Error fetching Home Assistant entities:", error)
      return NextResponse.json({ error: "Failed to fetch entities" }, { status: 500 })
    }
  } else {
    // Fetch specific entity state
    try {
      const entityState = await fetchHomeAssistantEntityState(entityId)
      return NextResponse.json(entityState)
    } catch (error) {
      console.error("Error fetching Home Assistant entity state:", error)
      return NextResponse.json({ error: "Failed to fetch entity state" }, { status: 500 })
    }
  }
}

export async function POST(request: Request) {
  const body = await request.json()
  const { entityId, service, serviceData } = body

  if (!entityId || !service) {
    return NextResponse.json({ error: "Entity ID and service are required" }, { status: 400 })
  }

  try {
    // Call Home Assistant service
    const result = await callHomeAssistantService(entityId, service, serviceData)
    return NextResponse.json(result)
  } catch (error) {
    console.error("Error calling Home Assistant service:", error)
    return NextResponse.json({ error: "Failed to call service" }, { status: 500 })
  }
}

// Helper function to fetch all Home Assistant entities
async function fetchHomeAssistantEntities() {
  const haUrl = process.env.HOME_ASSISTANT_URL
  const haToken = process.env.HOME_ASSISTANT_TOKEN

  if (!haUrl || !haToken) {
    throw new Error("HOME_ASSISTANT_URL or HOME_ASSISTANT_TOKEN environment variable is not set")
  }

  const response = await fetch(`${haUrl}/api/states`, {
    headers: {
      Authorization: `Bearer ${haToken}`,
      "Content-Type": "application/json",
    },
  })

  if (!response.ok) {
    throw new Error(`Failed to fetch entities: ${response.statusText}`)
  }

  return response.json()
}

// Helper function to fetch Home Assistant entity state
async function fetchHomeAssistantEntityState(entityId: string) {
  const haUrl = process.env.HOME_ASSISTANT_URL
  const haToken = process.env.HOME_ASSISTANT_TOKEN

  if (!haUrl || !haToken) {
    throw new Error("HOME_ASSISTANT_URL or HOME_ASSISTANT_TOKEN environment variable is not set")
  }

  const response = await fetch(`${haUrl}/api/states/${entityId}`, {
    headers: {
      Authorization: `Bearer ${haToken}`,
      "Content-Type": "application/json",
    },
  })

  if (!response.ok) {
    throw new Error(`Failed to fetch entity state: ${response.statusText}`)
  }

  return response.json()
}

// Helper function to call Home Assistant service
async function callHomeAssistantService(entityId: string, service: string, serviceData?: any) {
  const haUrl = process.env.HOME_ASSISTANT_URL
  const haToken = process.env.HOME_ASSISTANT_TOKEN

  if (!haUrl || !haToken) {
    throw new Error("HOME_ASSISTANT_URL or HOME_ASSISTANT_TOKEN environment variable is not set")
  }

  // Parse domain and service
  const [domain, serviceAction] = service.split(".")

  const payload = {
    entity_id: entityId,
    ...serviceData,
  }

  const response = await fetch(`${haUrl}/api/services/${domain}/${serviceAction}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${haToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  })

  if (!response.ok) {
    throw new Error(`Failed to call service: ${response.statusText}`)
  }

  return response.json()
}

