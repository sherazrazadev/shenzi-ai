import { NextRequest, NextResponse } from "next/server";

const API_GATEWAY_URL = process.env.API_GATEWAY_URL || "http://localhost:8000";

export async function POST(request: NextRequest) {
  try {
    const { messages } = await request.json();

    if (!messages || !Array.isArray(messages)) {
      return NextResponse.json(
        { error: "Messages array is required" },
        { status: 400 },
      );
    }

    // Proxy the streaming request to the agent service
    const gatewayResponse = await fetch(`${API_GATEWAY_URL}/chat/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ messages }),
    });

    if (!gatewayResponse.ok) {
      throw new Error(
        `Gateway responded with status: ${gatewayResponse.status}`,
      );
    }

    // Return the streaming response
    return new Response(gatewayResponse.body, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  } catch (error) {
    console.error("Streaming chat API error:", error);

    // Return error as SSE
    const errorStream = new ReadableStream({
      start(controller) {
        controller.enqueue(
          `data: Error: ${error instanceof Error ? error.message : "Unknown error"}\n\n`,
        );
        controller.close();
      },
    });

    return new Response(errorStream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
      },
    });
  }
}
