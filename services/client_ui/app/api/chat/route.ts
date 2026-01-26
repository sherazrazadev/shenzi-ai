import { NextRequest, NextResponse } from "next/server";

const API_GATEWAY_URL = process.env.API_GATEWAY_URL || "http://localhost:8000";

export async function POST(request: NextRequest) {
  let message = "";

  try {
    const body = await request.json();
    message = body.message;

    if (!message) {
      return NextResponse.json(
        { error: "Message is required" },
        { status: 400 },
      );
    }

    // Prepare messages array
    const messages = [{ role: "user", content: message }];

    // Proxy the request to the API gateway chat endpoint
    const gatewayResponse = await fetch(`${API_GATEWAY_URL}/chat`, {
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

    const data = await gatewayResponse.json();

    return NextResponse.json({
      response: data.response || "Sorry, I couldn't process your message.",
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Chat API error:", error);

    // Fallback dummy response
    let response =
      "Sorry, I'm having trouble connecting to the AI service. This is a fallback response.";

    if (message) {
      const lowerMessage = message.toLowerCase();
      if (lowerMessage.includes("hello") || lowerMessage.includes("hi")) {
        response = "Hello! How can I assist you today?";
      } else if (
        lowerMessage.includes("task") ||
        lowerMessage.includes("todo")
      ) {
        response =
          "I can help you manage your tasks! Would you like me to create a new task or list your existing ones?";
      }
    }

    return NextResponse.json({
      response,
      timestamp: new Date().toISOString(),
    });
  }
}
