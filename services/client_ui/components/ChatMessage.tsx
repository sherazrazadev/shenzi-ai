import ReactMarkdown from "react-markdown";

interface ChatMessageProps {
  message: string;
  isUser: boolean;
  timestamp: Date;
}

export default function ChatMessage({
  message,
  isUser,
  timestamp,
}: ChatMessageProps) {
  // Normalize lists and paragraphs; avoid breaking Markdown lists so items are grouped and aligned
  const formattedMessage = message
    // Normalize CRLF to LF
    .replace(/(\r\n|\r)/g, "\n")
    // Convert patterns like ": + ..." and " + ..." into proper list items
    .replace(/:\s*\+\s*/g, ":\n- ")
    .replace(/\s\+\s/g, "\n- ")
    // Remove occurrences like ":*" that appear when model outputs "bullets:* Heading:"
    .replace(/:\s*\*\s*/g, ":\n")
    // Convert newline-starting single '*' into list items (require a following space so '**' is not matched)
    .replace(/\n\*\s+/g, "\n- ")
    // Convert standalone ' * ' separators into list items but avoid touching '**' (bold)
    .replace(/\s\*\s/g, "\n- ")
    // Move headings like "Structure:" to their own bold paragraph when followed by list items
    .replace(/\n- ([A-Z][^:\n]{1,60}):\s*\n- /g, "\n\n**$1:**\n- ")
    .replace(/([^\n])\s+([A-Z][^:\n]{1,60}:)\s*(?=\n-)/g, "$1\n\n**$2**")
    // Ensure numbered lists start on a new line
    .replace(/(\d+\.)/g, "\n$1")
    // Convert single newlines into paragraph breaks, but do NOT insert breaks before list markers (so lists stay grouped)
    .replace(/([^\n])\n(?!- |\d+\.)/g, "$1\n\n")
    .trim();

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div
        className={`max-w-2xl px-4 py-2 rounded-lg ${
          isUser ? "bg-blue-500 text-white" : "bg-gray-200 text-gray-800"
        }`}
      >
        <div className="text-sm leading-relaxed">
          <ReactMarkdown
            components={{
              ul: ({ children }) => (
                <ul className="list-disc list-inside ml-4 mb-3 space-y-1">
                  {children}
                </ul>
              ),
              ol: ({ children }) => (
                <ol className="list-decimal list-inside ml-4 mb-3 space-y-1">
                  {children}
                </ol>
              ),
              li: ({ children }) => <li className="mb-1">{children}</li>,
              strong: ({ children }) => (
                <strong className="font-bold text-base">{children}</strong>
              ),
              p: ({ children }) => <p className="mb-3 leading-6">{children}</p>,
              h1: ({ children }) => (
                <h1 className="text-2xl font-bold mb-3 text-gray-900">
                  {children}
                </h1>
              ),
              h2: ({ children }) => (
                <h2 className="text-xl font-bold mb-2 text-gray-900">
                  {children}
                </h2>
              ),
              h3: ({ children }) => (
                <h3 className="text-lg font-bold mb-2 text-gray-900">
                  {children}
                </h3>
              ),
              code: ({ children }) => (
                <code className="bg-gray-100 px-2 py-1 rounded text-sm font-mono">
                  {children}
                </code>
              ),
              blockquote: ({ children }) => (
                <blockquote className="border-l-4 border-gray-400 pl-4 italic text-gray-700 mb-3">
                  {children}
                </blockquote>
              ),
            }}
          >
            {formattedMessage}
          </ReactMarkdown>
        </div>
        <p
          className={`text-xs mt-2 ${isUser ? "text-blue-100" : "text-gray-500"}`}
        >
          {timestamp.toLocaleTimeString()}
        </p>
      </div>
    </div>
  );
}
