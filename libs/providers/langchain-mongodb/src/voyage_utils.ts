const VOYAGE_API_BASE_URL = "https://api.voyageai.com/v1";
const MONGODB_AI_BASE_URL = "https://ai.mongodb.com/v1";

/**
 * Default base URL for a Voyage AI by MongoDB API key. Keys created in the
 * MongoDB Atlas UI start with `al-` and use the MongoDB AI endpoint; all other
 * keys use the Voyage AI endpoint. An explicitly passed base URL always wins
 * over this default.
 */
export function getDefaultVoyageBaseUrl(apiKey: string): string {
  return apiKey.startsWith("al-") ? MONGODB_AI_BASE_URL : VOYAGE_API_BASE_URL;
}

/**
 * Extracts a readable message from a Voyage error body, which is normally
 * `{"detail": "<message>"}`.
 */
export function extractErrorMessage(body: unknown): string {
  if (typeof body === "object" && body !== null) {
    const b = body as Record<string, unknown>;
    if (typeof b.detail === "string") return b.detail;
    const err = b.error;
    if (
      typeof err === "object" &&
      err !== null &&
      typeof (err as Record<string, unknown>).message === "string"
    ) {
      return (err as Record<string, unknown>).message as string;
    }
    return JSON.stringify(body);
  }
  return `Unknown error: ${String(body)}`;
}
