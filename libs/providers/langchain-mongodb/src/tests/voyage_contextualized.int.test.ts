import { describe, expect, test } from "vitest";
import { VoyageContextualizedEmbeddings } from "../voyage_contextualized.js";

// oxlint-disable-next-line no-process-env
const VOYAGE_API_KEY =
  process.env.VOYAGE_API_KEY ?? process.env.VOYAGEAI_API_KEY;

const describeIfKey = VOYAGE_API_KEY ? describe : describe.skip;

describeIfKey("VoyageContextualizedEmbeddings integration", () => {
  const embeddings = new VoyageContextualizedEmbeddings({
    modelName: "voyage-context-4",
    apiKey: VOYAGE_API_KEY,
  });

  test("contextualizedEmbed returns chunks and embeddings per document", async () => {
    const result = await embeddings.contextualizedEmbed([
      "MongoDB Atlas supports vector search. It is a managed database.",
      "LangChain offers many integrations.",
    ]);
    expect(result).toHaveLength(2);
    for (const doc of result) {
      expect(doc.embeddings.length).toBeGreaterThan(0);
      expect(doc.chunks).toHaveLength(doc.embeddings.length);
      expect(doc.embeddings[0]).toHaveLength(1024);
    }
  });

  test("embedQuery returns a vector of the requested dimension", async () => {
    const small = new VoyageContextualizedEmbeddings({
      modelName: "voyage-context-4",
      apiKey: VOYAGE_API_KEY,
      outputDimension: 256,
    });
    expect(await small.embedQuery("What is Atlas?")).toHaveLength(256);
  });

  test("embedDocuments returns one vector per text", async () => {
    const vectors = await embeddings.embedDocuments(["one", "two"]);
    expect(vectors).toHaveLength(2);
    expect(vectors[0]).toHaveLength(1024);
  });
});
