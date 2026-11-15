import { beforeEach, describe, expect, it, vi, afterEach } from "vitest";
import { VoyageContextualizedEmbeddings } from "../voyage_contextualized.js";

type Doc = Array<{ embedding: number[]; text: string }>;

/** Builds a response: one entry per document, each holding its chunks. */
function mockResponse(docs: Doc[], shuffle = false): Response {
  const data = docs.map((chunks, docIndex) => ({
    index: docIndex,
    data: chunks.map((c, chunkIndex) => ({ ...c, index: chunkIndex })),
  }));
  return {
    ok: true,
    json: async () => ({
      data: shuffle ? [...data].reverse() : data,
      model: "voyage-context-4",
      usage: { total_tokens: 1 },
    }),
  } as unknown as Response;
}

function mockError(status: number, body: unknown): Response {
  return { ok: false, status, json: async () => body } as unknown as Response;
}

const bodyOf = (spy: ReturnType<typeof vi.spyOn>, call = 0) =>
  JSON.parse((spy.mock.calls[call]?.[1] as RequestInit).body as string);

beforeEach(() => {
  vi.stubEnv("VOYAGE_API_KEY", "test-voyage-key");
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("VoyageContextualizedEmbeddings", () => {
  it("requires modelName and an API key", () => {
    expect(() => new VoyageContextualizedEmbeddings({ modelName: "" })).toThrow(
      "modelName is required"
    );
    vi.unstubAllEnvs();
    vi.stubEnv("VOYAGE_API_KEY", "");
    vi.stubEnv("VOYAGEAI_API_KEY", "");
    expect(
      () =>
        new VoyageContextualizedEmbeddings({ modelName: "voyage-context-4" })
    ).toThrow("Voyage AI API key not found");
  });

  it("uses the MongoDB AI endpoint for al- keys unless basePath is set", () => {
    const atlas = new VoyageContextualizedEmbeddings({
      modelName: "voyage-context-4",
      apiKey: "al-key",
    });
    expect(atlas.apiUrl).toBe(
      "https://ai.mongodb.com/v1/contextualizedembeddings"
    );
    const explicit = new VoyageContextualizedEmbeddings({
      modelName: "voyage-context-4",
      apiKey: "al-key",
      basePath: "https://api.voyageai.com/v1",
    });
    expect(explicit.apiUrl).toBe(
      "https://api.voyageai.com/v1/contextualizedembeddings"
    );
  });

  it("sends a flat string list with auto-chunking in document mode", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      mockResponse([
        [
          { embedding: [1], text: "a" },
          { embedding: [2], text: "b" },
        ],
      ])
    );
    const emb = new VoyageContextualizedEmbeddings({
      modelName: "voyage-context-4",
      outputDimension: 256,
    });
    const result = await emb.contextualizedEmbed(["a b"]);

    expect(bodyOf(spy)).toEqual({
      model: "voyage-context-4",
      inputs: ["a b"],
      input_type: "document",
      enable_auto_chunking: true,
      chunk_size: 32000,
      output_dimension: 256,
    });
    expect(result).toEqual([{ chunks: ["a", "b"], embeddings: [[1], [2]] }]);
  });

  it("sends queries without auto-chunking", async () => {
    const spy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(mockResponse([[{ embedding: [7], text: "q" }]]));
    const emb = new VoyageContextualizedEmbeddings({
      modelName: "voyage-context-4",
    });
    expect(await emb.embedQuery("q")).toEqual([7]);
    const body = bodyOf(spy);
    expect(body.input_type).toBe("query");
    expect(body.enable_auto_chunking).toBeUndefined();
    expect(body.chunk_size).toBeUndefined();
  });

  it("splits documents into batches and keeps input order", async () => {
    const spy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        mockResponse(
          [[{ embedding: [0], text: "d0" }], [{ embedding: [1], text: "d1" }]],
          true
        )
      )
      .mockResolvedValueOnce(mockResponse([[{ embedding: [2], text: "d2" }]]));
    const emb = new VoyageContextualizedEmbeddings({
      modelName: "voyage-context-4",
      batchSize: 2,
    });
    const result = await emb.contextualizedEmbed(["d0", "d1", "d2"]);

    expect(spy).toHaveBeenCalledTimes(2);
    expect(bodyOf(spy, 0).inputs).toEqual(["d0", "d1"]);
    expect(bodyOf(spy, 1).inputs).toEqual(["d2"]);
    expect(result.map((r) => r.embeddings[0][0])).toEqual([0, 1, 2]);
  });

  it("rejects pre-chunked documents before calling the API", async () => {
    const spy = vi.spyOn(globalThis, "fetch");
    const emb = new VoyageContextualizedEmbeddings({
      modelName: "voyage-context-4",
    });
    await expect(
      emb.contextualizedEmbed([["a", "b"]] as unknown as string[])
    ).rejects.toThrow("one string per document");
    expect(spy).not.toHaveBeenCalled();
  });

  it("embedDocuments throws when a text is split into several chunks", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      mockResponse([
        [
          { embedding: [1], text: "a" },
          { embedding: [2], text: "b" },
        ],
      ])
    );
    const emb = new VoyageContextualizedEmbeddings({
      modelName: "voyage-context-4",
    });
    await expect(emb.embedDocuments(["a b"])).rejects.toThrow(
      "use contextualizedEmbed()"
    );
  });

  it("surfaces the API error detail without retrying 4xx", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockError(400, {
        detail: "auto-chunking expects one string per document",
      })
    );
    const emb = new VoyageContextualizedEmbeddings({
      modelName: "voyage-context-4",
    });
    await expect(emb.contextualizedEmbed(["a"])).rejects.toThrow(
      "Voyage AI API error (HTTP 400): auto-chunking expects one string per document"
    );
    expect(spy).toHaveBeenCalledTimes(1);
  });
});
