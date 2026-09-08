import { beforeEach, describe, expect, it, vi, afterEach } from "vitest";
import { VoyageContextualizedEmbeddings } from "../voyage_contextualized.js";

/**
 * Builds a mock response for the contextualized embeddings endpoint. Each
 * document maps to a list of chunk embeddings.
 */
function mockFetchResponse(documents: number[][][]): Response {
  return {
    ok: true,
    json: async () => ({
      object: "list",
      data: documents.map((chunks, docIndex) => ({
        object: "list",
        index: docIndex,
        data: chunks.map((embedding, index) => ({
          object: "embedding",
          embedding,
          index,
        })),
      })),
      model: "voyage-context-4",
      usage: { total_tokens: 10 },
    }),
  } as unknown as Response;
}

function mockFetchError(status: number, body: unknown): Response {
  return {
    ok: false,
    status,
    json: async () => body,
  } as unknown as Response;
}

beforeEach(() => {
  vi.stubEnv("VOYAGE_API_KEY", "test-voyage-key");
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("VoyageContextualizedEmbeddings constructor", () => {
  it("throws if no API key is available", () => {
    const k1 = process.env.VOYAGE_API_KEY;
    const k2 = process.env.VOYAGEAI_API_KEY;
    delete process.env.VOYAGE_API_KEY;
    delete process.env.VOYAGEAI_API_KEY;
    try {
      expect(() => new VoyageContextualizedEmbeddings()).toThrow(
        "Voyage AI API key not found"
      );
    } finally {
      if (k1 !== undefined) process.env.VOYAGE_API_KEY = k1;
      if (k2 !== undefined) process.env.VOYAGEAI_API_KEY = k2;
    }
  });

  it("defaults to the voyage-context-4 model", () => {
    const embeddings = new VoyageContextualizedEmbeddings();
    expect(embeddings.modelName).toBe("voyage-context-4");
  });

  it("defaults chunkSize to 32000", () => {
    const embeddings = new VoyageContextualizedEmbeddings();
    expect(embeddings.chunkSize).toBe(32000);
  });

  it("defaults basePath to voyageai.com and targets the contextualized endpoint", () => {
    const embeddings = new VoyageContextualizedEmbeddings();
    expect(embeddings.basePath).toBe("https://api.voyageai.com/v1");
    expect(embeddings.apiUrl).toBe(
      "https://api.voyageai.com/v1/contextualizedembeddings"
    );
  });

  it("accepts a custom basePath for MongoDB Atlas keys", () => {
    const embeddings = new VoyageContextualizedEmbeddings({
      basePath: "https://ai.mongodb.com/v1",
    });
    expect(embeddings.apiUrl).toBe(
      "https://ai.mongodb.com/v1/contextualizedembeddings"
    );
  });
});

describe("VoyageContextualizedEmbeddings.contextualizedEmbed", () => {
  it("calls contextualized_embed with a flat list[str] input, auto-chunking and chunk_size 32000", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        mockFetchResponse([
          [
            [1, 0],
            [0, 1],
          ],
          [[0.5, 0.5]],
        ])
      );

    const embeddings = new VoyageContextualizedEmbeddings();
    const result = await embeddings.contextualizedEmbed([
      "document one",
      "document two",
    ]);

    expect(fetchSpy).toHaveBeenCalledOnce();
    const [url, init] = fetchSpy.mock.calls[0];
    expect(url).toBe("https://api.voyageai.com/v1/contextualizedembeddings");

    const body = JSON.parse((init as RequestInit).body as string);
    expect(body.model).toBe("voyage-context-4");
    expect(body.input).toEqual(["document one", "document two"]);
    expect(body.enable_auto_chunking).toBe(true);
    expect(body.chunk_size).toBe(32000);
    expect(body.input_type).toBe("document");

    // One embedding per chunk, grouped per document.
    expect(result).toEqual([
      [
        [1, 0],
        [0, 1],
      ],
      [[0.5, 0.5]],
    ]);
  });

  it("honors a custom chunkSize", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(mockFetchResponse([[[1, 0]]]));

    const embeddings = new VoyageContextualizedEmbeddings({ chunkSize: 1024 });
    await embeddings.contextualizedEmbed(["doc"]);

    const body = JSON.parse(
      (fetchSpy.mock.calls[0][1] as RequestInit).body as string
    );
    expect(body.chunk_size).toBe(1024);
  });
});

describe("VoyageContextualizedEmbeddings.embedDocuments", () => {
  it("returns the leading chunk embedding per document", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      mockFetchResponse([
        [
          [1, 0],
          [0, 1],
        ],
        [[0.5, 0.5]],
      ])
    );

    const embeddings = new VoyageContextualizedEmbeddings();
    const result = await embeddings.embedDocuments(["doc one", "doc two"]);

    expect(result).toEqual([
      [1, 0],
      [0.5, 0.5],
    ]);
  });
});

describe("VoyageContextualizedEmbeddings.embedQuery", () => {
  it("embeds a query as a single chunk without auto-chunking", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(mockFetchResponse([[[0.1, 0.2, 0.3]]]));

    const embeddings = new VoyageContextualizedEmbeddings();
    const result = await embeddings.embedQuery("what is a vector store?");

    expect(result).toEqual([0.1, 0.2, 0.3]);

    const body = JSON.parse(
      (fetchSpy.mock.calls[0][1] as RequestInit).body as string
    );
    expect(body.input).toEqual([["what is a vector store?"]]);
    expect(body.input_type).toBe("query");
    expect(body.enable_auto_chunking).toBe(false);
  });
});

describe("VoyageContextualizedEmbeddings error handling", () => {
  it("throws with status code on non-OK response", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      mockFetchError(401, { detail: "Invalid API key" })
    );

    const embeddings = new VoyageContextualizedEmbeddings();
    await expect(embeddings.embedQuery("test")).rejects.toThrow(
      "Voyage AI API error (HTTP 401): Invalid API key"
    );
  });
});
