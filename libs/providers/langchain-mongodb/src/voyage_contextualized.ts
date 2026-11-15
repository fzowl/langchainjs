import { getEnvironmentVariable } from "@langchain/core/utils/env";
import { Embeddings, type EmbeddingsParams } from "@langchain/core/embeddings";
import { chunkArray } from "@langchain/core/utils/chunk_array";
import {
  extractErrorMessage,
  getDefaultVoyageBaseUrl,
} from "./voyage_utils.js";

/**
 * Maximum chunk size (in tokens) accepted by the Voyage AI by MongoDB
 * contextualized embeddings API. Documents are split server-side into chunks
 * of at most this size.
 */
const AUTO_CHUNK_SIZE = 32_000;

/**
 * Parameters of the VoyageContextualizedEmbeddings class.
 */
export interface VoyageContextualizedEmbeddingsParams extends EmbeddingsParams {
  /**
   * The contextualized embedding model: `voyage-context-4` (recommended) or
   * `voyage-context-3`.
   * @see https://docs.voyageai.com/docs/contextualized-chunk-embeddings
   */
  modelName: string;

  /**
   * Voyage AI by MongoDB API key. Falls back to the `VOYAGE_API_KEY`
   * environment variable.
   */
  apiKey?: string;

  /**
   * Base URL for API requests. When not set, it is chosen from the API key:
   * keys starting with `al-` use `https://ai.mongodb.com/v1`, all other keys
   * use `https://api.voyageai.com/v1`.
   */
  basePath?: string;

  /**
   * Number of documents sent per request. Contextualization happens inside a
   * document, so this does not change the embeddings. The API caps a request
   * at 1000 documents and 120K tokens in total, so the default is one
   * document per request; raise it only for small documents. Batches are
   * sent concurrently; set `maxConcurrency` to limit parallel requests.
   * @default 1
   */
  batchSize?: number;

  /**
   * The desired dimension of the output embeddings: 2048, 1024, 512 or 256.
   */
  outputDimension?: number;

  /**
   * The data type of the output embeddings: "float", "int8", "uint8",
   * "binary" or "ubinary".
   */
  outputDtype?: string;
}

/**
 * Embeddings of one document, one per chunk, in chunk order.
 */
export interface VoyageContextualizedDocumentEmbeddings {
  /** The text of each chunk, as split by the API. */
  chunks: string[];
  /** The embedding of each chunk, in the same order as `chunks`. */
  embeddings: number[][];
}

interface VoyageContextualizedResponse {
  data: Array<{
    index: number;
    data: Array<{ index: number; embedding: number[]; text?: string }>;
  }>;
  model: string;
  usage: { total_tokens: number };
}

type ContextualizedRequest = {
  model: string;
  inputs: string[];
  input_type: "document" | "query";
  enable_auto_chunking?: boolean;
  chunk_size?: number;
  output_dimension?: number;
  output_dtype?: string;
};

/**
 * Contextualized chunk embeddings from Voyage AI by MongoDB
 * (`voyage-context-4`, `voyage-context-3`).
 *
 * Documents are chunked server-side (`enable_auto_chunking: true`,
 * `chunk_size: 32000`) and every chunk is embedded in the context of the
 * whole document. Use {@link contextualizedEmbed} to get all chunks of each
 * document. {@link embedDocuments} requires every text to fit in one chunk.
 * Queries are embedded without chunking.
 */
export class VoyageContextualizedEmbeddings
  extends Embeddings
  implements Omit<VoyageContextualizedEmbeddingsParams, "apiKey">
{
  modelName: string;

  batchSize: number;

  basePath: string;

  apiUrl: string;

  outputDimension?: number;

  outputDtype?: string;

  private apiKey: string;

  constructor(fields: VoyageContextualizedEmbeddingsParams) {
    super(fields);

    const apiKey =
      fields.apiKey ??
      getEnvironmentVariable("VOYAGE_API_KEY") ??
      getEnvironmentVariable("VOYAGEAI_API_KEY");
    if (!apiKey) {
      throw new Error("Voyage AI API key not found");
    }
    if (!fields.modelName) {
      throw new Error("modelName is required");
    }

    this.apiKey = apiKey;
    this.modelName = fields.modelName;
    this.batchSize = fields.batchSize ?? 1;
    this.basePath = fields.basePath ?? getDefaultVoyageBaseUrl(apiKey);
    this.apiUrl = `${this.basePath}/contextualizedembeddings`;
    this.outputDimension = fields.outputDimension;
    this.outputDtype = fields.outputDtype;
  }

  /**
   * Embeds full documents. Each document is chunked by the API and every chunk
   * gets its own contextualized embedding.
   * @param documents - Full document texts, one string per document.
   * @returns One entry per document, in input order, holding the chunk texts
   * and their embeddings.
   */
  async contextualizedEmbed(
    documents: string[]
  ): Promise<VoyageContextualizedDocumentEmbeddings[]> {
    // Auto-chunking only accepts one string per document; a pre-chunked list
    // is rejected by the API with HTTP 400, so fail early with a clear error.
    for (const doc of documents) {
      if (typeof doc !== "string") {
        throw new Error(
          "contextualizedEmbed expects one string per document; pre-chunked documents (string[][]) are not supported with auto-chunking"
        );
      }
    }

    const batches = chunkArray(documents, this.batchSize);
    const responses = await Promise.all(
      batches.map((batch) =>
        this.request({
          model: this.modelName,
          inputs: batch,
          input_type: "document",
          enable_auto_chunking: true,
          chunk_size: AUTO_CHUNK_SIZE,
          output_dimension: this.outputDimension,
          output_dtype: this.outputDtype,
        })
      )
    );

    const results: VoyageContextualizedDocumentEmbeddings[] = [];
    responses.forEach((response) => {
      const docs = [...response.data].sort((a, b) => a.index - b.index);
      for (const doc of docs) {
        const chunks = [...doc.data].sort((a, b) => a.index - b.index);
        results.push({
          chunks: chunks.map((c) => c.text ?? ""),
          embeddings: chunks.map((c) => c.embedding),
        });
      }
    });
    return results;
  }

  /**
   * Embeds texts that each fit in a single chunk (up to 32K tokens), one
   * embedding per text. Throws if the API splits a text into several chunks;
   * use {@link contextualizedEmbed} for long documents.
   */
  async embedDocuments(texts: string[]): Promise<number[][]> {
    const documents = await this.contextualizedEmbed(texts);
    return documents.map((doc, i) => {
      if (doc.embeddings.length !== 1) {
        throw new Error(
          `Text at index ${i} was split into ${doc.embeddings.length} chunks; use contextualizedEmbed() to embed long documents`
        );
      }
      return doc.embeddings[0];
    });
  }

  /**
   * Embeds a query. Queries are not chunked.
   */
  async embedQuery(text: string): Promise<number[]> {
    const response = await this.request({
      model: this.modelName,
      inputs: [text],
      input_type: "query",
      output_dimension: this.outputDimension,
      output_dtype: this.outputDtype,
    });
    const embedding = response.data[0]?.data[0]?.embedding;
    if (!embedding) {
      throw new Error("Voyage AI API returned no embedding for the query");
    }
    return embedding;
  }

  private async request(
    body: ContextualizedRequest
  ): Promise<VoyageContextualizedResponse> {
    return this.caller.call(async () => {
      const response = await fetch(this.apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify(body),
      });

      let json: unknown;
      try {
        json = await response.json();
      } catch (error) {
        console.error("Failed to parse JSON response:", error);
        json = null;
      }

      if (!response.ok) {
        const err = new Error(
          `Voyage AI API error (HTTP ${response.status}): ${extractErrorMessage(json)}`
        );
        // Lets AsyncCaller skip retries for non-transient (4xx) errors.
        (err as NodeJS.ErrnoException & { status: number }).status =
          response.status;
        throw err;
      }

      return json as VoyageContextualizedResponse;
    });
  }
}
