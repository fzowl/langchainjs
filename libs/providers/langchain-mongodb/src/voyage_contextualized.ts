import { getEnvironmentVariable } from "@langchain/core/utils/env";
import { Embeddings, type EmbeddingsParams } from "@langchain/core/embeddings";

/**
 * Interface that extends EmbeddingsParams and defines additional parameters
 * specific to the VoyageContextualizedEmbeddings class.
 *
 * Contextualized chunk embeddings (e.g. `voyage-context-4`) embed each chunk of
 * a document **in the context of the other chunks from the same document**,
 * capturing both local chunk detail and global document-level semantics.
 *
 * This integration relies on VoyageAI by MongoDB server-side auto-chunking:
 * each document is passed to the `/v1/contextualizedembeddings` endpoint as a
 * single string in a flat `list[str]` input, with `enable_auto_chunking: true`
 * and a configurable `chunk_size`, so the service splits every document into
 * chunks for you.
 *
 * @see https://docs.voyageai.com/docs/contextualized-chunk-embeddings
 */
export interface VoyageContextualizedEmbeddingsParams extends EmbeddingsParams {
  /**
   * The VoyageAI by MongoDB contextualized embedding model to use.
   * Currently available models: `voyage-context-4` (recommended),
   * `voyage-context-3` (legacy).
   */
  modelName: string;

  /**
   * Base URL for VoyageAI by MongoDB API requests.
   * If your API key was created on the MongoDB Atlas UI, it should look like `'https://ai.mongodb.com/v1'`.
   * If your API key was created on the Voyage AI Dashboard, it should look like `'https://api.voyageai.com/v1'`.
   * @default "https://api.voyageai.com/v1"
   * @see https://www.mongodb.com/docs/voyageai/management/api-keys/?client-curl-default=curl#create-an-api-key
   */
  basePath?: string;

  /**
   * Input type for the embeddings request. Can be "query" or "document".
   */
  inputType?: string;

  /**
   * Target chunk size in tokens used when the service auto-chunks each
   * document. Must not exceed 32000.
   * @default 32000
   */
  chunkSize?: number;

  /**
   * The desired dimension of the output embeddings.
   * For `voyage-context-4` this can be 256, 512, 1024 (default), or 2048.
   */
  outputDimension?: number;

  /**
   * The data type of the output embeddings. Can be "float" (default), "int8",
   * "uint8", "binary", or "ubinary".
   */
  outputDtype?: string;

  /**
   * The format of the output embeddings. Can be "float" or "base64".
   */
  encodingFormat?: string;
}

/**
 * Interface for the request body sent to the contextualized embeddings
 * endpoint.
 */
export interface CreateVoyageContextualizedEmbeddingRequest {
  /**
   * The model to use, e.g. `voyage-context-4`.
   */
  model: string;

  /**
   * The inputs to embed.
   *
   * When `enable_auto_chunking` is `true` this must be a flat `list[str]` of
   * full-document strings; the service chunks each document server-side. When
   * auto-chunking is disabled it is a nested `list[list[str]]` where each inner
   * list holds one document's pre-computed chunks.
   */
  input: string[] | string[][];

  /**
   * Input type for the embeddings request. Can be "query" or "document".
   * Must be "document" when `enable_auto_chunking` is `true`.
   */
  input_type?: string;

  /**
   * Whether the service should automatically chunk each input document.
   */
  enable_auto_chunking?: boolean;

  /**
   * Target chunk size in tokens when `enable_auto_chunking` is `true`.
   * Must not exceed 32000.
   */
  chunk_size?: number;

  /**
   * The desired dimension of the output embeddings.
   */
  output_dimension?: number;

  /**
   * The data type of the output embeddings.
   */
  output_dtype?: string;

  /**
   * The format of the output embeddings.
   */
  encoding_format?: string;
}

/**
 * A single chunk embedding returned by the contextualized embeddings endpoint.
 */
interface VoyageContextualizedEmbeddingItem {
  object: "embedding";
  embedding: number[];
  index: number;
}

/**
 * The shape of a successful response from the VoyageAI by MongoDB
 * contextualized embeddings endpoint. The top-level `data` array contains one
 * entry per input document, and each entry contains the embeddings for that
 * document's chunks.
 * @see https://docs.voyageai.com/reference/contextualized-embeddings-api
 */
interface VoyageContextualizedEmbeddingResponse {
  object: "list";
  data: Array<{
    object: "list";
    data: VoyageContextualizedEmbeddingItem[];
    index: number;
  }>;
  model: string;
  usage: { total_tokens: number };
}

function extractErrorMessage(body: unknown): string {
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

/**
 * A class for generating contextualized chunk embeddings using the VoyageAI by
 * MongoDB API (e.g. the `voyage-context-4` model).
 *
 * In addition to the standard {@link Embeddings} methods (`embedQuery` and
 * `embedDocuments`), this class exposes {@link contextualizedEmbed} for the
 * contextualized use case: it calls the `contextualized_embed` API with a flat
 * `list[str]` input, `enable_auto_chunking: true` and `chunk_size` (default
 * `32000`), so each document is chunked server-side and every chunk is embedded
 * in the context of its sibling chunks. It returns one embedding per chunk,
 * grouped per input document.
 */
export class VoyageContextualizedEmbeddings
  extends Embeddings
  implements VoyageContextualizedEmbeddingsParams
{
  modelName = "voyage-context-4";

  chunkSize = 32000;

  private apiKey: string;

  /** Do not modify directly. Pass in the basePath option to the constructor. */
  basePath?: string = "https://api.voyageai.com/v1";

  apiUrl: string;

  headers?: Record<string, string>;

  inputType?: string;

  outputDimension?: number;

  outputDtype?: string;

  encodingFormat?: string;

  /**
   * Constructor for the VoyageContextualizedEmbeddings class.
   * @param fields - An optional object with properties to configure the instance.
   */
  constructor(
    fields?: Partial<VoyageContextualizedEmbeddingsParams> & {
      verbose?: boolean;
      apiKey?: string;
      inputType?: string;
      basePath?: string;
    }
  ) {
    const fieldsWithDefaults = { ...fields };

    super(fieldsWithDefaults);

    const apiKey =
      fieldsWithDefaults?.apiKey ??
      getEnvironmentVariable("VOYAGE_API_KEY") ??
      getEnvironmentVariable("VOYAGEAI_API_KEY");

    if (!apiKey) {
      throw new Error("Voyage AI API key not found");
    }

    this.modelName = fieldsWithDefaults?.modelName ?? this.modelName;
    this.chunkSize = fieldsWithDefaults?.chunkSize ?? this.chunkSize;
    this.apiKey = apiKey;
    this.basePath = fieldsWithDefaults?.basePath ?? this.basePath;
    this.apiUrl = `${this.basePath}/contextualizedembeddings`;
    this.inputType = fieldsWithDefaults?.inputType;
    this.outputDimension = fieldsWithDefaults?.outputDimension;
    this.outputDtype = fieldsWithDefaults?.outputDtype;
    this.encodingFormat = fieldsWithDefaults?.encodingFormat;
  }

  /**
   * Generates embeddings for an array of full-document strings.
   *
   * Each document is auto-chunked server-side; because the {@link Embeddings}
   * contract expects a single vector per input, the first (leading) chunk
   * embedding of each document is returned. Use {@link contextualizedEmbed} to
   * retrieve every chunk embedding grouped per document.
   * @param texts - An array of full-document strings to embed.
   * @returns A Promise that resolves to one embedding per input document.
   */
  async embedDocuments(texts: string[]): Promise<number[][]> {
    const grouped = await this.contextualizedEmbed(
      texts,
      this.inputType ?? "document"
    );

    return grouped.map((chunks) => chunks[0]);
  }

  /**
   * Generates an embedding for a single query text.
   *
   * Queries are short and embedded as a single chunk, so auto-chunking is not
   * used for this path.
   * @param text - A string to generate an embedding for.
   * @returns A Promise that resolves to the embedding vector.
   */
  async embedQuery(text: string): Promise<number[]> {
    const response = await this.embeddingWithRetry({
      model: this.modelName,
      input: [[text]],
      input_type: "query",
      enable_auto_chunking: false,
      output_dimension: this.outputDimension,
      output_dtype: this.outputDtype,
      encoding_format: this.encodingFormat,
    });

    return response.data[0].data[0].embedding;
  }

  /**
   * Generates contextualized chunk embeddings for a list of full-document
   * strings using VoyageAI by MongoDB server-side auto-chunking.
   *
   * The `contextualized_embed` API is invoked with a flat `list[str]` input,
   * `enable_auto_chunking: true` and `chunk_size` (default `32000`), so every
   * document is chunked server-side and each chunk is embedded in the context
   * of the other chunks from the same document.
   * @param documents - An array of full-document strings to embed.
   * @param inputType - Optional override for the request input type. Auto-chunking requires "document".
   * @returns A Promise that resolves to a nested array of embeddings: one
   *   embedding per chunk, grouped per input document.
   */
  async contextualizedEmbed(
    documents: string[],
    inputType?: string
  ): Promise<number[][][]> {
    const response = await this.embeddingWithRetry({
      model: this.modelName,
      input: documents,
      input_type: inputType ?? this.inputType ?? "document",
      enable_auto_chunking: true,
      chunk_size: this.chunkSize,
      output_dimension: this.outputDimension,
      output_dtype: this.outputDtype,
      encoding_format: this.encodingFormat,
    });

    return response.data.map((doc) => doc.data.map((chunk) => chunk.embedding));
  }

  /**
   * Makes a request to the VoyageAI by MongoDB contextualized embeddings endpoint.
   * @param request - An object with properties to configure the request.
   * @returns A Promise that resolves to the response from the VoyageAI by MongoDB API.
   */
  private async embeddingWithRetry(
    request: CreateVoyageContextualizedEmbeddingRequest
  ): Promise<VoyageContextualizedEmbeddingResponse> {
    const makeCompletionRequest = async () => {
      const url = `${this.apiUrl}`;
      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${this.apiKey}`,
          ...this.headers,
        },
        body: JSON.stringify(request),
      });

      let json: unknown;
      try {
        json = await response.json();
      } catch (error) {
        console.error("Failed to parse JSON response:", error);
        json = null;
      }

      if (!response.ok) {
        const message = extractErrorMessage(json);
        const err = new Error(
          `Voyage AI API error (HTTP ${response.status}): ${message}`
        );
        // Attach status so AsyncCaller's defaultFailedAttemptHandler can
        // skip retries for non-transient HTTP errors (4xx).
        (err as NodeJS.ErrnoException & { status: number }).status =
          response.status;
        throw err;
      }

      return json as VoyageContextualizedEmbeddingResponse;
    };

    return this.caller.call(makeCompletionRequest);
  }
}
