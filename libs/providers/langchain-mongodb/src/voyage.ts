import { getEnvironmentVariable } from "@langchain/core/utils/env";
import { Embeddings, type EmbeddingsParams } from "@langchain/core/embeddings";
import { chunkArray } from "@langchain/core/utils/chunk_array";

/**
 * Interface that extends EmbeddingsParams and defines additional
 * parameters specific to the VoyageEmbeddings class.
 */
export interface VoyageEmbeddingsParams extends EmbeddingsParams {
  /**
   * The VoyageAI by MongoDB embedding model to use.
   *
   * Currently available models (see https://docs.voyageai.com/docs/embeddings):
   * - `voyage-4-large` — best general-purpose & multilingual retrieval quality.
   * - `voyage-4` — general-purpose & multilingual retrieval.
   * - `voyage-4-lite` — optimized for latency and cost.
   * - `voyage-4-nano` — open-weight, available on Hugging Face.
   * - `voyage-code-4` — optimized for code retrieval and coding agents.
   * - `voyage-finance-2` — optimized for finance retrieval and RAG.
   * - `voyage-law-2` — optimized for legal retrieval and RAG.
   *
   * For contextualized chunk embeddings use `voyage-context-4` with
   * {@link VoyageEmbeddings.contextualizedEmbed}.
   */
  modelName: string;

  /**
   * Base URL for Voyage API requests.
   * If your API key was created on the MongoDB Atlas UI, it should look like `'https://ai.mongodb.com/v1'`.
   * If your API key was created on the Voyage AI Dashboard, it should look like `'https://api.voyageai.com/v1'`.
   * @default "https://api.voyageai.com/v1"
   * @see https://www.mongodb.com/docs/voyageai/management/api-keys/?client-curl-default=curl#create-an-api-key
   */
  basePath?: string;

  /**
   * The maximum number of documents to embed in a single request. This is
   * limited by the Voyage AI API to a maximum of 8.
   */
  batchSize?: number;

  /**
   * Input type for the embeddings request.
   */
  inputType?: string;

  /**
   * Whether to truncate the input texts to the maximum length allowed by the model.
   */
  truncation?: boolean;

  /**
   * The desired dimension of the output embeddings.
   */
  outputDimension?: number;

  /**
   * The data type of the output embeddings. Can be "float" or "int8".
   */
  outputDtype?: string;

  /**
   * The format of the output embeddings. Can be "float", "base64", or "ubinary".
   */
  encodingFormat?: string;
}

/**
 * Interface for the request body to generate embeddings.
 */
export interface CreateVoyageEmbeddingRequest {
  /**
   * @type {string}
   * @memberof CreateVoyageEmbeddingRequest
   */
  model: string;

  /**
   *  Text to generate vector expectation
   * @type {CreateEmbeddingRequestInput}
   * @memberof CreateVoyageEmbeddingRequest
   */
  input: string | string[];

  /**
   * Input type for the embeddings request.
   */
  input_type?: string;

  /**
   * Whether to truncate the input texts.
   */
  truncation?: boolean;

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
 * The shape of a successful response from the VoyageAI by MongoDB embeddings
 * endpoint.
 * @see https://docs.voyageai.com/reference/embeddings-api
 */
interface VoyageEmbeddingResponse {
  object: "list";
  data: Array<{ object: "embedding"; embedding: number[]; index: number }>;
  model: string;
  usage: { total_tokens: number };
}

/**
 * Options for a contextualized chunk embeddings request.
 * @see https://docs.voyageai.com/docs/contextualized-chunk-embeddings
 */
export interface ContextualizedEmbedOptions {
  /**
   * The contextualized model to use, e.g. `voyage-context-4`.
   * Defaults to the instance `modelName`.
   */
  model?: string;

  /**
   * Input type for the embeddings request. One of `document` or `query`.
   * Required to be `document` when {@link enableAutoChunking} is enabled.
   */
  inputType?: string;

  /**
   * The desired dimension of the output embeddings.
   */
  outputDimension?: number;

  /**
   * The data type of the output embeddings.
   */
  outputDtype?: string;

  /**
   * The format of the output embeddings.
   */
  encodingFormat?: string;

  /**
   * Whether the API should automatically chunk the inputs. Requires a flat
   * `string[]` input and `inputType` set to `document`.
   */
  enableAutoChunking?: boolean;

  /**
   * Target chunk size (in tokens) when auto chunking is enabled.
   */
  chunkSize?: number;

  /**
   * Overlap (in tokens) between consecutive chunks when auto chunking is
   * enabled. Must be smaller than {@link chunkSize}.
   */
  chunkOverlap?: number;
}

/**
 * Interface for the request body sent to the contextualized embeddings
 * endpoint.
 * @see https://docs.voyageai.com/docs/contextualized-chunk-embeddings
 */
export interface CreateVoyageContextualizedEmbeddingRequest {
  model: string;

  /**
   * Input texts to vectorize. Per the official spec both a flat list of
   * strings (`string[]`) and a list of lists of strings (`string[][]`) are
   * supported.
   */
  inputs: string[] | string[][];

  input_type?: string;

  output_dimension?: number;

  output_dtype?: string;

  encoding_format?: string;

  enable_auto_chunking?: boolean;

  chunk_size?: number;

  chunk_overlap?: number;
}

/**
 * The shape of a successful response from the VoyageAI by MongoDB
 * contextualized embeddings endpoint. Each top-level `data` entry corresponds
 * to one input document and holds the per-chunk embeddings for that document.
 * @see https://docs.voyageai.com/reference/contextualized-embeddings-api
 */
interface VoyageContextualizedEmbeddingResponse {
  object: "list";
  data: Array<{
    object: "list";
    data: Array<{ object: "embedding"; embedding: number[]; index: number }>;
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
 * A class for generating embeddings using the VoyageAI by MongoDB API.
 */
export class VoyageEmbeddings
  extends Embeddings
  implements VoyageEmbeddingsParams
{
  modelName = "voyage-3";

  batchSize = 8;

  private apiKey: string;

  /** Do not modify directly. Pass in the basePath option to the constructor. */
  basePath?: string = "https://api.voyageai.com/v1";

  apiUrl: string;

  contextualizedApiUrl: string;

  headers?: Record<string, string>;

  inputType?: string;

  truncation?: boolean;

  outputDimension?: number;

  outputDtype?: string;

  encodingFormat?: string;

  /**
   * Constructor for the VoyageEmbeddings class.
   * @param fields - An optional object with properties to configure the instance.
   */
  constructor(
    fields?: Partial<VoyageEmbeddingsParams> & {
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
    this.batchSize = fieldsWithDefaults?.batchSize ?? this.batchSize;
    this.apiKey = apiKey;
    this.basePath = fieldsWithDefaults?.basePath ?? this.basePath;
    this.apiUrl = `${this.basePath}/embeddings`;
    this.contextualizedApiUrl = `${this.basePath}/contextualizedembeddings`;
    this.inputType = fieldsWithDefaults?.inputType;
    this.truncation = fieldsWithDefaults?.truncation;
    this.outputDimension = fieldsWithDefaults?.outputDimension;
    this.outputDtype = fieldsWithDefaults?.outputDtype;
    this.encodingFormat = fieldsWithDefaults?.encodingFormat;
  }

  /**
   * Generates embeddings for an array of texts.
   * @param texts - An array of strings to generate embeddings for.
   * @returns A Promise that resolves to an array of embeddings.
   */
  async embedDocuments(texts: string[]): Promise<number[][]> {
    const batches = chunkArray(texts, this.batchSize);

    const batchRequests = batches.map((batch) =>
      this.embeddingWithRetry({
        model: this.modelName,
        input: batch,
        input_type: this.inputType,
        truncation: this.truncation,
        output_dimension: this.outputDimension,
        output_dtype: this.outputDtype,
        encoding_format: this.encodingFormat,
      })
    );

    const batchResponses = await Promise.all(batchRequests);

    const embeddings: number[][] = [];

    for (let i = 0; i < batchResponses.length; i += 1) {
      const batch = batches[i];
      const { data: batchResponse } = batchResponses[i];
      for (let j = 0; j < batch.length; j += 1) {
        embeddings.push(batchResponse[j].embedding);
      }
    }

    return embeddings;
  }

  /**
   * Generates an embedding for a single text.
   * @param text - A string to generate an embedding for.
   * @returns A Promise that resolves to an array of numbers representing the embedding.
   */
  async embedQuery(text: string): Promise<number[]> {
    const { data } = await this.embeddingWithRetry({
      model: this.modelName,
      input: text,
      input_type: this.inputType,
      truncation: this.truncation,
      output_dimension: this.outputDimension,
      output_dtype: this.outputDtype,
      encoding_format: this.encodingFormat,
    });

    return data[0].embedding;
  }

  /**
   * Generates contextualized chunk embeddings for one or more documents.
   *
   * Per the official specification the `inputs` parameter accepts both
   * `string[]` (a flat list of texts) and `string[][]` (a list of documents,
   * each already split into chunks) — both formats are supported.
   *
   * The returned value mirrors the input structure: one entry per input
   * document, each holding the ordered per-chunk embeddings for that document.
   *
   * @param inputs - Texts to embed, as `string[]` or `string[][]`.
   * @param options - Optional per-request configuration (model, input type,
   * auto chunking, etc.). Use a contextualized model such as `voyage-context-4`.
   * @returns A Promise resolving to per-document arrays of per-chunk embeddings.
   * @see https://docs.voyageai.com/docs/contextualized-chunk-embeddings
   */
  async contextualizedEmbed(
    inputs: string[] | string[][],
    options?: ContextualizedEmbedOptions
  ): Promise<number[][][]> {
    const { data } = await this.contextualizedEmbeddingWithRetry({
      model: options?.model ?? this.modelName,
      inputs,
      input_type: options?.inputType ?? this.inputType,
      output_dimension: options?.outputDimension ?? this.outputDimension,
      output_dtype: options?.outputDtype ?? this.outputDtype,
      encoding_format: options?.encodingFormat ?? this.encodingFormat,
      enable_auto_chunking: options?.enableAutoChunking,
      chunk_size: options?.chunkSize,
      chunk_overlap: options?.chunkOverlap,
    });

    return [...data]
      .sort((a, b) => a.index - b.index)
      .map((doc) =>
        [...doc.data].sort((a, b) => a.index - b.index).map((c) => c.embedding)
      );
  }

  /**
   * Makes a request to the VoyageAI by MongoDB API to generate embeddings for
   * an array of texts.
   * @param request - An object with properties to configure the request.
   * @returns A Promise that resolves to the response from the API.
   */
  private async embeddingWithRetry(
    request: CreateVoyageEmbeddingRequest
  ): Promise<VoyageEmbeddingResponse> {
    return this.postJson<VoyageEmbeddingResponse>(this.apiUrl, request);
  }

  /**
   * Makes a request to the VoyageAI by MongoDB contextualized embeddings API.
   * @param request - An object with properties to configure the request.
   * @returns A Promise that resolves to the response from the API.
   */
  private async contextualizedEmbeddingWithRetry(
    request: CreateVoyageContextualizedEmbeddingRequest
  ): Promise<VoyageContextualizedEmbeddingResponse> {
    return this.postJson<VoyageContextualizedEmbeddingResponse>(
      this.contextualizedApiUrl,
      request
    );
  }

  /**
   * POSTs a JSON body to the given VoyageAI by MongoDB endpoint through the
   * retrying `AsyncCaller`, returning the parsed response.
   */
  private async postJson<T>(url: string, request: unknown): Promise<T> {
    const makeCompletionRequest = async () => {
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

      return json as T;
    };

    return this.caller.call(makeCompletionRequest);
  }
}
