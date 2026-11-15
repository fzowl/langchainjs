import { getEnvironmentVariable } from "@langchain/core/utils/env";
import { Embeddings, type EmbeddingsParams } from "@langchain/core/embeddings";
import { chunkArray } from "@langchain/core/utils/chunk_array";
import {
  extractErrorMessage,
  getDefaultVoyageBaseUrl,
} from "./voyage_utils.js";

/**
 * Interface that extends EmbeddingsParams and defines additional
 * parameters specific to the VoyageEmbeddings class.
 */
export interface VoyageEmbeddingsParams extends EmbeddingsParams {
  /**
   * The Voyage AI by MongoDB embedding model to use. Current models (default
   * dimension 1024, Matryoshka `outputDimension` 2048/1024/512/256):
   * - `voyage-4-large`, `voyage-4`, `voyage-4-lite`: general-purpose, 32K
   *   token context.
   * - `voyage-code-4`: code retrieval, 32K token context.
   * - `voyage-finance-2` (32K) and `voyage-law-2` (16K): domain-specific,
   *   fixed 1024 dimensions.
   *
   * Earlier models such as `voyage-3.5`, `voyage-3.5-lite`, `voyage-3-large`,
   * `voyage-code-3` and `voyage-3` remain available. For contextualized chunk
   * embeddings (`voyage-context-4`) see `VoyageContextualizedEmbeddings`.
   * @see https://docs.voyageai.com/docs/embeddings
   */
  modelName: string;

  /**
   * Base URL for Voyage API requests. When not set, it is chosen from the API
   * key: keys starting with `al-` (created in the MongoDB Atlas UI) use
   * `'https://ai.mongodb.com/v1'`, all other keys (created on the Voyage AI
   * Dashboard) use `'https://api.voyageai.com/v1'`. A value passed here always
   * takes precedence.
   * @see https://www.mongodb.com/docs/voyageai/management/api-keys/?client-curl-default=curl#create-an-api-key
   */
  basePath?: string;

  /**
   * The maximum number of documents to embed in a single request. The Voyage
   * AI API accepts up to 1000 inputs per request, and also caps the total
   * number of tokens per request (1M for the `-lite` models, 320K for
   * `voyage-4` and `voyage-3.5`, 120K for the others), so keep this value
   * small enough for your document sizes.
   * @default 8
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
 * The shape of a successful response from the Voyage AI embeddings endpoint.
 * @see https://docs.voyageai.com/reference/embeddings-api
 */
interface VoyageEmbeddingResponse {
  object: "list";
  data: Array<{ object: "embedding"; embedding: number[]; index: number }>;
  model: string;
  usage: { total_tokens: number };
}

/**
 * A class for generating embeddings using the Voyage AI by MongoDB API.
 */
export class VoyageEmbeddings
  extends Embeddings
  implements VoyageEmbeddingsParams
{
  modelName = "voyage-3";

  batchSize = 8;

  private apiKey: string;

  /** Do not modify directly. Pass in the basePath option to the constructor. */
  basePath?: string;

  apiUrl: string;

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
    this.basePath =
      fieldsWithDefaults?.basePath ?? getDefaultVoyageBaseUrl(apiKey);
    this.apiUrl = `${this.basePath}/embeddings`;
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
   * Makes a request to the Voyage AI by MongoDB API to generate embeddings for an array of texts.
   * @param request - An object with properties to configure the request.
   * @returns A Promise that resolves to the response from the Voyage AI API.
   */
  private async embeddingWithRetry(
    request: CreateVoyageEmbeddingRequest
  ): Promise<VoyageEmbeddingResponse> {
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

      return json as VoyageEmbeddingResponse;
    };

    return this.caller.call(makeCompletionRequest);
  }
}
