# @langchain/mongodb

This package contains the LangChain.js integrations for MongoDB through their SDK.

## Installation

```bash npm2yarn
npm install @langchain/mongodb @langchain/core
```

This package, along with the main LangChain package, depends on [`@langchain/core`](https://npmjs.com/package/@langchain/core/).
If you are using this package with other LangChain packages, you should make sure that all of the packages depend on the same instance of @langchain/core.
You can do so by adding appropriate field to your project's `package.json` like this:

```json
{
  "name": "your-project",
  "version": "0.0.0",
  "dependencies": {
    "@langchain/core": "^0.3.0",
    "@langchain/mongodb": "^0.0.0"
  },
  "resolutions": {
    "@langchain/core": "^0.3.0"
  },
  "overrides": {
    "@langchain/core": "^0.3.0"
  },
  "pnpm": {
    "overrides": {
      "@langchain/core": "^0.3.0"
    }
  }
}
```

The field you need depends on the package manager you're using, but we recommend adding a field for the common `yarn`, `npm`, and `pnpm` to maximize compatibility.

## VoyageAI by MongoDB embeddings

This package ships `VoyageEmbeddings`, a LangChain embeddings integration for the
[VoyageAI by MongoDB](https://docs.voyageai.com/) API. An API key can be created
either from the MongoDB Atlas UI or from the Voyage AI Dashboard; set it via the
`VOYAGE_API_KEY` (or `VOYAGEAI_API_KEY`) environment variable, or pass `apiKey`
to the constructor.

```typescript
import { VoyageEmbeddings } from "@langchain/mongodb";

const embeddings = new VoyageEmbeddings({ modelName: "voyage-4" });

const vector = await embeddings.embedQuery("What is MongoDB Atlas?");
const vectors = await embeddings.embedDocuments(["doc one", "doc two"]);
```

If your API key was created on the MongoDB Atlas UI, point `basePath` at the
Atlas endpoint:

```typescript
const embeddings = new VoyageEmbeddings({
  modelName: "voyage-4",
  basePath: "https://ai.mongodb.com/v1",
});
```

### Available models

The currently available models are listed at
[docs.voyageai.com/docs/embeddings](https://docs.voyageai.com/docs/embeddings):

| Model              | Context length | Notes                                                 |
| ------------------ | -------------- | ----------------------------------------------------- |
| `voyage-4-large`   | 32,000         | Best general-purpose & multilingual retrieval quality |
| `voyage-4`         | 32,000         | General-purpose & multilingual retrieval              |
| `voyage-4-lite`    | 32,000         | Optimized for latency and cost                        |
| `voyage-4-nano`    | 32,000         | Open-weight, available on Hugging Face                |
| `voyage-code-4`    | 32,000         | Optimized for code retrieval and coding agents        |
| `voyage-finance-2` | 32,000         | Optimized for finance retrieval and RAG               |
| `voyage-law-2`     | 16,000         | Optimized for legal retrieval and RAG                 |
| `voyage-context-4` | 32,000         | Contextualized chunk embeddings (see below)           |

### Contextualized chunk embeddings

For contextualized chunk embeddings use a contextualized model such as
`voyage-context-4` with `contextualizedEmbed`. Per the
[official specification](https://docs.voyageai.com/docs/contextualized-chunk-embeddings)
the inputs may be either a flat list of strings (`string[]`) or a list of
documents already split into chunks (`string[][]`) — both formats are supported.
The result mirrors the input: one entry per document, each holding the ordered
per-chunk embeddings.

```typescript
const embeddings = new VoyageEmbeddings({ modelName: "voyage-context-4" });

// string[][] — documents already split into chunks
const perDocument = await embeddings.contextualizedEmbed(
  [["Chunk 1 of doc A.", "Chunk 2 of doc A."], ["Chunk 1 of doc B."]],
  { inputType: "document" }
);

// string[] — a flat list, e.g. with server-side auto chunking
const autoChunked = await embeddings.contextualizedEmbed(
  ["A long document to be chunked by the API."],
  { inputType: "document", enableAutoChunking: true, chunkSize: 512 }
);
```

## Development

To develop the MongoDB package, you'll need to follow these instructions:

### Install dependencies

```bash
pnpm install
```

### Build the package

```bash
pnpm build
```

Or from the repo root:

```bash
pnpm build --filter @langchain/mongodb
```

### Run tests

Test files should live within a `tests/` file in the `src/` folder. Unit tests should end in `.test.ts` and integration tests should
end in `.int.test.ts`:

```bash
$ pnpm test
$ pnpm test:int
```

The tests in this package require an instance of MongoDB Atlas running, either running locally or as a remote Atlas cluster. A URI pointing to
an existing Atlas cluster can be provided to the tests by specifying the `MONGODB_ATLAS_URI` environment variable:

```bash
MONGODB_ATLAS_URI='<atlas URI>' pnpm test:int
```

If running against a remote Atlas cluster, the user must have readWrite permissions on the `langchain_test` database.

If no `MONGODB_ATLAS_URI` is provided, the test suite will attempt to launch an instance of local Atlas in a container using [testcontainers](https://testcontainers.com/). This requires a container engine, see the [testcontainer backing engine documentation](https://node.testcontainers.org/supported-container-runtimes/) for details.

### Lint & Format

Run the linter & formatter to ensure your code is up to standard:

```bash
pnpm lint && pnpm format
```

### Adding new entrypoints

If you add a new file to be exported, either import & re-export from `src/index.ts`, or add it to the `exports` field in the `package.json` file and run `pnpm build` to generate the new entrypoint.
