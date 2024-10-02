

# SpringAI MongoDB Vector Demo

This project demonstrates how to perform vector similarity search using MongoDB and OpenAI embeddings. Since local MongoDB does not support vector search (e.g., `$vectorSearch`), manual cosine similarity calculation is implemented. MongoDB Atlas, however, supports **Atlas Search**, which includes native vector search features, and is recommended for production use.

## Running MongoDB

To run MongoDB locally, use Docker:

```bash
docker pull mongodb/atlas
docker run -it --name mongodb -p 27017:27017 mongodb/atlas
```

## MongoDB Schema Setup

First, connect to your MongoDB instance and create the database and collection:

```javascript
use springaivectordb
db.createCollection("vector_store")
```

In local MongoDB, vector search is not supported, so instead of using the `createSearchIndexes` command, we manually calculate cosine similarity for stored vectors. Ensure that `spring.ai.vectorstore.mongodb.initialize-schema=false` is set in your configuration.

### Pipeline for Cosine Similarity Calculation

Pipeline finds the top 5 documents in your MongoDB collection whose vector embeddings are most similar to the searchVector based on cosine similarity
```javascript
var searchVector = Array(1536).fill().map(() => Math.random());

db.vector_store.aggregate([
    {
        $addFields: {
            dotProduct: {
                $reduce: {
                    input: {
                        $zip: { inputs: ["$embedding", searchVector] }
                    },
                    initialValue: 0,
                    in: {
                        $add: [
                            "$$value",
                            { $multiply: [{ $arrayElemAt: ["$$this", 0] }, { $arrayElemAt: ["$$this", 1] }] }
                        ]
                    }
                }
            },
            magnitudeA: {
                $sqrt: {
                    $reduce: {
                        input: "$embedding",
                        initialValue: 0,
                        in: {
                            $add: [
                                "$$value",
                                { $multiply: ["$$this", "$$this"] }
                            ]
                        }
                    }
                }
            },
            magnitudeB: {
                $sqrt: {
                    $reduce: {
                        input: searchVector,
                        initialValue: 0,
                        in: {
                            $add: [
                                "$$value",
                                { $multiply: ["$$this", "$$this"] }
                            ]
                        }
                    }
                }
            }
        }
    },
    {
        $addFields: {
            cosineSimilarity: {
                $divide: ["$dotProduct", { $multiply: ["$magnitudeA", "$magnitudeB"] }]
            }
        }
    },
    {
        $sort: { cosineSimilarity: -1 }
    },
    {
        $limit: 5 // Return the top 5 most similar documents
    }
]);
```

## Searching for Similar Documents

We use **OpenAI embeddings** from the class `OpenAIEmbeddingService` to generate query vectors and search for similar documents stored in MongoDB.

The OpenAI embeddings are generated using the model `text-embedding-ada-002`. The query vector is then compared to stored vectors, and cosine similarity is used to rank the documents.

### Main Controller

The main logic for loading documents and performing searches is in `AIController`. The `load` endpoint reads from a file and stores the vectors in MongoDB, while the `search` endpoint calculates the cosine similarity between the input message vector and stored document vectors.

Here’s a summary of the key methods:

- **/load**: Reads text from `input.txt`, splits it into tokens, and stores it in the vector store.
- **/search**: Takes a message as input, generates the query vector using OpenAI, and returns the top similar documents based on cosine similarity.

```java
@GetMapping("/load")
public String load() throws IOException {
    // Load documents from input file and store them in MongoDB
    // Each document is split into tokens and added to the vector store
}

@GetMapping("/search")
public String search(@RequestParam(value = "message", defaultValue = "learn how to grow things") String message) {
    // Generate query vector using OpenAI embeddings
    // Retrieve stored documents from MongoDB
    // Calculate cosine similarity between the query vector and stored document vectors
    // Return the top 2 most similar documents
}
```

## OpenAIEmbeddingService

The `OpenAIEmbeddingService` class is responsible for generating embeddings using OpenAI’s API. The embeddings are used for calculating similarity between the input text and stored vectors.

The service sends a POST request to OpenAI's `/v1/embeddings` API with the input message and model, and then extracts the embedding data from the response.

```java
public double[] generateQueryVector(String message) {
    // Calls OpenAI API and retrieves embeddings for the input message
    // Converts the response into a double[] for further use in similarity calculations
}
```

## Running the Application

1. Clone the repository and navigate to the project directory.
2. Make sure you have Docker and MongoDB installed and running.
3. Run the following Docker command to pull and start the MongoDB Atlas container:

```bash
docker pull mongodb/atlas
docker run -it --name mongodb -p 27017:27017 mongodb/atlas
```

4. Set up your MongoDB schema and load the vector store as described above.
5. Add your OpenAI API key in `application.properties`:

```properties
spring.ai.openai.api-key=your-openai-api-key
```

6. Build and run the application using Maven:

```bash
mvn clean install
mvn spring-boot:run
```

7. Use the `/load` endpoint to load the document vectors into MongoDB and `/search` to search for similar documents based on the input message.

## Conclusion

This project demonstrates how to implement vector search manually using MongoDB and cosine similarity, with vector embeddings generated by OpenAI. While this works with local MongoDB, it is recommended to use **MongoDB Atlas** for full support of vector search with **Atlas Search**.
