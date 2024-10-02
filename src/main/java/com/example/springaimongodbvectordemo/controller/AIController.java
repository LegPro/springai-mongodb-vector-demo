package com.example.springaimongodbvectordemo.controller;

import com.example.springaimongodbvectordemo.service.OpenAIEmbeddingService;
import com.mongodb.client.MongoClients;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoCursor;
import com.mongodb.client.MongoDatabase;
import org.bson.Document;
import org.springframework.ai.transformer.splitter.TextSplitter;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

@RestController
public class AIController {

    @Value("classpath:input.txt")
    Resource resourceFile;

    VectorStore vectorStore;
    OpenAIEmbeddingService openAIEmbeddingService;
    public AIController(VectorStore vectorStore, OpenAIEmbeddingService openAIEmbeddingService) {
        this.vectorStore = vectorStore;
        this.openAIEmbeddingService = openAIEmbeddingService;
    }

    @GetMapping("/load")
    public String load() throws IOException {
        List<org.springframework.ai.document.Document> documents = Files.lines(resourceFile.getFile().toPath())
                .map(org.springframework.ai.document.Document::new)
                .toList();
        TextSplitter textSplitter = new TokenTextSplitter();
        for(int i=0;i<documents.size();i++){
            List<org.springframework.ai.document.Document> splittedDocs = textSplitter.split(documents.get(i));
            vectorStore.add(splittedDocs);
            System.out.println("Document "+documents.get(i).getContent()+" added to vector store");
        }
        return resourceFile.getFilename();
    }

    @GetMapping("/search")
    public String search(@RequestParam(value = "message", defaultValue = "learn how to grow things") String message) {
        // Connect to MongoDB to fetch stored documents with embeddings
        MongoDatabase database = MongoClients.create("mongodb://localhost:27017")
                .getDatabase("springaivectordb");
        MongoCollection<Document> collection = database.getCollection("vector_store");

        //  query vector generates this using OpenAI
        double[] queryVector = openAIEmbeddingService.generateQueryVector(message);

        // Retrieve documents and their embeddings from MongoDB
        List<SimilarityResult> similarityResults = new ArrayList<>();
        try (MongoCursor<Document> cursor = collection.find().iterator()) {
            while (cursor.hasNext()) {
                Document doc = cursor.next();
                List<Double> embeddingList = (List<Double>) doc.get("embedding");
                double[] embeddingArray = embeddingList.stream().mapToDouble(Double::doubleValue).toArray();

                // Calculate cosine similarity
                double similarity = cosineSimilarity(queryVector, embeddingArray);
                String metadata = doc.getString("content");

                similarityResults.add(new SimilarityResult(metadata, similarity));
            }
        }

        // Sort by cosine similarity in descending order
        Collections.sort(similarityResults);

        // Return the top 2 most similar documents as a response
        return similarityResults.stream()
                .limit(2) // Top 2 results
                .map(result -> result.metadata + " - Cosine Similarity: " + result.similarity)
                .collect(Collectors.joining(", "));
    }

    // Function to calculate cosine similarity between two vectors
    public static double cosineSimilarity(double[] vectorA, double[] vectorB) {
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        for (int i = 0; i < vectorA.length; i++) {
            dotProduct += vectorA[i] * vectorB[i];
            normA += Math.pow(vectorA[i], 2);
            normB += Math.pow(vectorB[i], 2);
        }
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }


    // Class to store results
    public static class SimilarityResult implements Comparable<SimilarityResult> {
        String metadata;
        double similarity;

        public SimilarityResult(String metadata, double similarity) {
            this.metadata = metadata;
            this.similarity = similarity;
        }

        @Override
        public int compareTo(SimilarityResult o) {
            return Double.compare(o.similarity, this.similarity); // Sort in descending order
        }
    }
}
