package com.example.springaimongodbvectordemo.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.List;
import java.util.Map;

@Service
public class OpenAIEmbeddingService {

    @Value("${spring.ai.openai.api-key}")
    private String apiKey;

    private static final String OPENAI_EMBEDDING_ENDPOINT = "https://api.openai.com/v1/embeddings";
    private static final String MODEL = "text-embedding-ada-002";  // You can change the model to your preferred version

    // Method to generate embeddings using OpenAI's API
    public double[] generateQueryVector(String message) {
        RestTemplate restTemplate = new RestTemplate();

        // Prepare the request headers
        HttpHeaders headers = new HttpHeaders();
        headers.set("Authorization", "Bearer " + apiKey);
        headers.set("Content-Type", "application/json");

        // Prepare the request body
        String requestBody = String.format("{\"input\": \"%s\", \"model\": \"%s\"}", message, MODEL);

        // Prepare the HTTP entity
        HttpEntity<String> requestEntity = new HttpEntity<>(requestBody, headers);

        // Call the OpenAI embedding API
        ResponseEntity<Map> response = restTemplate.exchange(OPENAI_EMBEDDING_ENDPOINT, HttpMethod.POST, requestEntity, Map.class);

        // Extract embedding data from the response
        Map<String, Object> responseBody = response.getBody();
        List<Double> embeddingList = (List<Double>) ((Map<String, Object>) ((List<?>) responseBody.get("data")).get(0)).get("embedding");

        // Convert List<Double> to double[]
        return embeddingList.stream().mapToDouble(Double::doubleValue).toArray();
    }
}
