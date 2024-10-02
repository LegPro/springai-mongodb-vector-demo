package com.example.springaimongodbvectordemo.config;

import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.MongoDBAtlasVectorStore;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.context.annotation.Bean;
import org.springframework.data.mongodb.core.MongoTemplate;

public class AppConfig {
    @Bean
    public VectorStore mongodbVectorStore(MongoTemplate mongoTemplate, EmbeddingModel embeddingModel) {
        return new MongoDBAtlasVectorStore(mongoTemplate, embeddingModel, MongoDBAtlasVectorStore.MongoDBVectorStoreConfig.builder().build(),
                true);
    }
}
