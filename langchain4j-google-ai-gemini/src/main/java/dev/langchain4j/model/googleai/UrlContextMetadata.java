package dev.langchain4j.model.googleai;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public record UrlContextMetadata(List<UrlMetadata> urlMetadata) {

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record UrlMetadata(String retrievedUrl, String urlRetrievalStatus) {}

    static UrlContextMetadata fromGemini(
            GeminiGenerateContentResponse.GeminiUrlContextMetadata geminiUrlContextMetadata) {
        if (geminiUrlContextMetadata == null || geminiUrlContextMetadata.urlMetadata() == null) {
            return null;
        }
        return new UrlContextMetadata(geminiUrlContextMetadata.urlMetadata().stream()
                .map(m -> new UrlMetadata(
                        m.retrievedUrl(),
                        m.urlRetrievalStatus() != null ? m.urlRetrievalStatus().toString() : null))
                .toList());
    }
}
