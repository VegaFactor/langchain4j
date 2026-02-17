package dev.langchain4j.model.googleai;

import java.util.List;

/**
 * URL context metadata from the Gemini API response, containing information about URLs
 * that were retrieved and used for context during generation.
 */
public record UrlContextMetadata(List<UrlMetadata> urlMetadata) {

    /**
     * Metadata about a single URL that was retrieved for context.
     */
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
