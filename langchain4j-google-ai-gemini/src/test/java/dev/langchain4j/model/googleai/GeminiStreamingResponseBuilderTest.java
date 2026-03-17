package dev.langchain4j.model.googleai;

import static org.assertj.core.api.Assertions.assertThat;

import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.googleai.GeminiGenerateContentResponse.GeminiCandidate;
import dev.langchain4j.model.googleai.GeminiGenerateContentResponse.GeminiCandidate.GeminiFinishReason;
import dev.langchain4j.model.googleai.GeminiGenerateContentResponse.GeminiUrlContextMetadata;
import dev.langchain4j.model.googleai.GeminiGenerateContentResponse.GeminiUrlMetadata;
import dev.langchain4j.model.googleai.GeminiGenerateContentResponse.GeminiUrlRetrievalStatus;
import dev.langchain4j.model.googleai.GeminiGenerateContentResponse.GeminiUsageMetadata;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class GeminiStreamingResponseBuilderTest {

    private GeminiStreamingResponseBuilder builder;

    @BeforeEach
    void setUp() {
        builder = new GeminiStreamingResponseBuilder(false, null);
    }

    @Test
    void shouldPreserveGroundingMetadataFromFinalChunk() {
        // Given: first chunk with text only
        builder.append(createTextChunk("Hello "));

        // And: final chunk with grounding metadata at response level
        GroundingMetadata grounding = GroundingMetadata.builder()
                .webSearchQueries(List.of("test query"))
                .groundingChunks(List.of(new GroundingMetadata.GroundingChunk(
                        new GroundingMetadata.GroundingChunk.Web("https://example.com", "Example"), null, null)))
                .build();

        builder.append(createFinalChunkWithResponseGrounding("world", grounding));

        // When
        ChatResponse response = builder.build();

        // Then
        GoogleAiGeminiChatResponseMetadata metadata =
                (GoogleAiGeminiChatResponseMetadata) response.metadata();
        assertThat(metadata.groundingMetadata()).isNotNull();
        assertThat(metadata.groundingMetadata().webSearchQueries()).containsExactly("test query");
        assertThat(metadata.groundingMetadata().groundingChunks()).hasSize(1);
        assertThat(metadata.groundingMetadata().groundingChunks().get(0).web().uri())
                .isEqualTo("https://example.com");
    }

    @Test
    void shouldPreserveUrlContextMetadataFromCandidate() {
        // Given: chunk with URL context metadata on the candidate
        GeminiUrlMetadata urlMeta =
                new GeminiUrlMetadata("https://example.com/page", GeminiUrlRetrievalStatus.URL_RETRIEVAL_STATUS_SUCCESS);
        GeminiUrlContextMetadata urlContextMeta = new GeminiUrlContextMetadata(List.of(urlMeta));

        builder.append(createChunkWithUrlContext("result text", urlContextMeta));

        // When
        ChatResponse response = builder.build();

        // Then
        GoogleAiGeminiChatResponseMetadata metadata =
                (GoogleAiGeminiChatResponseMetadata) response.metadata();
        assertThat(metadata.urlContextMetadata()).isNotNull();
        assertThat(metadata.urlContextMetadata().urlMetadata()).hasSize(1);
        assertThat(metadata.urlContextMetadata().urlMetadata().get(0).retrievedUrl())
                .isEqualTo("https://example.com/page");
        assertThat(metadata.urlContextMetadata().urlMetadata().get(0).urlRetrievalStatus())
                .isEqualTo("URL_RETRIEVAL_STATUS_SUCCESS");
    }

    @Test
    void shouldFallBackToCandidateLevelGroundingMetadata() {
        // Given: chunk with grounding metadata on candidate (not response level)
        GroundingMetadata candidateGrounding = GroundingMetadata.builder()
                .webSearchQueries(List.of("candidate query"))
                .build();

        builder.append(createChunkWithCandidateGrounding("text", candidateGrounding));

        // When
        ChatResponse response = builder.build();

        // Then
        GoogleAiGeminiChatResponseMetadata metadata =
                (GoogleAiGeminiChatResponseMetadata) response.metadata();
        assertThat(metadata.groundingMetadata()).isNotNull();
        assertThat(metadata.groundingMetadata().webSearchQueries()).containsExactly("candidate query");
    }

    @Test
    void shouldPreferResponseLevelGroundingOverCandidateLevel() {
        // Given: chunk with both response-level and candidate-level grounding
        GroundingMetadata responseGrounding = GroundingMetadata.builder()
                .webSearchQueries(List.of("response query"))
                .build();
        GroundingMetadata candidateGrounding = GroundingMetadata.builder()
                .webSearchQueries(List.of("candidate query"))
                .build();

        builder.append(createChunkWithBothGrounding("text", responseGrounding, candidateGrounding));

        // When
        ChatResponse response = builder.build();

        // Then
        GoogleAiGeminiChatResponseMetadata metadata =
                (GoogleAiGeminiChatResponseMetadata) response.metadata();
        assertThat(metadata.groundingMetadata()).isNotNull();
        assertThat(metadata.groundingMetadata().webSearchQueries()).containsExactly("response query");
    }

    @Test
    void shouldNotDowngradeResponseLevelGroundingWithLaterCandidateLevelGrounding() {
        // Given: first chunk carries response-level grounding
        GroundingMetadata responseGrounding = GroundingMetadata.builder()
                .webSearchQueries(List.of("response query"))
                .build();
        builder.append(createFinalChunkWithResponseGrounding("first ", responseGrounding));

        // And: a subsequent chunk carries only candidate-level grounding (no response-level)
        GroundingMetadata candidateGrounding = GroundingMetadata.builder()
                .webSearchQueries(List.of("candidate query"))
                .build();
        builder.append(createChunkWithCandidateGrounding("second", candidateGrounding));

        // When
        ChatResponse response = builder.build();

        // Then: the response-level grounding from the earlier chunk must not be overwritten
        GoogleAiGeminiChatResponseMetadata metadata =
                (GoogleAiGeminiChatResponseMetadata) response.metadata();
        assertThat(metadata.groundingMetadata()).isNotNull();
        assertThat(metadata.groundingMetadata().webSearchQueries()).containsExactly("response query");
    }

    @Test
    void shouldHaveNullMetadataWhenAbsentFromAllChunks() {
        // Given: chunks with no grounding or URL context metadata
        builder.append(createTextChunk("Hello "));
        builder.append(createTextChunk("world"));

        // When
        ChatResponse response = builder.build();

        // Then
        GoogleAiGeminiChatResponseMetadata metadata =
                (GoogleAiGeminiChatResponseMetadata) response.metadata();
        assertThat(metadata.groundingMetadata()).isNull();
        assertThat(metadata.urlContextMetadata()).isNull();
    }

    @Test
    void shouldAccumulateTextWhilePreservingMetadata() {
        // Given: multiple text chunks, then final chunk with metadata
        builder.append(createTextChunk("Hello "));
        builder.append(createTextChunk("beautiful "));

        GroundingMetadata grounding = GroundingMetadata.builder()
                .webSearchQueries(List.of("search"))
                .build();
        builder.append(createFinalChunkWithResponseGrounding("world", grounding));

        // When
        ChatResponse response = builder.build();

        // Then
        assertThat(response.aiMessage().text()).isEqualTo("Hello beautiful world");
        GoogleAiGeminiChatResponseMetadata metadata =
                (GoogleAiGeminiChatResponseMetadata) response.metadata();
        assertThat(metadata.groundingMetadata()).isNotNull();
        assertThat(metadata.groundingMetadata().webSearchQueries()).containsExactly("search");
    }

    // --- Helper methods ---

    private GeminiGenerateContentResponse createTextChunk(String text) {
        return new GeminiGenerateContentResponse(
                "resp-id",
                "gemini-2.0-flash",
                List.of(createCandidate(text, null, null, null)),
                createUsageMetadata(),
                null);
    }

    private GeminiGenerateContentResponse createFinalChunkWithResponseGrounding(
            String text, GroundingMetadata responseGrounding) {
        return new GeminiGenerateContentResponse(
                "resp-id",
                "gemini-2.0-flash",
                List.of(createCandidate(text, GeminiFinishReason.STOP, null, null)),
                createUsageMetadata(),
                responseGrounding);
    }

    private GeminiGenerateContentResponse createChunkWithCandidateGrounding(
            String text, GroundingMetadata candidateGrounding) {
        return new GeminiGenerateContentResponse(
                "resp-id",
                "gemini-2.0-flash",
                List.of(createCandidate(text, GeminiFinishReason.STOP, null, candidateGrounding)),
                createUsageMetadata(),
                null);
    }

    private GeminiGenerateContentResponse createChunkWithBothGrounding(
            String text, GroundingMetadata responseGrounding, GroundingMetadata candidateGrounding) {
        return new GeminiGenerateContentResponse(
                "resp-id",
                "gemini-2.0-flash",
                List.of(createCandidate(text, GeminiFinishReason.STOP, null, candidateGrounding)),
                createUsageMetadata(),
                responseGrounding);
    }

    private GeminiGenerateContentResponse createChunkWithUrlContext(
            String text, GeminiUrlContextMetadata urlContextMetadata) {
        return new GeminiGenerateContentResponse(
                "resp-id",
                "gemini-2.0-flash",
                List.of(createCandidate(text, GeminiFinishReason.STOP, urlContextMetadata, null)),
                createUsageMetadata(),
                null);
    }

    private GeminiCandidate createCandidate(
            String text,
            GeminiFinishReason finishReason,
            GeminiUrlContextMetadata urlContextMetadata,
            GroundingMetadata groundingMetadata) {
        GeminiContent content = new GeminiContent(
                List.of(GeminiContent.GeminiPart.builder().text(text).build()), "model");
        return new GeminiCandidate(content, finishReason, urlContextMetadata, groundingMetadata);
    }

    private GeminiUsageMetadata createUsageMetadata() {
        return GeminiUsageMetadata.builder()
                .promptTokenCount(10)
                .candidatesTokenCount(20)
                .totalTokenCount(30)
                .build();
    }
}
