package io.legado.app.data.repository.ai

import io.legado.app.domain.model.AiCapability
import io.legado.app.domain.model.AiGenerateRequest
import io.legado.app.domain.model.AiGenerationParams
import io.legado.app.domain.model.AiMessage
import io.legado.app.domain.model.AiMessageRole
import io.legado.app.domain.model.AiModelConfig
import io.legado.app.domain.model.AiProtocol
import io.legado.app.domain.model.AiProviderConfig
import io.legado.app.domain.model.AiReasoningLevel
import io.legado.app.domain.model.AiToolCall
import kotlinx.coroutines.channels.Channel
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import okio.Buffer

class BedrockConverseHandlerTest {

    private fun request(
        modelId: String,
        level: AiReasoningLevel,
        maxTokens: Int? = null,
        messages: List<AiMessage> = listOf(AiMessage(AiMessageRole.USER, "hi")),
    ) = AiGenerateRequest(
        model = AiModelConfig(
            id = "m",
            provider = AiProviderConfig("p", "AWS Bedrock", AiProtocol.AWS_BEDROCK_CONVERSE, "https://bedrock-runtime.us-east-1.amazonaws.com", "key"),
            displayName = modelId,
            modelId = modelId,
            capabilities = setOf(AiCapability.REASONING),
        ),
        messages = messages,
        params = AiGenerationParams(temperature = 0.3f, maxOutputTokens = maxTokens, topP = 0.9f, reasoningLevel = level),
    )

    @Test
    fun claudeThinkingDefaultsToAdaptiveEffortAndDropsSampling() {
        val config = mutableMapOf<String, Any>("maxTokens" to 2048, "temperature" to 0.3f, "topP" to 0.9f)

        val fields = request("us.anthropic.claude-sonnet-5-5", AiReasoningLevel.HIGH).bedrockReasoningFields(config)

        assertEquals(
            mapOf("thinking" to mapOf("type" to "adaptive"), "output_config" to mapOf("effort" to "high")),
            fields
        )
        assertEquals(mapOf<String, Any>("maxTokens" to 2048), config)
        assertEquals(
            mapOf("effort" to "max"),
            request("us.anthropic.claude-sonnet-5-5", AiReasoningLevel.MAX).bedrockReasoningFields(mutableMapOf())
                ?.get("output_config")
        )
    }

    @Test
    fun olderClaudeShapeUsesBudgetAboveMaxTokens() {
        val config = mutableMapOf<String, Any>("maxTokens" to 2048, "temperature" to 0.3f, "topP" to 0.9f)

        val fields = request("us.anthropic.claude-sonnet-4-5-20250929-v1:0", AiReasoningLevel.HIGH)
            .bedrockReasoningFields(config, BedrockRequestShape(adaptiveThinking = false))

        assertEquals(mapOf("thinking" to mapOf("type" to "enabled", "budget_tokens" to 8_000)), fields)
        assertEquals(mapOf<String, Any>("maxTokens" to 12_096), config)
    }

    @Test
    fun claudeSendsNoThinkingWhenOffOrAfterToolCalls() {
        val messages = listOf(
            AiMessage(AiMessageRole.USER, "hi"),
            AiMessage(AiMessageRole.ASSISTANT, "", toolCalls = listOf(AiToolCall("c", "t", "{}"))),
            AiMessage(AiMessageRole.TOOL, "ok", toolCallId = "c"),
        )
        assertNull(
            request("us.anthropic.claude-sonnet-5-5", AiReasoningLevel.MEDIUM, messages = messages)
                .bedrockReasoningFields(mutableMapOf())
        )
        assertNull(request("us.anthropic.claude-sonnet-5-5", AiReasoningLevel.OFF).bedrockReasoningFields(mutableMapOf()))
    }

    /** Error messages returned by Bedrock on 2026-11-01 for the models in the comment on [BedrockRequestShape]. */
    @Test
    fun requestShapeAdjustsToBedrockErrors() {
        val start = BedrockRequestShape()
        val temperature = "HTTP 400: {\"message\":\"The model returned the following errors: `temperature` is deprecated for this model.\"}"
        val adaptive = "HTTP 400: {\"message\":\"The model returned the following errors: adaptive thinking is not supported on this model\"}"
        val effort = "HTTP 400: {\"message\":\"The model returned the following errors: output_config.effort: Extra inputs are not permitted\"}"
        val enabled = "HTTP 400: {\"message\":\"The model returned the following errors: \\\"thinking.type.enabled\\\" is not supported for this model.\"}"
            .replace("\\\"", "\"")

        assertEquals(BedrockRequestShape(sampling = false), start.adjustFor(temperature))
        assertEquals(BedrockRequestShape(adaptiveThinking = false), start.adjustFor(adaptive))
        assertEquals(BedrockRequestShape(adaptiveThinking = false), start.adjustFor(effort))
        assertEquals(start, BedrockRequestShape(adaptiveThinking = false).adjustFor(enabled))
        assertNull(start.adjustFor("HTTP 400: {\"message\":\"Malformed input request\"}"))
        assertNull(BedrockRequestShape(sampling = false).adjustFor(temperature))
    }

    @Test
    fun bodyOmitsSamplingWhenShapeSaysSo() {
        val noReasoning = request("us.anthropic.claude-sonnet-5-5", AiReasoningLevel.AUTO, maxTokens = 500)

        assertEquals(
            mapOf("maxTokens" to 500, "temperature" to 0.3f, "topP" to 0.9f),
            noReasoning.toConverseBody()["inferenceConfig"]
        )
        assertEquals(
            mapOf("maxTokens" to 500),
            noReasoning.toConverseBody(BedrockRequestShape(sampling = false))["inferenceConfig"]
        )
    }

    @Test
    fun novaAndGptOssUseTheirOwnReasoningFields() {
        val novaConfig = mutableMapOf<String, Any>("maxTokens" to 1000, "temperature" to 0.3f)
        assertEquals(
            mapOf("reasoningConfig" to mapOf("type" to "enabled", "maxReasoningEffort" to "high")),
            request("us.amazon.nova-2-lite-v1:0", AiReasoningLevel.MAX).bedrockReasoningFields(novaConfig)
        )
        assertEquals(emptyMap<String, Any>(), novaConfig)

        assertEquals(
            mapOf("reasoning_effort" to "low"),
            request("openai.gpt-oss-120b-1:0", AiReasoningLevel.LOW).bedrockReasoningFields(mutableMapOf())
        )
    }

    @Test
    fun autoLevelOrOtherModelsSendNoReasoningFields() {
        assertNull(request("us.anthropic.claude-sonnet-5-5", AiReasoningLevel.AUTO).bedrockReasoningFields(mutableMapOf()))
        assertNull(request("meta.llama3-70b-instruct-v1:0", AiReasoningLevel.HIGH).bedrockReasoningFields(mutableMapOf()))
    }

    @Test
    fun systemMessagesAreDroppedAndSameRoleTurnsMerge() {
        val messages = listOf(
            AiMessage(AiMessageRole.SYSTEM, "be brief"),
            AiMessage(AiMessageRole.USER, "hi"),
            AiMessage(
                AiMessageRole.ASSISTANT,
                "",
                toolCalls = listOf(AiToolCall("call_1", "lookup", """{"q":"x"}""")),
            ),
            AiMessage(AiMessageRole.TOOL, "result 1", toolCallId = "call_1"),
            AiMessage(AiMessageRole.USER, "and then?"),
        ).toBedrockMessages()

        assertEquals(listOf("user", "assistant", "user"), messages.map { it["role"] })
        // The blank assistant text is skipped; only the toolUse block remains.
        assertEquals(1, (messages[1]["content"] as List<*>).size)
        // The tool result and the following user text share one user turn.
        assertEquals(2, (messages[2]["content"] as List<*>).size)
    }

    @Test
    fun parsesFoundationModelsAndInferenceProfiles() {
        val models = """
            {"modelSummaries":[
              {"modelId":"amazon.nova-lite-v1:0","modelName":"Nova Lite","providerName":"Amazon","outputModalities":["TEXT"]},
              {"modelId":"amazon.titan-image-generator-v2:0","modelName":"Titan Image","outputModalities":["IMAGE"]}
            ],
            "inferenceProfileSummaries":[
              {"inferenceProfileId":"us.anthropic.claude-sonnet-4-20250514-v1:0","inferenceProfileName":"US Claude Sonnet 4"}
            ]}
        """.trimIndent().toJsonObject().toBedrockModels()

        assertEquals(
            listOf("amazon.nova-lite-v1:0", "us.anthropic.claude-sonnet-4-20250514-v1:0"),
            models.map { it.id }
        )
        assertEquals("Amazon Nova Lite", models.first().name)
    }

    @Test
    fun decodesEventStreamFrames() {
        val source = Buffer()
            .write(frame("contentBlockDelta", """{"contentBlockIndex":0,"delta":{"text":"Hel"}}"""))
            .write(frame("contentBlockDelta", """{"contentBlockIndex":0,"delta":{"text":"lo"}}"""))
        val items = Channel<EventStreamItem>(Channel.UNLIMITED)

        pumpEventStream(source, items)

        val first = items.tryReceive().getOrThrow() as EventStreamItem.Message
        val second = items.tryReceive().getOrThrow() as EventStreamItem.Message
        assertEquals("event", first.headers[":message-type"])
        assertEquals("contentBlockDelta", first.headers[":event-type"])
        assertEquals("""{"contentBlockIndex":0,"delta":{"text":"lo"}}""", second.payload)
    }

    /** Builds one event stream frame with zeroed CRCs (the reader does not verify them). */
    private fun frame(eventType: String, payload: String): ByteArray {
        val headers = Buffer()
        listOf(":message-type" to "event", ":event-type" to eventType).forEach { (name, value) ->
            headers.writeByte(name.length).writeUtf8(name)
            headers.writeByte(7).writeShort(value.length).writeUtf8(value)
        }
        val headerBytes = headers.readByteArray()
        val payloadBytes = payload.toByteArray()
        return Buffer()
            .writeInt(12 + headerBytes.size + payloadBytes.size + 4)
            .writeInt(headerBytes.size)
            .writeInt(0)
            .write(headerBytes)
            .write(payloadBytes)
            .writeInt(0)
            .readByteArray()
    }
}
