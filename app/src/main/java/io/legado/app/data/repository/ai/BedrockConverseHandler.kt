package io.legado.app.data.repository.ai

import com.google.gson.JsonObject
import io.legado.app.domain.gateway.AiStreamEvent
import io.legado.app.domain.model.AiAvailableModel
import io.legado.app.domain.model.AiGenerateRequest
import io.legado.app.domain.model.AiGenerateResponse
import io.legado.app.domain.model.AiHttpException
import io.legado.app.domain.model.AiMessage
import io.legado.app.domain.model.AiMessageRole
import io.legado.app.domain.model.AiProtocol
import io.legado.app.domain.model.AiProviderConfig
import io.legado.app.domain.model.AiReasoningLevel
import io.legado.app.domain.model.AiToolDefinition
import io.legado.app.help.http.addHeaders
import io.legado.app.help.http.newCallResponse
import io.legado.app.help.http.newCallStrResponse
import io.legado.app.help.http.okHttpClient
import io.legado.app.help.http.postJson
import io.legado.app.utils.GSON
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.Response
import okio.Buffer
import okio.BufferedSource
import java.net.URLEncoder
import java.util.concurrent.ConcurrentHashMap

/**
 * AWS Bedrock Converse API, authenticated with a Bedrock API key (Bearer token).
 *
 * Bedrock serves inference and model listing from different hosts, so both URLs are configured
 * separately on the provider:
 * - baseUrl: runtime endpoint, e.g. `https://bedrock-runtime.us-east-1.amazonaws.com`
 * - modelsUrl: control-plane list endpoint, e.g.
 *   `https://bedrock.us-east-1.amazonaws.com/inference-profiles` or `.../foundation-models`
 */
class BedrockConverseHandler : AiProtocolHandler {

    override val protocols = setOf(AiProtocol.AWS_BEDROCK_CONVERSE)

    override suspend fun generate(request: AiGenerateRequest): Result<AiGenerateResponse> =
        withContext(Dispatchers.IO) {
            runCatching { generateInternal(request) }
        }

    override suspend fun stream(
        request: AiGenerateRequest,
        emitEvent: suspend (AiStreamEvent) -> Unit
    ) {
        streamInternal(request, emitEvent)
    }

    override suspend fun fetchModels(provider: AiProviderConfig): Result<List<AiAvailableModel>> =
        withContext(Dispatchers.IO) {
            runCatching { fetchModelsInternal(provider) }
        }

    private suspend fun generateInternal(request: AiGenerateRequest): AiGenerateResponse {
        val provider = request.model.provider
        requireConfigured(request)
        val keyRotator = KeyRotator(provider.apiKey)
        val url = request.modelUrl("converse")
        return request.withRequestShape { shape ->
            generateWithShape(request, url, GSON.toJson(request.toConverseBody(shape)), keyRotator)
        }
    }

    private suspend fun generateWithShape(
        request: AiGenerateRequest,
        url: String,
        body: String,
        keyRotator: KeyRotator,
    ): AiGenerateResponse {
        val provider = request.model.provider
        return retryWithBackoff(maxAttempts = 3, keyRotator = keyRotator) {
            val response = request.httpClient().newCallStrResponse {
                url(url)
                postJson(body)
                addHeaders(provider.bedrockHeaders(request.sessionId, keyRotator.currentKey))
            }
            if (!response.isSuccessful()) {
                throw AiHttpException(response.code(), response.httpErrorMessage())
            }
            val text = response.body?.toJsonObject()
                ?.getAsJsonObject("output")
                ?.getAsJsonObject("message")
                ?.get("content")
                ?.asJsonArrayOrNull()
                ?.mapNotNull { it.asJsonObjectOrNull()?.getString("text") }
                ?.joinToString("")
            if (text.isNullOrBlank()) {
                throw Exception("Empty AI response")
            } else {
                AiGenerateResponse(text = text, rawBody = response.body)
            }
        }
    }

    private suspend fun streamInternal(
        request: AiGenerateRequest,
        emitEvent: suspend (AiStreamEvent) -> Unit
    ) {
        val provider = request.model.provider
        requireConfigured(request)
        val keyRotator = KeyRotator(provider.apiKey)
        val url = request.modelUrl("converse-stream")
        val response = request.withRequestShape { shape ->
            val body = GSON.toJson(request.toConverseBody(shape))
            retryWithBackoff(maxAttempts = 3, keyRotator = keyRotator) {
                request.httpClient().newCallResponse {
                    url(url)
                    postJson(body)
                    addHeaders(
                        provider.bedrockHeaders(request.sessionId, keyRotator.currentKey) +
                            ("Accept" to "application/vnd.amazon.eventstream")
                    )
                }.also {
                    if (!it.isSuccessful) {
                        throw AiHttpException(it.code, it.httpErrorMessage())
                    }
                }
            }
        }
        val toolBlocks = mutableMapOf<Int, AnthropicToolBlock>()
        response.readEventStream { headers, payload ->
            val json = payload.toJsonObject() ?: JsonObject()
            when (headers[":message-type"]) {
                "exception" -> throw Exception(
                    "${headers[":exception-type"] ?: "Bedrock error"}: ${json.getString("message").orEmpty()}"
                )
                "error" -> throw Exception(
                    "${headers[":error-code"] ?: "Bedrock error"}: ${headers[":error-message"].orEmpty()}"
                )
            }
            val index = json.get("contentBlockIndex")?.takeIf { !it.isJsonNull }?.asInt
            when (headers[":event-type"]) {
                "contentBlockStart" -> {
                    val toolUse = json.getAsJsonObject("start")?.getAsJsonObject("toolUse")
                        ?: return@readEventStream
                    val id = toolUse.getString("toolUseId") ?: "bedrock_tool_$index"
                    val name = toolUse.getString("name").orEmpty()
                    index?.let { toolBlocks[it] = AnthropicToolBlock(id, name) }
                    emitEvent(AiStreamEvent.ToolCallDelta(id, index, name, null, "tool_use"))
                }
                "contentBlockDelta" -> {
                    val delta = json.getAsJsonObject("delta") ?: return@readEventStream
                    delta.getString("text")?.takeIf { it.isNotEmpty() }?.let {
                        emitEvent(AiStreamEvent.Content(it))
                    }
                    delta.getAsJsonObject("reasoningContent")?.getString("text")
                        ?.takeIf { it.isNotEmpty() }
                        ?.let { emitEvent(AiStreamEvent.Reasoning(it)) }
                    delta.getAsJsonObject("toolUse")?.getString("input")
                        ?.takeIf { it.isNotEmpty() }
                        ?.let {
                            val block = index?.let { i -> toolBlocks[i] }
                            emitEvent(
                                AiStreamEvent.ToolCallDelta(
                                    id = block?.id ?: index?.let { i -> "bedrock_tool_$i" },
                                    index = index,
                                    name = block?.name,
                                    argumentsDelta = it,
                                    rawType = "input_json_delta"
                                )
                            )
                        }
                }
                "contentBlockStop" -> index?.let { toolBlocks.remove(it) }
            }
        }
    }

    private suspend fun fetchModelsInternal(provider: AiProviderConfig): List<AiAvailableModel> {
        require(provider.apiKey.isNotBlank()) { "Bedrock configuration incomplete: API key is required" }
        val modelsUrl = provider.modelsUrl?.takeIf { it.isNotBlank() }
            ?: throw IllegalArgumentException(
                "Bedrock models URL is required, e.g. https://bedrock.us-east-1.amazonaws.com/inference-profiles"
            )
        val keyRotator = KeyRotator(provider.apiKey)
        val models = mutableListOf<AiAvailableModel>()
        var nextToken: String? = null
        for (page in 1..MAX_MODEL_PAGES) {
            val pageUrl = modelsUrl.toHttpUrl().newBuilder()
                .apply { nextToken?.let { setQueryParameter("nextToken", it) } }
                .build()
                .toString()
            val json = retryWithBackoff(maxAttempts = 2, keyRotator = keyRotator) {
                val response = okHttpClient.newCallStrResponse {
                    url(pageUrl)
                    addHeaders(provider.bedrockHeaders(null, keyRotator.currentKey))
                }
                if (!response.isSuccessful()) {
                    throw AiHttpException(response.code(), response.httpErrorMessage())
                }
                response.body?.toJsonObject()
            }
            models += json.toBedrockModels()
            nextToken = json?.getString("nextToken")?.takeIf { it.isNotBlank() } ?: break
        }
        return models.distinctBy { it.id }.sortedBy { it.name.lowercase() }
    }

    private fun requireConfigured(request: AiGenerateRequest) {
        val provider = request.model.provider
        require(provider.baseUrl.isNotBlank() && provider.apiKey.isNotBlank() && request.model.modelId.isNotBlank()) {
            "Bedrock configuration incomplete: baseUrl, apiKey, and model are required"
        }
    }

    private companion object {
        /** ListInferenceProfiles / ListFoundationModels pages; guards against a token that never ends. */
        const val MAX_MODEL_PAGES = 50
    }
}

private fun AiProviderConfig.bedrockHeaders(sessionId: String?, apiKey: String): Map<String, String> =
    aiRequestHeaders(
        sessionId = sessionId,
        protocolHeaders = mapOf(
            "Authorization" to "Bearer $apiKey",
            "Content-Type" to "application/json"
        )
    )

/** Model ids may be ARNs containing `/` and `:`, so the id is sent as one encoded path segment. */
private fun AiGenerateRequest.modelUrl(action: String): String {
    val modelId = URLEncoder.encode(model.modelId, "UTF-8").replace("+", "%20")
    return "${model.provider.baseUrl.trimEnd('/')}/model/$modelId/$action"
}

/**
 * Request shape a Bedrock model accepts. Claude generations disagree (checked on Bedrock
 * 2026-11-01): 5.x rejects `temperature` and `thinking.type: enabled`, 4.5 rejects adaptive
 * thinking and `output_config.effort`, 4.6 takes both. Rather than pin model versions, requests
 * start from the current shape and [adjustFor] flips the one setting a 400 names.
 */
internal data class BedrockRequestShape(
    val adaptiveThinking: Boolean = true,
    val sampling: Boolean = true,
) {
    /** The shape to retry with after a 400 [message], or null when the error is not about the shape. */
    fun adjustFor(message: String): BedrockRequestShape? = when {
        sampling && ("`temperature` is deprecated" in message || "`top_p` is deprecated" in message) ->
            copy(sampling = false)
        adaptiveThinking && ("adaptive thinking is not supported" in message ||
            "output_config.effort: Extra inputs are not permitted" in message) -> copy(adaptiveThinking = false)
        !adaptiveThinking && "\"thinking.type.enabled\" is not supported" in message -> copy(adaptiveThinking = true)
        else -> null
    }
}

/** Learned per model id so later chunks of the same chapter skip the failing first attempt. */
private val bedrockRequestShapes = ConcurrentHashMap<String, BedrockRequestShape>()

/** Runs [block] with the model's request shape, retrying up to 3 times while a 400 names a shape problem. */
private suspend fun <T> AiGenerateRequest.withRequestShape(block: suspend (BedrockRequestShape) -> T): T {
    var shape = bedrockRequestShapes[model.modelId] ?: BedrockRequestShape()
    repeat(3) {
        try {
            return block(shape)
        } catch (e: AiHttpException) {
            val next = shape.takeIf { e.statusCode == 400 }?.adjustFor(e.message.orEmpty()) ?: throw e
            shape = next
            bedrockRequestShapes[model.modelId] = next
        }
    }
    return block(shape)
}

internal fun AiGenerateRequest.toConverseBody(shape: BedrockRequestShape = BedrockRequestShape()): Map<String, Any?> {
    val body = mutableMapOf<String, Any?>("messages" to messages.toBedrockMessages())
    messages.filter { it.role == AiMessageRole.SYSTEM && it.content.isNotBlank() }
        .takeIf { it.isNotEmpty() }
        ?.let { system -> body["system"] = system.map { mapOf("text" to it.content) } }
    val inferenceConfig = mutableMapOf<String, Any>()
    (params.maxOutputTokens ?: model.maxOutputTokens.takeIf { it > 0 })?.let { inferenceConfig["maxTokens"] = it }
    if (shape.sampling) {
        params.temperature?.let { inferenceConfig["temperature"] = it }
        params.topP?.let { inferenceConfig["topP"] = it }
    }
    bedrockReasoningFields(inferenceConfig, shape)?.let { body["additionalModelRequestFields"] = it }
    if (inferenceConfig.isNotEmpty()) body["inferenceConfig"] = inferenceConfig
    tools.takeIf { it.isNotEmpty() }?.let { body["toolConfig"] = mapOf("tools" to it.toBedrockTools()) }
    return body
}

/**
 * Model-family reasoning switches for Converse `additionalModelRequestFields`, and the
 * [inferenceConfig] adjustments each family requires while reasoning:
 * - Anthropic Claude: adaptive `thinking` + `output_config.effort`, or on older models `thinking`
 *   with `budget_tokens` (min 1024, below maxTokens); no temperature/topP either way
 * - Amazon Nova: `reasoningConfig` with `maxReasoningEffort` (high also forbids maxTokens)
 * - OpenAI gpt-oss: `reasoning_effort`
 * AUTO leaves the model default; models without the reasoning capability get nothing.
 */
internal fun AiGenerateRequest.bedrockReasoningFields(
    inferenceConfig: MutableMap<String, Any>,
    shape: BedrockRequestShape = BedrockRequestShape(),
): Map<String, Any>? {
    if (!hasReasoningCapability(model.capabilities)) return null
    val level = params.reasoningLevel
    if (level == AiReasoningLevel.AUTO) return null
    val id = model.modelId.lowercase()
    val effort = when (level) {
        AiReasoningLevel.LOW -> "low"
        AiReasoningLevel.MEDIUM -> "medium"
        else -> "high"
    }
    return when {
        "anthropic.claude" in id -> {
            // Claude rejects a tool-use turn replayed without its signed thinking block, which
            // AiMessage does not keep, so no thinking once the conversation has tool calls. OFF
            // also sends nothing: newer models reject `thinking.type: disabled`.
            if (level == AiReasoningLevel.OFF || messages.any { it.toolCalls.isNotEmpty() }) return null
            inferenceConfig.remove("temperature")
            inferenceConfig.remove("topP")
            if (shape.adaptiveThinking) {
                // Accepted on 4.6 and 5.x: low / medium / high / max (4.6 rejects xhigh).
                val claudeEffort = if (level == AiReasoningLevel.MAX) "max" else effort
                return mapOf("thinking" to mapOf("type" to "adaptive"), "output_config" to mapOf("effort" to claudeEffort))
            }
            val budget = level.budgetTokens.coerceAtLeast(1024)
            val maxTokens = (inferenceConfig["maxTokens"] as? Int) ?: 0
            if (maxTokens <= budget) inferenceConfig["maxTokens"] = budget + 4096
            mapOf("thinking" to mapOf("type" to "enabled", "budget_tokens" to budget))
        }
        "amazon.nova" in id -> {
            if (level == AiReasoningLevel.OFF) {
                return mapOf("reasoningConfig" to mapOf("type" to "disabled"))
            }
            inferenceConfig.remove("temperature")
            inferenceConfig.remove("topP")
            if (effort == "high") inferenceConfig.remove("maxTokens")
            mapOf("reasoningConfig" to mapOf("type" to "enabled", "maxReasoningEffort" to effort))
        }
        "openai.gpt-oss" in id -> {
            if (level == AiReasoningLevel.OFF) null else mapOf("reasoning_effort" to effort)
        }
        else -> null
    }
}

/** Converse requires alternating roles, so consecutive same-role messages (e.g. tool results) merge. */
internal fun List<AiMessage>.toBedrockMessages(): List<Map<String, Any?>> {
    val result = mutableListOf<Pair<String, MutableList<Map<String, Any?>>>>()
    filter { it.role != AiMessageRole.SYSTEM }.forEach { message ->
        val role = if (message.role == AiMessageRole.ASSISTANT) "assistant" else "user"
        val blocks = message.toBedrockContent().takeIf { it.isNotEmpty() } ?: return@forEach
        val last = result.lastOrNull()
        if (last?.first == role) last.second += blocks else result += role to blocks.toMutableList()
    }
    return result.map { (role, content) -> mapOf("role" to role, "content" to content) }
}

private fun AiMessage.toBedrockContent(): List<Map<String, Any?>> = buildList {
    if (role == AiMessageRole.TOOL) {
        add(
            mapOf(
                "toolResult" to mapOf(
                    "toolUseId" to toolCallId,
                    "content" to listOf(mapOf("text" to content.ifBlank { " " }))
                )
            )
        )
        return@buildList
    }
    // Bedrock rejects empty text blocks.
    if (content.isNotBlank()) add(mapOf("text" to content))
    images.forEach { image ->
        add(
            mapOf(
                "image" to mapOf(
                    "format" to image.mimeType.substringAfter('/'),
                    "source" to mapOf("bytes" to image.base64Data)
                )
            )
        )
    }
    toolCalls.forEach {
        add(
            mapOf(
                "toolUse" to mapOf(
                    "toolUseId" to it.id,
                    "name" to it.name,
                    "input" to (it.arguments.toJsonObject() ?: JsonObject())
                )
            )
        )
    }
}

private fun List<AiToolDefinition>.toBedrockTools(): List<Map<String, Any?>> = map {
    mapOf(
        "toolSpec" to mapOf(
            "name" to it.name,
            "description" to it.description,
            "inputSchema" to mapOf("json" to it.inputSchema)
        )
    )
}

/** Accepts both ListFoundationModels and ListInferenceProfiles responses. */
internal fun JsonObject?.toBedrockModels(): List<AiAvailableModel> {
    this ?: return emptyList()
    val foundationModels = get("modelSummaries")?.asJsonArrayOrNull()?.toList().orEmpty()
        .mapNotNull { it.asJsonObjectOrNull() }
        .filter { summary ->
            val outputs = summary.get("outputModalities")?.asJsonArrayOrNull()
                ?: return@filter true
            outputs.any { !it.isJsonNull && it.asString == "TEXT" }
        }
        .mapNotNull { summary ->
            val id = summary.getString("modelId")?.takeIf { it.isNotBlank() } ?: return@mapNotNull null
            val name = summary.getString("modelName")?.takeIf { it.isNotBlank() } ?: id
            val providerName = summary.getString("providerName")?.takeIf { it.isNotBlank() }
            AiAvailableModel(id = id, name = providerName?.let { "$it $name" } ?: name)
        }
    val inferenceProfiles = get("inferenceProfileSummaries")?.asJsonArrayOrNull()?.toList().orEmpty()
        .mapNotNull { it.asJsonObjectOrNull() }
        .mapNotNull { summary ->
            val id = summary.getString("inferenceProfileId")?.takeIf { it.isNotBlank() }
                ?: return@mapNotNull null
            AiAvailableModel(
                id = id,
                name = summary.getString("inferenceProfileName")?.takeIf { it.isNotBlank() } ?: id
            )
        }
    return (foundationModels + inferenceProfiles)
        .distinctBy { it.id }
        .sortedBy { it.name.lowercase() }
}

internal sealed interface EventStreamItem {
    data class Message(val headers: Map<String, String>, val payload: String) : EventStreamItem
    data class Failure(val error: Throwable) : EventStreamItem
    data object End : EventStreamItem
}

/**
 * Reads an `application/vnd.amazon.eventstream` body frame by frame. Same threading model as
 * [readSseData]: the blocking read runs in a child coroutine and the response is closed on exit,
 * so cancelling the caller also ends a socket read that is still waiting.
 */
private suspend fun Response.readEventStream(onMessage: suspend (Map<String, String>, String) -> Unit) {
    val response = this
    val items = Channel<EventStreamItem>(Channel.UNLIMITED)
    coroutineScope {
        launch {
            try {
                pumpEventStream(body.source(), items)
                items.trySend(EventStreamItem.End)
            } catch (e: Throwable) {
                items.trySend(EventStreamItem.Failure(e))
            }
        }
        try {
            while (true) {
                when (val item = items.receive()) {
                    is EventStreamItem.Message -> onMessage(item.headers, item.payload)
                    is EventStreamItem.Failure -> throw item.error
                    EventStreamItem.End -> break
                }
            }
        } finally {
            items.close()
            runCatching { response.close() }
        }
    }
}

/** Frame: total length, headers length, prelude CRC, headers, payload, message CRC (TLS covers integrity). */
internal fun pumpEventStream(source: BufferedSource, items: Channel<EventStreamItem>) {
    source.use {
        while (!source.exhausted()) {
            val totalLength = source.readInt().toLong()
            val headersLength = source.readInt().toLong()
            source.readInt()
            val headers = parseEventStreamHeaders(Buffer().write(source.readByteArray(headersLength)))
            val payload = source.readUtf8(totalLength - headersLength - 16)
            source.readInt()
            items.trySend(EventStreamItem.Message(headers, payload))
        }
    }
}

/** Keeps string-typed headers (`:event-type`, `:message-type`, …) and skips other value types. */
private fun parseEventStreamHeaders(buffer: Buffer): Map<String, String> {
    val headers = mutableMapOf<String, String>()
    while (!buffer.exhausted()) {
        val name = buffer.readUtf8((buffer.readByte().toInt() and 0xff).toLong())
        when (val type = buffer.readByte().toInt()) {
            0, 1 -> Unit
            2 -> buffer.skip(1)
            3 -> buffer.skip(2)
            4 -> buffer.skip(4)
            5, 8 -> buffer.skip(8)
            6 -> buffer.skip((buffer.readShort().toInt() and 0xffff).toLong())
            7 -> headers[name] = buffer.readUtf8((buffer.readShort().toInt() and 0xffff).toLong())
            9 -> buffer.skip(16)
            else -> throw IllegalStateException("Unknown event stream header type $type")
        }
    }
    return headers
}
