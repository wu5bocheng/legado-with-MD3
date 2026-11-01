package io.legado.app.data.repository.ai

import okhttp3.HttpUrl
import java.security.MessageDigest
import java.time.ZoneOffset
import java.time.ZonedDateTime
import java.time.format.DateTimeFormatter
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec

/** IAM credentials entered in the API key field as `AccessKeyId:SecretAccessKey[:SessionToken]`. */
internal data class AwsCredentials(
    val accessKeyId: String,
    val secretAccessKey: String,
    val sessionToken: String? = null,
) {
    companion object {
        /**
         * Bedrock API keys are base64 and never contain `:`, so a colon marks IAM credentials.
         * Returns null for a Bedrock API key (sent as a Bearer token instead).
         */
        fun parse(key: String): AwsCredentials? {
            if (':' !in key) return null
            val parts = key.split(':').map { it.trim() }
            require(parts.size in 2..3 && parts[0].isNotEmpty() && parts[1].isNotEmpty()) {
                "AWS credentials must be AccessKeyId:SecretAccessKey or AccessKeyId:SecretAccessKey:SessionToken"
            }
            return AwsCredentials(parts[0], parts[1], parts.getOrNull(2)?.takeIf { it.isNotEmpty() })
        }
    }
}

/**
 * AWS Signature Version 4 for a single request.
 * Docs: https://docs.aws.amazon.com/IAM/latest/UserGuide/create-signed-request.html
 */
internal object AwsSigV4 {

    private val amzDateFormat = DateTimeFormatter.ofPattern("yyyyMMdd'T'HHmmss'Z'")
    private val regionInHost = Regex("""(?:^|\.)([a-z]{2}(?:-gov|-iso[a-z]?)?-[a-z]+-\d+)\.""")

    /** Region taken from a standard or VPC endpoint host, e.g. `bedrock-runtime.us-east-1.amazonaws.com`. */
    fun regionOf(url: HttpUrl): String =
        regionInHost.find(url.host)?.groupValues?.get(1)
            ?: throw IllegalArgumentException(
                "Cannot find the AWS region in ${url.host}; use an endpoint such as bedrock-runtime.us-east-1.amazonaws.com"
            )

    /**
     * Returns the headers to add to the request: `X-Amz-Date`, `X-Amz-Security-Token` when the
     * credentials are temporary, and `Authorization`. [extraSignedHeaders] are signed as well and
     * must be sent unchanged.
     */
    fun sign(
        method: String,
        url: HttpUrl,
        payload: ByteArray,
        credentials: AwsCredentials,
        region: String,
        service: String,
        extraSignedHeaders: Map<String, String> = emptyMap(),
        now: ZonedDateTime = ZonedDateTime.now(ZoneOffset.UTC),
    ): Map<String, String> {
        val amzDate = now.withZoneSameInstant(ZoneOffset.UTC).format(amzDateFormat)
        val date = amzDate.substring(0, 8)
        val added = buildMap {
            put("X-Amz-Date", amzDate)
            credentials.sessionToken?.let { put("X-Amz-Security-Token", it) }
        }
        val host = if (url.port == HttpUrl.defaultPort(url.scheme)) url.host else "${url.host}:${url.port}"
        val signed = (extraSignedHeaders + added + ("Host" to host))
            .mapKeys { it.key.lowercase() }
            .mapValues { it.value.trim().replace(Regex("\\s+"), " ") }
            .toSortedMap()
        val signedHeaderNames = signed.keys.joinToString(";")
        val canonicalRequest = listOf(
            method.uppercase(),
            canonicalUri(url),
            canonicalQuery(url),
            signed.entries.joinToString("") { "${it.key}:${it.value}\n" },
            signedHeaderNames,
            sha256Hex(payload),
        ).joinToString("\n")
        val scope = "$date/$region/$service/aws4_request"
        val stringToSign = listOf(
            "AWS4-HMAC-SHA256",
            amzDate,
            scope,
            sha256Hex(canonicalRequest.toByteArray()),
        ).joinToString("\n")
        val signingKey = listOf(date, region, service, "aws4_request")
            .fold("AWS4${credentials.secretAccessKey}".toByteArray()) { key, part -> hmac(key, part) }
        val signature = hmac(signingKey, stringToSign).toHex()
        return added + (
            "Authorization" to "AWS4-HMAC-SHA256 Credential=${credentials.accessKeyId}/$scope, " +
                "SignedHeaders=$signedHeaderNames, Signature=$signature"
            )
    }

    /** Non-S3 services encode each path segment again, so a sent `%3A` is signed as `%253A`. */
    private fun canonicalUri(url: HttpUrl): String =
        "/" + url.encodedPathSegments.joinToString("/") { uriEncode(it) }

    private fun canonicalQuery(url: HttpUrl): String =
        (0 until url.querySize)
            .map { uriEncode(url.queryParameterName(it)) to uriEncode(url.queryParameterValue(it).orEmpty()) }
            .sortedWith(compareBy({ it.first }, { it.second }))
            .joinToString("&") { "${it.first}=${it.second}" }

    private fun uriEncode(value: String): String = buildString {
        value.toByteArray().forEach { byte ->
            val c = byte.toInt().toChar()
            if (c in 'A'..'Z' || c in 'a'..'z' || c in '0'..'9' || c in "-_.~") {
                append(c)
            } else {
                append('%').append("%02X".format(byte.toInt() and 0xff))
            }
        }
    }

    private fun hmac(key: ByteArray, data: String): ByteArray =
        Mac.getInstance("HmacSHA256").run {
            init(SecretKeySpec(key, "HmacSHA256"))
            doFinal(data.toByteArray())
        }

    private fun sha256Hex(data: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(data).toHex()

    private fun ByteArray.toHex(): String = joinToString("") { "%02x".format(it.toInt() and 0xff) }
}
