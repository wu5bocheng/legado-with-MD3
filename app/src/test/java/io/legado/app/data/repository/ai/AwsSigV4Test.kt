package io.legado.app.data.repository.ai

import okhttp3.HttpUrl.Companion.toHttpUrl
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import java.time.ZoneOffset
import java.time.ZonedDateTime

class AwsSigV4Test {

    private val exampleSecret = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"

    /** Worked example from the AWS SigV4 documentation (IAM ListUsers). */
    @Test
    fun matchesAwsDocumentationExample() {
        val headers = AwsSigV4.sign(
            method = "GET",
            url = "https://iam.amazonaws.com/?Action=ListUsers&Version=2010-05-08".toHttpUrl(),
            payload = ByteArray(0),
            credentials = AwsCredentials("AKIDEXAMPLE", exampleSecret),
            region = "us-east-1",
            service = "iam",
            extraSignedHeaders = mapOf("Content-Type" to "application/x-www-form-urlencoded; charset=utf-8"),
            now = ZonedDateTime.of(2015, 8, 30, 12, 36, 0, 0, ZoneOffset.UTC),
        )

        assertEquals("20150830T123600Z", headers["X-Amz-Date"])
        assertEquals(
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/iam/aws4_request, " +
                "SignedHeaders=content-type;host;x-amz-date, " +
                "Signature=5d672d79c15b13162d9279b0855cfba6789a8edb4c82c400e06b5924a6f2b5d7",
            headers["Authorization"]
        )
    }

    /**
     * Reference signatures from botocore's SigV4Auth for the same requests and credentials. The
     * model id's encoded `:` must be encoded again in the canonical path, and query values sorted
     * and encoded, or Bedrock answers 403 SignatureDoesNotMatch.
     */
    @Test
    fun matchesBotocoreForBedrockConverseAndModelList() {
        val credentials = AwsCredentials("AKIDEXAMPLE", exampleSecret, "SESSIONTOKEN")
        val now = ZonedDateTime.of(2026, 11, 1, 7, 30, 25, 0, ZoneOffset.UTC)

        val converse = AwsSigV4.sign(
            method = "POST",
            url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-sonnet-4-20250514-v1%3A0/converse".toHttpUrl(),
            payload = """{"messages":[]}""".toByteArray(),
            credentials = credentials,
            region = "us-east-1",
            service = "bedrock",
            now = now,
        )
        assertEquals("SESSIONTOKEN", converse["X-Amz-Security-Token"])
        assertEquals(
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20261101/us-east-1/bedrock/aws4_request, " +
                "SignedHeaders=host;x-amz-date;x-amz-security-token, " +
                "Signature=2d33e7114d74b822432ae63ee6f08e99e1242f7eb4b6cd896be034cdbaae6e99",
            converse["Authorization"]
        )

        val models = AwsSigV4.sign(
            method = "GET",
            url = "https://bedrock.us-east-1.amazonaws.com/inference-profiles?maxResults=1000&nextToken=a%2Fb%3D".toHttpUrl(),
            payload = ByteArray(0),
            credentials = credentials,
            region = "us-east-1",
            service = "bedrock",
            now = now,
        )
        assertEquals(
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20261101/us-east-1/bedrock/aws4_request, " +
                "SignedHeaders=host;x-amz-date;x-amz-security-token, " +
                "Signature=da85129d456c06e72b16163a6d36c163a8f45d940567a6863649246611276244",
            models["Authorization"]
        )
    }

    @Test
    fun readsRegionFromStandardAndVpcEndpoints() {
        assertEquals("us-east-1", AwsSigV4.regionOf("https://bedrock-runtime.us-east-1.amazonaws.com".toHttpUrl()))
        assertEquals("eu-central-1", AwsSigV4.regionOf("https://bedrock.eu-central-1.amazonaws.com/foundation-models".toHttpUrl()))
        assertEquals("us-gov-west-1", AwsSigV4.regionOf("https://bedrock-runtime-fips.us-gov-west-1.amazonaws.com".toHttpUrl()))
        assertEquals(
            "ap-northeast-1",
            AwsSigV4.regionOf("https://vpce-0abc.bedrock-runtime.ap-northeast-1.vpce.amazonaws.com".toHttpUrl())
        )
    }

    @Test
    fun parsesIamCredentialsButNotBedrockApiKeys() {
        assertNull(AwsCredentials.parse("ABSKQmVkcm9ja0FQSUtleS1leGFtcGxlPQ=="))
        assertEquals(AwsCredentials("AKID", "secret"), AwsCredentials.parse("AKID:secret"))
        assertEquals(AwsCredentials("AKID", "secret", "token"), AwsCredentials.parse(" AKID : secret : token "))
    }
}
