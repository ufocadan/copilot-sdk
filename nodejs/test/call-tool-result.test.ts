import { describe, expect, it } from "vitest";
import { isCallToolResult, convertCallToolResult } from "../src/types.js";
import type { CallToolResult, ToolResultObject } from "../src/types.js";

describe("isCallToolResult", () => {
    it("returns true for a text-only CallToolResult", () => {
        expect(
            isCallToolResult({
                content: [{ type: "text", text: "hello" }],
            })
        ).toBe(true);
    });

    it("returns true for CallToolResult with isError", () => {
        expect(
            isCallToolResult({
                content: [{ type: "text", text: "oops" }],
                isError: true,
            })
        ).toBe(true);
    });

    it("returns true for CallToolResult with image content", () => {
        expect(
            isCallToolResult({
                content: [{ type: "image", data: "abc123", mimeType: "image/png" }],
            })
        ).toBe(true);
    });

    it("returns true for CallToolResult with resource content", () => {
        expect(
            isCallToolResult({
                content: [
                    { type: "resource", resource: { uri: "file:///tmp/out.txt", text: "data" } },
                ],
            })
        ).toBe(true);
    });

    it("returns true for empty content array", () => {
        expect(isCallToolResult({ content: [] })).toBe(true);
    });

    it("returns false for null", () => {
        expect(isCallToolResult(null)).toBe(false);
    });

    it("returns false for a string", () => {
        expect(isCallToolResult("hello")).toBe(false);
    });

    it("returns false for a ToolResultObject", () => {
        expect(
            isCallToolResult({ textResultForLlm: "hi", resultType: "success" })
        ).toBe(false);
    });

    it("returns false when content is not an array", () => {
        expect(isCallToolResult({ content: "text" })).toBe(false);
    });

    it("returns false when content items lack type field", () => {
        expect(isCallToolResult({ content: [{ text: "no type" }] })).toBe(false);
    });
});

describe("convertCallToolResult", () => {
    it("extracts text from text content blocks", () => {
        const input: CallToolResult = {
            content: [
                { type: "text", text: "line 1" },
                { type: "text", text: "line 2" },
            ],
        };

        const result = convertCallToolResult(input);

        expect(result.textResultForLlm).toBe("line 1\nline 2");
        expect(result.resultType).toBe("success");
        expect(result.binaryResultsForLlm).toBeUndefined();
    });

    it("maps isError to failure resultType", () => {
        const input: CallToolResult = {
            content: [{ type: "text", text: "error occurred" }],
            isError: true,
        };

        const result = convertCallToolResult(input);

        expect(result.textResultForLlm).toBe("error occurred");
        expect(result.resultType).toBe("failure");
    });

    it("maps isError: false to success", () => {
        const input: CallToolResult = {
            content: [{ type: "text", text: "ok" }],
            isError: false,
        };

        expect(convertCallToolResult(input).resultType).toBe("success");
    });

    it("converts image content to binaryResultsForLlm", () => {
        const input: CallToolResult = {
            content: [{ type: "image", data: "base64data", mimeType: "image/png" }],
        };

        const result = convertCallToolResult(input);

        expect(result.textResultForLlm).toBe("");
        expect(result.binaryResultsForLlm).toHaveLength(1);
        expect(result.binaryResultsForLlm![0]).toEqual({
            data: "base64data",
            mimeType: "image/png",
            type: "image",
        });
    });

    it("converts resource with text to textResultForLlm", () => {
        const input: CallToolResult = {
            content: [
                {
                    type: "resource",
                    resource: { uri: "file:///tmp/data.txt", text: "file contents" },
                },
            ],
        };

        const result = convertCallToolResult(input);

        expect(result.textResultForLlm).toBe("file contents");
    });

    it("converts resource with blob to binaryResultsForLlm", () => {
        const input: CallToolResult = {
            content: [
                {
                    type: "resource",
                    resource: {
                        uri: "file:///tmp/image.png",
                        mimeType: "image/png",
                        blob: "blobdata",
                    },
                },
            ],
        };

        const result = convertCallToolResult(input);

        expect(result.binaryResultsForLlm).toHaveLength(1);
        expect(result.binaryResultsForLlm![0]).toEqual({
            data: "blobdata",
            mimeType: "image/png",
            type: "resource",
            description: "file:///tmp/image.png",
        });
    });

    it("handles mixed content types", () => {
        const input: CallToolResult = {
            content: [
                { type: "text", text: "Analysis complete" },
                { type: "image", data: "chartdata", mimeType: "image/svg+xml" },
                {
                    type: "resource",
                    resource: { uri: "file:///report.txt", text: "Report details" },
                },
            ],
        };

        const result = convertCallToolResult(input);

        expect(result.textResultForLlm).toBe("Analysis complete\nReport details");
        expect(result.binaryResultsForLlm).toHaveLength(1);
        expect(result.binaryResultsForLlm![0]!.mimeType).toBe("image/svg+xml");
    });

    it("handles empty content array", () => {
        const result = convertCallToolResult({ content: [] });

        expect(result.textResultForLlm).toBe("");
        expect(result.resultType).toBe("success");
        expect(result.binaryResultsForLlm).toBeUndefined();
    });

    it("defaults resource blob mimeType to application/octet-stream", () => {
        const input: CallToolResult = {
            content: [
                {
                    type: "resource",
                    resource: { uri: "file:///data.bin", blob: "binarydata" },
                },
            ],
        };

        const result = convertCallToolResult(input);

        expect(result.binaryResultsForLlm![0]!.mimeType).toBe("application/octet-stream");
    });

    it("handles text block with missing text field without corrupting output", () => {
        // isCallToolResult only checks that type is a string, not that type-specific
        // fields are present. convertCallToolResult must be defensive at runtime.
        const input = { content: [{ type: "text" }] } as unknown as CallToolResult;

        const result = convertCallToolResult(input);

        expect(result.textResultForLlm).toBe("");
        expect(result.textResultForLlm).not.toBe("undefined");
    });

    it("handles resource block with missing resource field without crashing", () => {
        // A resource content item missing the resource field would crash with an
        // unguarded block.resource.text access. Optional chaining must be used.
        const input = { content: [{ type: "resource" }] } as unknown as CallToolResult;

        expect(() => convertCallToolResult(input)).not.toThrow();
        const result = convertCallToolResult(input);
        expect(result.textResultForLlm).toBe("");
    });
});
