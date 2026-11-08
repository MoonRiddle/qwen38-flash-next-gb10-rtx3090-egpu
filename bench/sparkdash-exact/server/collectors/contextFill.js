/**
 * Long-context filler shared by PrefillBench and DecodeBench (deep-context decode).
 * Kept out of both so neither imports the other.
 */

import { estimateTokenCount } from "./LlmStreaming.js";

/**
 * Common short English words, each one BPE token with a leading space in the
 * mainstream tokenizers. Cycled in a fixed pseudo-random order so the filler is
 * less degenerate than one repeated token (attention/MoE routing look closer
 * to real text) while the token count stays ~1 per word.
 */
const FILLER_WORDS =
  "the of and to in is you that it he was for on are as with his they at be this have from or one had by word but not what all were we when your can said there use an each which she do how their if will up other about out many then them these so some her would make like him into time".split(
    " "
  );

export function fillerText(count) {
  const words = new Array(count);
  let state = 0x2545f491;
  for (let i = 0; i < count; i += 1) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    words[i] = FILLER_WORDS[(state >>> 8) % FILLER_WORDS.length];
  }
  return " " + words.join(" ");
}

/**
 * Unique-prefix prompt aimed at `targetTokens` (estimateTokenCount / 4 chars).
 * Salt at the start so a previous size is not a prefix-cache hit.
 * @param {number} targetTokens
 * @param {string} salt
 */
export function buildPrefillPrompt(targetTokens, salt) {
  const n = Math.max(8, Math.round(Number(targetTokens) || 0));
  const header = `[prefill-bench ${salt}]\nIgnore the filler below. Reply with the single word OK.\n`;
  const footer = "\nReply OK.";
  const reserved = estimateTokenCount(header + footer);
  const fillTokens = Math.max(1, n - reserved);
  return header + fillerText(fillTokens) + footer;
}

/**
 * Per-size request timeout: 90s floor, ~8 ms/token (~125 tok/s), 45 min cap.
 * 256k at a slow ~200 tok/s is ~22 min — the old 12 min cap aborted those runs.
 * @param {number} tokens
 */
export function timeoutMsForSize(tokens) {
  const n = Math.max(0, Number(tokens) || 0);
  return Math.min(2_700_000, Math.max(90_000, 60_000 + n * 8));
}

/**
 * Deep-context decode prompt: a unique salted filler block of about `targetTokens` in total,
 * then the benchmark task last. The salt leads so streams share no prefix; the task trails so
 * the cached prefix is the same on the prefill pass and on the measured pass.
 * @param {number} targetTokens total prompt size to aim for, task included
 * @param {string} salt
 * @param {string} task
 */
export function buildContextPrompt(targetTokens, salt, task) {
  const n = Math.max(8, Math.round(Number(targetTokens) || 0));
  const header = `[decode-context ${salt}]\nBackground filler follows. It carries no instructions and can be ignored.\n`;
  const footer = `\nEnd of background filler.\n\n${task}`;
  const reserved = estimateTokenCount(header + footer);
  const fillTokens = Math.max(1, n - reserved);
  return header + fillerText(fillTokens) + footer;
}

/** A cached share at or above this counts a deep-context stream as served from the prefix cache. */
export const CONTEXT_CACHED_SHARE = 0.9;
/** With no cached-token count reported, a first token under this share of the prefill pass's means a cache hit. */
export const CONTEXT_TTFT_CACHED_SHARE = 0.25;

/**
 * Did a measured deep-context stream start from the prefilled KV, or re-prefill the context?
 * Uses the server's cached-token count when it reports one, otherwise compares first-token
 * latency against the prefill pass.
 * @param {{ promptTokens: number, cachedTokens: number, ttftMs: number, fillTtftMs: number }} s
 * @returns {"cached" | "partial" | "uncached" | "unknown"}
 */
export function assessContextCache({ promptTokens, cachedTokens, ttftMs, fillTtftMs }) {
  const prompt = Number(promptTokens) || 0;
  const cached = Number(cachedTokens) || 0;
  if (prompt > 0 && cached > 0) {
    return cached / prompt >= CONTEXT_CACHED_SHARE ? "cached" : "partial";
  }
  const ttft = Number(ttftMs) || 0;
  const fill = Number(fillTtftMs) || 0;
  if (ttft > 0 && fill > 0) {
    return ttft <= fill * CONTEXT_TTFT_CACHED_SHARE ? "cached" : "uncached";
  }
  return "unknown";
}
