#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { createHash, randomUUID } from "node:crypto";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const SOURCE_COMMIT = "7c2b9638b025cd39fb43f95c225de62980f130ee";
const SCHEMA = "sparkdash-exact-invocation-v1";
const PROTOCOL_ID = "mia-sparkdash-native-7c2b963-c1-v1";
const TERMINAL = new Set(["completed", "failed", "cancelled"]);
let nativeModulesPromise;
let invocationActive = false;

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function errorRecord(error, stage) {
  return {
    stage,
    name: error?.name || "Error",
    message: error?.message || String(error),
    ...(error?.code != null ? { code: String(error.code) } : {}),
    ...(error?.stack ? { stack: error.stack } : {}),
  };
}

function requiredString(value, name) {
  if (typeof value !== "string" || !value.trim()) {
    throw new TypeError(`${name} must be a non-empty string`);
  }
  return value.trim();
}

function requireJsonObject(value, name) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError(`${name} must be a JSON object`);
  }
  // Reject values which JSON would silently drop or change in parent metadata.
  JSON.stringify(value, (_key, item) => {
    if (typeof item === "number" && !Number.isFinite(item)) {
      throw new TypeError(`${name} contains a non-finite number`);
    }
    if (["undefined", "function", "symbol", "bigint"].includes(typeof item)) {
      throw new TypeError(`${name} must contain only JSON values`);
    }
    return item;
  });
  return value;
}

function installedUndiciVersion() {
  try {
    const packageUrl = import.meta.resolve("undici/package.json");
    return JSON.parse(fs.readFileSync(fileURLToPath(packageUrl), "utf8")).version;
  } catch {
    return null;
  }
}

function collectSourceEvidence() {
  const lockBytes = fs.readFileSync(path.join(ROOT, "source-lock.json"));
  const lock = JSON.parse(lockBytes.toString("utf8"));
  const hashes = {};
  const mismatches = [];
  for (const entry of lock.files) {
    try {
      const bytes = fs.readFileSync(path.join(ROOT, entry.path));
      const actual = sha256(bytes);
      hashes[entry.path] = actual;
      if (actual !== entry.sha256 || bytes.byteLength !== entry.bytes) {
        mismatches.push({ path: entry.path, expected_sha256: entry.sha256, actual_sha256: actual });
      }
    } catch (error) {
      hashes[entry.path] = null;
      mismatches.push({ path: entry.path, error: error.message });
    }
  }
  if (lock.commit !== SOURCE_COMMIT) {
    mismatches.push({ path: "source-lock.json", expected_commit: SOURCE_COMMIT, actual_commit: lock.commit });
  }
  return {
    repo: lock.repo,
    commit: lock.commit,
    hashes,
    verified: mismatches.length === 0,
    verified_at: new Date().toISOString(),
    source_lock_sha256: sha256(lockBytes),
    mismatches,
  };
}

async function loadNativeModules(singletonDir) {
  if (!nativeModulesPromise) {
    // The upstream files construct singleton managers at import time. Isolate
    // those constructors before importing; per-invocation managers below also
    // receive explicit native history/active paths through their public API.
    fs.mkdirSync(singletonDir, { recursive: true });
    const paths = {
      BENCH_HISTORY_PATH: path.join(singletonDir, "bench-history.json"),
      BENCH_ACTIVE_PATH: path.join(singletonDir, "bench-active.json"),
      PREFILL_BENCH_HISTORY_PATH: path.join(singletonDir, "prefill-bench-history.json"),
      PREFILL_BENCH_ACTIVE_PATH: path.join(singletonDir, "prefill-bench-active.json"),
    };
    nativeModulesPromise = (async () => {
      const previous = {};
      for (const [key, value] of Object.entries(paths)) {
        previous[key] = process.env[key];
        process.env[key] = value;
      }
      try {
        const decode = await import("./server/collectors/DecodeBench.js");
        const prefill = await import("./server/collectors/PrefillBench.js");
        const target = await import("./src/shared/llmTarget.js");
        const streaming = await import("./server/collectors/LlmStreaming.js");
        return { decode, prefill, target, streaming };
      } finally {
        for (const [key, value] of Object.entries(previous)) {
          if (value === undefined) delete process.env[key];
          else process.env[key] = value;
        }
      }
    })();
  }
  return nativeModulesPromise;
}

// Mirrors the upstream constants in server/collectors/DecodeBench.js and
// src/shared/prefillBench.js. They are duplicated here because nativeOptions()
// runs before the pinned native modules are imported.
const ALLOWED_CONCURRENCIES = new Set([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32]);
const MIN_MAX_TOKENS = 64;
const MAX_MAX_TOKENS = 2048;
const PREFILL_MIN_CONTEXT_SIZE = 256;
const PREFILL_MAX_CONTEXT_SIZE = 300000;

function integerList(value, name, predicate, expectation) {
  if (!Array.isArray(value) || value.length === 0) {
    throw new TypeError(`${name} must be a non-empty array of ${expectation}`);
  }
  for (const entry of value) {
    if (!Number.isInteger(entry) || !predicate(entry)) {
      throw new TypeError(`${name} must be a non-empty array of ${expectation}`);
    }
  }
  return [...value];
}

function nativeOptions(caseName, opts, sparkId, host, port, modelId) {
  const common = {
    sparkId,
    lanIp: host,
    port,
    modelId,
    // Both collectors derive TLS from config.tls, which is recorded only with
    // their native host option. The native lanIp still selects the actual host.
    host,
    tls: opts.tls === true,
  };
  if (caseName === "prefill") {
    const sizes = opts.contextSizes == null
      ? [4096, 8192, 16384]
      : integerList(
        opts.contextSizes,
        "contextSizes",
        (n) => n >= PREFILL_MIN_CONTEXT_SIZE && n <= PREFILL_MAX_CONTEXT_SIZE,
        `integers between ${PREFILL_MIN_CONTEXT_SIZE} and ${PREFILL_MAX_CONTEXT_SIZE}`,
      );
    const out = { ...common, contextSizes: sizes };
    if (opts.contextLength != null) out.contextLength = opts.contextLength;
    return out;
  }
  const concurrencies = opts.concurrencies == null
    ? [1]
    : integerList(
      opts.concurrencies,
      "concurrencies",
      (n) => ALLOWED_CONCURRENCIES.has(n),
      `1,2,3,4,5,6,8,10,12,16,24,32`,
    );
  let maxTokens = 400;
  if (opts.maxTokens != null) {
    maxTokens = Number(opts.maxTokens);
    if (!Number.isInteger(maxTokens) || maxTokens < MIN_MAX_TOKENS || maxTokens > MAX_MAX_TOKENS) {
      throw new TypeError(`maxTokens must be an integer between ${MIN_MAX_TOKENS} and ${MAX_MAX_TOKENS}`);
    }
  }
  return {
    ...common,
    concurrencies,
    maxTokens,
    promptType: caseName,
    temperature: caseName === "code" ? 0 : 0.7,
    thinking: caseName === "prose",
    debug: false,
  };
}

function resultHasFailures(job) {
  return (
    !job || job.status !== "completed" || job.error != null ||
    !Array.isArray(job.results) || job.results.length === 0 ||
    job.results.some((row) => row.error != null || row.streamsFailed > 0 ||
      (Number.isFinite(row.samplesRequested) && row.samples < row.samplesRequested))
  );
}

/**
 * Run exactly one pinned native job, never a campaign or a retry-until-fast loop.
 *
 * Required options:
 *   case: "code" | "prose" | "prefill"
 *   host, port, modelId, outputDir, configurationId, engineInstanceId
 *   configuration: parent-provided fresh effective server controls (JSON object)
 *   freshSnapshot: parent-provided fresh identity/control snapshot (JSON object)
 *   metadata: parent campaign/model/storage/profiler metadata (JSON object)
 * Optional:
 *   runId (safe file stem; default UUID), sparkId (default "sparkdash-exact")
 *   tls (default false), apiKey (never written), contextLength (prefill only)
 *
 * Returns the envelope. The original public job is at envelope.native_job.
 * A job-level failure is returned and saved, not retried or rewritten.
 * An invocation exception is saved and rethrown with envelopePath/envelope.
 * Existing envelope or invocation state directories are never replaced.
 */
export async function runCase(opts = {}) {
  if (invocationActive) throw new Error("runCase calls must be awaited sequentially in one process");
  const caseName = requiredString(opts.case, "case");
  if (!["code", "prose", "prefill"].includes(caseName)) {
    throw new TypeError("case must be code, prose, or prefill");
  }
  const host = requiredString(opts.host, "host");
  const port = Number(opts.port);
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new TypeError("port must be 1..65535");
  const modelId = requiredString(opts.modelId, "modelId");
  const configurationId = requiredString(opts.configurationId, "configurationId");
  const engineInstanceId = requiredString(opts.engineInstanceId, "engineInstanceId");
  const configuration = requireJsonObject(opts.configuration, "configuration");
  const freshSnapshot = requireJsonObject(opts.freshSnapshot, "freshSnapshot");
  const metadata = requireJsonObject(opts.metadata, "metadata");
  const sparkId = opts.sparkId == null ? "sparkdash-exact" : requiredString(opts.sparkId, "sparkId");
  if (opts.apiKey != null && typeof opts.apiKey !== "string") throw new TypeError("apiKey must be a string");
  if (opts.tls != null && typeof opts.tls !== "boolean") throw new TypeError("tls must be a boolean");
  const runId = opts.runId == null ? randomUUID() : requiredString(opts.runId, "runId");
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(runId) || runId.endsWith(".json")) {
    throw new TypeError("runId must be a safe file stem without a .json extension");
  }
  const outputDir = path.resolve(requiredString(opts.outputDir, "outputDir"));
  const envelopePath = path.join(outputDir, `${runId}.json`);
  const stateDir = path.join(outputDir, runId);
  if (fs.existsSync(envelopePath)) throw new Error(`Refusing to replace existing output: ${envelopePath}`);
  fs.mkdirSync(outputDir, { recursive: true });
  // Non-recursive mkdir is also a cross-process exclusive reservation for runId.
  fs.mkdirSync(stateDir);
  invocationActive = true;

  const createdAt = new Date().toISOString();
  const benchmarkKind = caseName === "prefill" ? "prefill" : "decode";
  const nativeDir = path.join(stateDir, "native");
  const historyPath = path.join(nativeDir, benchmarkKind === "decode" ? "bench-history.json" : "prefill-bench-history.json");
  const activePath = path.join(nativeDir, benchmarkKind === "decode" ? "bench-active.json" : "prefill-bench-active.json");
  const requested = nativeOptions(caseName, opts, sparkId, host, port, modelId);
  const envelope = {
    schema: SCHEMA,
    protocol_id: PROTOCOL_ID,
    run_id: runId,
    created_at: createdAt,
    completed_at: null,
    benchmark_kind: benchmarkKind,
    workload: benchmarkKind === "prefill" ? "generic" : caseName,
    configuration_id: configurationId,
    configuration,
    engine_instance_id: engineInstanceId,
    engine: { host, port, tls: opts.tls === true, model_id: modelId, base_url: null },
    source: null,
    native_job: null,
    native_job_modified: false,
    options: {
      requested,
      effective_config: null,
      defaults: null,
      applied_defaults: benchmarkKind === "decode"
        ? { backend: null, contextTokens: 0, sampleHardware: null, resolveTarget: null, owner: null }
        : { ...(opts.contextLength == null ? { contextLength: null } : {}), resolveTarget: null, owner: null },
      api_key_present: Boolean(opts.apiKey?.trim()),
    },
    runtime: {
      node: process.version,
      platform: process.platform,
      arch: process.arch,
      exec_path: process.execPath,
      undici_version: installedUndiciVersion(),
    },
    client_host: {
      hostname: os.hostname(),
      platform: process.platform,
      arch: process.arch,
      measurement_location: "HTTP client executes on this host; engine location is parent-provided metadata",
    },
    metadata,
    fresh_snapshot: freshSnapshot,
    artifacts: { envelope_path: envelopePath, state_dir: stateDir, history_path: historyPath, active_path: activePath },
    status: "invocation_failed",
    failure: null,
    limitations: {
      complete_http_or_sse_raw_exposed: false,
      warmup_or_calibration_individual_results_exposed: false,
      prefill_individual_attempts_exposed: false,
      prefill_public_result: "One median-rate row per size plus samples/samplesRequested counts; the pinned source does not expose individual attempt results. Errors/notices exposed by the public job are preserved without reconstruction.",
      cached_tokens_zero_means: "Upstream uses 0 for unknown/none; 0 alone does not prove absence of a prefix-cache hit. Parent fresh server configuration is separate evidence.",
    },
  };

  let stage = "source";
  let invocationError = null;
  try {
    envelope.source = collectSourceEvidence();
    if (!envelope.source.verified) throw new Error("Pinned source hash or size mismatch; no native job started");
    stage = "runtime";
    const [major, minor] = process.versions.node.split(".").map(Number);
    if (major < 22 || (major === 22 && minor < 19)) throw new Error("undici 8.11.2 requires Node >=22.19.0");
    if (envelope.runtime.undici_version !== "8.11.2") throw new Error("Install the pinned undici 8.11.2 dependency before running a native job");
    stage = "import";
    const native = await loadNativeModules(path.join(stateDir, "import-singletons"));
    envelope.engine.base_url = native.target.formatLlmBaseUrl({ host, port, tls: opts.tls === true });
    fs.mkdirSync(nativeDir);
    const Manager = benchmarkKind === "decode" ? native.decode.DecodeBenchManager : native.prefill.PrefillBenchManager;
    const manager = new Manager(historyPath, activePath);
    envelope.options.defaults = benchmarkKind === "decode" ? native.decode.DECODE_BENCH_DEFAULTS : native.prefill.PREFILL_BENCH_DEFAULTS;

    let resolveTerminal;
    let rejectTerminal;
    const terminalPromise = new Promise((resolve, reject) => {
      resolveTerminal = resolve;
      rejectTerminal = reject;
    });
    // The upstream sink receives an internal object. Always use getJob to obtain
    // the public snapshot, including PrefillBench's history fallback after finish.
    manager.setEventSink((_kind, internalJob) => {
      try {
        const publicJob = manager.getJob(internalJob.benchId);
        if (!publicJob || !TERMINAL.has(publicJob.status)) {
          throw new Error("Native terminal event did not expose a terminal public job");
        }
        resolveTerminal(publicJob);
      } catch (error) {
        rejectTerminal(error);
      }
    });
    stage = "start";
    const startOptions = { ...requested, apiKey: opts.apiKey ?? null };
    const started = manager.start(startOptions);
    envelope.options.effective_config = started.config;
    stage = "await_terminal_event";
    envelope.native_job = await terminalPromise;
    envelope.options.effective_config = envelope.native_job.config;
    envelope.status = envelope.native_job.status;
  } catch (error) {
    invocationError = error;
    envelope.failure = errorRecord(error, stage);
  } finally {
    envelope.completed_at = new Date().toISOString();
    try {
      fs.writeFileSync(envelopePath, `${JSON.stringify(envelope, null, 2)}\n`, { flag: "wx", mode: 0o600 });
    } finally {
      invocationActive = false;
    }
  }
  if (invocationError) {
    invocationError.envelopePath = envelopePath;
    invocationError.envelope = envelope;
    throw invocationError;
  }
  return envelope;
}

export async function closeNativeClient() {
  // This closes the source-owned dispatcher through its own public helper.
  // Call once after the last runCase if the importing process must exit.
  if (!nativeModulesPromise) return;
  const native = await nativeModulesPromise;
  return native.streaming.closeLlmStreamAgent();
}

const HELP = `Usage: node runner.mjs --options /absolute/path/to/options.json

One CLI invocation executes exactly one native job; no automatic repeats or retries.
Required JSON keys:
  case: "code" | "prose" | "prefill"
  host, port, modelId, outputDir, configurationId, engineInstanceId
  configuration: fresh effective server controls (JSON object)
  freshSnapshot: fresh parent identity/control snapshot (JSON object)
  metadata: parent campaign/model/storage/profiler metadata (JSON object)
Optional keys:
  runId (file stem; defaults to UUID), sparkId, tls, apiKey
  contextLength (prefill only; source context-fit rule applies)

code: C1, maxTokens 400, temperature 0, thinking false, debug false
prose: C1, maxTokens 400, temperature 0.7, thinking true, debug false
prefill: generic contextSizes [4096,16384]; source warmups/calibration/repeats
No backend override, deep-context fill, request rewriting, extra warmup, or retry.
The original source may itself perform its documented HTTP-400 fallback.
Output: outputDir/<runId>.json; native files: outputDir/<runId>/native/.
Existing output or state directory causes refusal. Nonzero exit preserves failures.
Import API: await runCase(opts); await closeNativeClient() after the last job.
`;

async function cli() {
  const args = process.argv.slice(2);
  if (args.length === 1 && ["--help", "-h"].includes(args[0])) {
    process.stdout.write(HELP);
    return;
  }
  if (args.length !== 2 || args[0] !== "--options") {
    throw new Error("Use --options /absolute/path/to/options.json, or --help");
  }
  const options = JSON.parse(fs.readFileSync(path.resolve(args[1]), "utf8"));
  try {
    const envelope = await runCase(options);
    process.stdout.write(`${JSON.stringify({ envelope_path: envelope.artifacts.envelope_path, run_id: envelope.run_id, status: envelope.status })}\n`);
    if (resultHasFailures(envelope.native_job)) process.exitCode = 1;
  } catch (error) {
    process.stderr.write(`${JSON.stringify({ error: error.message, envelope_path: error.envelopePath ?? null })}\n`);
    process.exitCode = 1;
  } finally {
    await closeNativeClient().catch((error) => {
      process.stderr.write(`${JSON.stringify({ dispatcher_close_error: error.message })}\n`);
      process.exitCode = 1;
    });
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  cli().catch((error) => {
    process.stderr.write(`${JSON.stringify({ error: error.message })}\n`);
    process.exitCode = 1;
  });
}
