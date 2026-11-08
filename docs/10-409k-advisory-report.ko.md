# 10. 409,600 컨텍스트 확장: 외부 자문용 상세 보고서

이 문서는 이 작업을 처음 접하는 외부 AI 또는 엔지니어를 위해 쓰였다. 0장에서 모델·퀀트·하드웨어·저장소·
측정 규약·용어를 모두 설명하고, 1장부터 실제로 실행한 세 경로와 실측 수치, 실패 지점의 정확한 오류를
순서대로 적는다.

영어 요약은 `docs/06-results.md`의 "409,600 window (config extension, sep-e and sep-f)" 절에 있고,
경로 설계와 실행 기록은 `docs/09-context-extension-routes.ko.md`에 있다. 이 문서는 그 둘을 자문용으로
재구성한 것이다.

---

## 0. 배경

### 0.1 한 문장 요약

125B MoE 모델을 INT4로, GB10 통합 메모리 박스 한 대와 PCIe로 붙인 RTX 3090 한 장으로 서빙하면서,
컨텍스트 윈도우를 262,144에서 409,600으로 넓히는 작업이다. 세 가지 경로를 실제로 실행했고, 두 경로는
성공했으며 한 경로는 백엔드 두 곳에서 막혔다.

### 0.2 이 보고서가 답하는 질문

1. 409,600 윈도우가 이 하드웨어에서 실제로 가능한가.
2. 가능하다면 어떤 설정이 필요하고, 기존 262,144 대비 얼마를 잃는가.
3. 더 넓히려면(524,288~819,200) 어느 구성 요소를 줄이거나 어떤 커널을 패치해야 하는가.

### 0.3 하드웨어

| 장치 | 성격 | 이 작업에서의 역할 |
|---|---|---|
| GB10 (SM121) | CPU와 GPU가 121.7 GiB 통합 메모리를 공유, 246 GB/s | 모델 가중치 대부분과 PLE 페이지 캐시 |
| RTX 3090 (SM86) | 별도 24 GiB GDDR6X, 833 GB/s, eGPU | hot-expert 캐시와 KV 캐시. 실질 병목 |
| 링크 | PCIe Gen4 x4, Non-P2P | 두 카드가 서로 메모리를 직접 못 읽고 호스트 RAM 경유 |
| 호스트 RAM | 121 GiB | PLE 51 GiB와 페이지 캐시 |
| PLE 저장소 | USB 외장 (10 Gbps T7 Shield 또는 20 Gbps NVMe), 사본 2벌 동일 | mmap으로 읽음 |

실측 왕복 지연은 20KB 기준 33.14µs, `nvbandwidth` 복사 대역폭은 host→device 6.20 GB/s,
device→host 6.59 GB/s다. 3090은 Zagitta/gpu-unlock으로 BF16 연산이 70~80% 빨라진 상태이며, 이 작업은
그 상태를 변경하지 않는다.

### 0.4 모델과 퀀트

대상은 Hugging Face의 `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`다.

| 항목 | 값 |
|---|---|
| 총 파라미터 | 125B |
| 토큰당 활성 | 4.8B |
| 구조 | MoE, routed expert 512개 유지, 토큰당 5개 선택 (원본은 10개, 활성 6.0B) |
| 양자화 | INT4 AutoRound (Intel 계보, W4A16) |
| healing | always-on 공유 컴포넌트 37.75M 파라미터, 14,667 예제 |
| lm_head | int4 GPTQ-Marlin g128 (BF16 대비 top-1 97.2%), int8 99.6% |
| draft head | BF16 top-10 유지 |
| 네이티브 최대 위치 | `max_position_embeddings = 262144` |

특수 구성 요소는 네 가지다. **MTP**는 투기적 디코딩용 초안 머리이고, **QSA ring**은 attention 링 버퍼로
용량이 MTP 깊이를 따른다(15에서 20행, 5에서 12행). **GDN**은 상태 유지형 선형 attention 36층이고,
**PLE**는 가중치와 별도인 51 GiB int4 n-gram 조회 테이블로 Layer 1에서만 쓰인다.

expert 삭제는 없었다. 512개 routed expert가 전부 남아 있고 라우팅 깊이만 줄었다. "산문 expert를
잘라냈다"는 설명은 이 체크포인트에서 사실이 아니다.

### 0.5 저장소

| 저장소 | 종류 | 역할 | 고정 |
|---|---|---|---|
| `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound` | HF 모델 | 서빙 대상 | 체크포인트 |
| `azampatti/Qwen3.8-Flash-Next-Int4-FAST` | HF 모델/레시피 | 저자 레시피, 수락률 출처 | 문서 |
| `azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ` | HF 모델 | 비교용 대안 | 체크포인트 |
| `nvidia/Qwen3.8-Flash-Next-FP4-Local` | HF 모델 | 계보 원본, 적재 안 함 | — |
| `MiaAI-Lab/sparkDash` | GitHub | 측정 하네스 | `7c2b9638…` |
| `MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold` | GitHub | 검증 도구 | `20e15e41…` |
| `Saren-Arterius/qwen3.8-Flash-DGX-AutoRound` | GitHub | 엔진 이미지 기반 | `a5b-autoround-bottleneck-q2` |
| `Zagitta/gpu-unlock` | GitHub | 3090 BF16 unlock | — |

엔진은 정품 vLLM이 아니라 포크다. `vLLM 0.1.dev20073+g8e685d198`, `torch 2.13.0+cu130`, `triton 3.7.1`
위에 `peer_worker.py`와 `benchmark_worker.py`를 바인드 마운트한다. 이 두 파일이 owner/helper 장치 분할,
trunk 이동, draft 배치, PLE 경로를 결정한다.

### 0.6 측정 규약

decode는 스트리밍 토큰 타임스탬프로 계산한다.

```
decodeTps = (completionTokens - 1) / (tLast - tFirst) * 1000
```

첫 토큰은 prefill 결과라 제외한다. 요청은 `min_tokens = max_tokens`, `ignore_eos`, `stop: []`로 고정되고
400토큰, 작업당 32토큰 워밍업 1회다. 산문은 temp 0.7 thinking on, 코드는 temp 0 thinking off다.

prefill은 시도마다 고유 salt를 붙이고, TTFT에서 보정 overhead를 뺀다.

```
rate = promptTokens / (ttftMs - overheadMs)
```

overhead 보정은 TTFT의 절반을 넘지 못하고, 비중이 30%를 넘으면 `lowConfidence`로 표시된다. 반복은
32K 이하 3회, 131K 이하 2회, 그 이상 1회다.

### 0.7 용어

| 용어 | 뜻 |
|---|---|
| A5B | 토큰당 활성 약 5B. 실제 4.8B |
| MoE | 토큰마다 일부 expert만 활성화하는 구조 |
| MTP | Multi-Token Prediction, 투기적 디코딩 초안 머리 |
| 수락률 | draft 제안 중 타깃이 받아들인 비율 |
| PLE | 가중치 밖의 int4 n-gram 조회 테이블, 51 GiB |
| QSA ring | attention 링 버퍼, MTP 깊이를 따름 |
| GDN | 상태 유지형 선형 attention 층 |
| block_size | KV 페이지 크기(토큰) |
| KV cache | 생성 중 과거 토큰의 key/value 저장소 |
| admission | 기동 시 보고하는 KV 수용량 |
| 1.04x | 요청 윈도우 대비 KV가 1.04배 들어간다는 뜻 |
| TTFT | 첫 토큰까지의 시간 |
| OOM | 메모리 할당 실패 |
| eGPU | 외장 GPU |
| Non-P2P | 두 GPU가 서로 메모리를 직접 못 읽는 연결 |
| owner/helper | 포크가 정한 두 장치 역할 |
| YaRN | RoPE 스케일링으로 컨텍스트를 늘리는 기법 |
| 봉투 | sparkDash가 작업 1건마다 남기는 불변 JSON |

---

## 1. 문제 정의

모델의 네이티브 한계는 262,144다. 더 큰 `--max-model-len`은 기동 시 거부된다.

```
Value error, User-specified max_model_len (348800) is greater than the derived max_model_len
(max_position_embeddings=262144.0)
```

초기에는 `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`로 409,600을 강제했다. admission은 통과했지만 기동 피크에서
OOM이 났다.

```
GPU KV cache size: 481,483 tokens, Maximum concurrency for 409,600 tokens per request: 1.18x
memory allocation failed with OOM on device 0   (1.28 GB request, 44 MB free)
```

당시 해석은 "세 상주 객체가 24 GiB에 안 들어간다"였다. draft 9.09 GiB + target 6.19 GiB + KV 16 GiB =
31.28 GiB. 그 뒤 외부 피드백이 두 가지를 지적했다.

1. 실패 원인은 총량이 아니라 **기동 피크**다. 262,144의 정상 상태는 22,913 MiB지만 스테이징 피크는 그보다
   약 794 MiB 높고, 여기에 KV를 2 GiB 더하면 24,576 MiB를 넘는다.
2. KV를 14 GiB로 유지해도 409,600이 들어간다. 36개 GDN 상태는 컨텍스트 길이와 무관한 고정 블록이라
   요청당 블록 수가 `⌈L/1792⌉ + 67`로만 늘어난다.

이 보고서는 그 지적을 실제로 검증한 기록이다.

---

## 2. 경로 A: 설정만으로 409,600

### 2.1 진단

체크포인트의 `config.json`을 열어 보니 피드백의 진단이 정확했다.

- 최상위에 `max_position_embeddings`가 **없다**.
- `text_config.max_position_embeddings`만 262144다.
- `text_config.rope_parameters.rope_type`은 `default`다.

즉 vLLM이 `text_config` 값으로 윈도우를 유도했고, 그보다 큰 값은 거부됐다. MTP 드래프트도 같은 구조라서
드래프트의 RoPE 테이블이 262,144행에 고정됐다.

### 2.2 구현

원본 체크포인트는 읽기 전용이므로 래퍼 뷰를 만들었다. `target-409k`와 `draft-409k`가 그것이고, 나머지
파일은 심볼릭 링크, `config.json`만 사본으로 둔다.

```
max_position_embeddings: 409600            (최상위와 text_config 양쪽)
rope_type: yarn, factor: 1.5625            (= 409600 / 262144)
original_max_position_embeddings: 262144
mrope_interleaved: true, mrope_section: [11, 11, 10]
partial_rotary_factor: 0.25, rope_theta: 10000000
```

이 과정에서 포크에 필요한 수정이 하나 더 드러났다. `transformers`의 `standardize_rope_params()`가
`PretrainedConfig.__init__`보다 먼저 실행되면서 `self.max_position_embeddings`를 읽는데, 그 시점에는
속성이 아직 없다.

```
AttributeError: 'Qwen4ExpConfig' object has no attribute 'max_position_embeddings'
```

그래서 `Qwen3_8FlashNextConfig.__init__` 첫머리에서 `kwargs`의 값을 먼저 `self`에 설정하도록 모듈을
패치하고, `run_engine_hostnet.sh`에 그 파일을 읽기 전용으로 마운트했다.

### 2.3 기동 결과

```
GPU KV cache size: 427,468 tokens, Maximum concurrency for 409,600 tokens per request: 1.04x
engine_load_seconds 297.25
```

3090은 22,985 MiB(여유 1,591 MiB), 호스트 RAM 여유 52 GiB였고 OOM은 없었다. 피드백의 산출(311블록,
요청당 296블록, 1.0507x)과 방향이 같고 크기는 조금 작다. 어느 쪽이든 결론은 같다: 14 GiB KV로 409,600이
들어간다.

### 2.4 측정

| 작업 | 회차 | median tok/s |
|---|---|---:|
| code | 01 | 177.80 |
| code | 02 | 197.91 |
| code | 03 | 181.78 |
| prose | 01 (콜드) | 27.34 |
| prose | 02 | 67.28 |
| prose | 03 | 72.84 |

| 크기 | promptTokens | 샘플 | prefillTps | TTFT |
|---|---:|---:|---:|---:|
| 4K | 4,134 | 3 | 2,753.36 | 1.64 s |
| 16K | 16,419 | 3 | 2,688.15 | 6.24 s |
| 32K | 32,805 | 3 | 2,623.35 | 12.64 s |

모든 행이 `streamsOk == 1`, `streamsFailed == 0`이다.

### 2.5 262,144 경계

피드백이 지적한 첫 함정은 262,145번째 토큰에서 MTP 드래프트의 RoPE 테이블이 범위를 벗어나 크래시하는
현상이다. 262,209토큰 프롬프트로 16토큰을 강제 생성했다.

```
prompt_tokens: 262,209
completion_tokens: 16
finish_reason: length
ok: true
```

엔진 코어는 죽지 않았다. 즉 드래프트가 확장된 `max_position_embeddings`와 YaRN 테이블을 상속했다.

---

## 3. 경로 B: MTP expert INT4

### 3.1 근거

MTP 드래프트 계층의 routed expert 1,536텐서(512 expert × 3 proj)는 `model_extra_tensors.safetensors`에
BF16으로 남아 있다. expert 하나당 파라미터는 4,915,200개(3 × 2,560 × 640)이고, 512개 합계는 약 2.52B로
5,033,164,800바이트(4.69 GiB)다. 여기에 드래프트용 attention·공유 expert·임베딩·헤드 버퍼 4.40 GiB가
더해져 3090에서 9.09 GiB를 차지한다.

### 3.2 변환

`quantize_mtp_int4.py`가 이 expert들만 int4 GPTQ-Marlin g128 RTN으로 다시 패킹한다. 이 체크포인트의
`quantization_config`는 이미 도구가 요구하는 조건을 만족한다.

```
quant_method gptq  bits 4  group_size 128  sym True  desc_act False
dynamic includes "-:.*layers\.48\..*"
```

다만 `model.safetensors.index.json`의 `metadata.total_size`(175,330,224,632)가 실제 인덱스 텐서 바이트
합(74,914,614,008)과 달라서 도구가 오류로 처리한다. 그 검사만 경고로 낮춘 사본으로 변환했다.

변환 후 MTP expert는 1,307,443,200바이트(1.22 GiB)가 되고, 회수량은 3.47 GiB다.

### 3.3 기동 시도 세 번

**첫 실패, 링크 경로.** 뷰의 심볼릭 링크가 컨테이너에 마운트되지 않는 `/models/...`를 가리켰다. 엔진은
`-v "$MODELS:$MODELS:ro"`로 `/srv/vault/int4-models`만 마운트하므로 `/models`가 보이지 않는다.

```
Couldn't instantiate the backend tokenizer
```

링크를 `/srv/vault/int4-models/...` 절대 경로로 바꾸자 풀렸다.

**두 번째 실패, 디렉터리 권한.** 변환 산출물 디렉터리가 `drwx------`(700)라 엔진이 tokenizer를 읽지 못했다.
`chmod 755`로 열자 진행했다.

**세 번째, 17 GiB OOM.**

```
GPU KV cache size: 518,185 tokens, Maximum concurrency for 409,600 tokens per request: 1.27x
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.27 GiB.
GPU 0 has a total capacity of 23.61 GiB of which 116.19 MiB is free.
```

16 GiB도 같은 지점에서 실패했다.

```
GPU KV cache size: 487,946 tokens, Maximum concurrency for 409,600 tokens per request: 1.19x
torch.OutOfMemoryError: Tried to allocate 1.19 GiB. 1004.19 MiB free.
```

14 GiB에서 성공했다.

```
GPU KV cache size: 427,468 tokens, Maximum concurrency for 409,600 tokens per request: 1.04x
engine_load_seconds 305.18
```

즉 MTP expert를 int4로 줄여 3.47 GiB를 회수했지만, 그 여유를 KV로 다시 밀어넣으면 기동 피크가 다시 넘는다.
409,600에서 이 카드의 실질 KV 상한은 14 GiB다.

### 3.4 측정

| 작업 | 회차 | median tok/s |
|---|---|---:|
| code | 01 | 184.67 |
| code | 02 | 185.89 |
| code | 03 | 199.78 |
| prose | 01 (콜드) | 27.00 |
| prose | 02 | 70.93 |
| prose | 03 | 71.90 |

| 크기 | promptTokens | 샘플 | prefillTps | TTFT |
|---|---:|---:|---:|---:|
| 4K | 4,134 | 3 | 2,813.41 | 1.57 s |
| 16K | 16,420 | 3 | 2,699.81 | 6.19 s |
| 32K | 32,809 | 3 | 2,629.04 | 12.58 s |

경로 A 대비 code +2.3%, prefill 4K +2.2%, prose −1.3%다. 피드백이 예상한 +2~4% 범위 안이다.

---

## 4. 경로 C: 8비트 KV

### 4.1 전제

이 포크의 `qsa.py`는 `supported_kv_cache_dtypes`에 `auto`, `bfloat16`, `fp8`, `fp8_e4m3`을 두고 있다.
즉 `fp8_e4m3`는 코드 수정 없이 시도할 수 있다. `ops/qsa.py`도 `uint8 → float8_e4m3fn` view를 지원한다.

### 4.2 fp8_e4m3 시도

admission은 크게 늘었다.

```
GPU KV cache size: 677,415 tokens, Maximum concurrency for 409,600 tokens per request: 1.65x
```

그리고 Triton이 커널 컴파일에서 거부했다.

```
triton.compiler.errors.CompilationError
ValueError("type fp8e4nv not supported in this architecture. The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')")
```

피드백이 말한 SM86 제약이 그대로 재현됐다. 수용량 1.65x는 피드백의 673,997~819,200 예상과 일치한다.

### 4.3 fp8_e5m2 시도

Triton이 지원한다고 밝힌 `fp8e5`(e5m2)로 패치했다. `qsa.py`의 두 dtype 검사 목록에 `fp8_e5m2`를 넣고,
`ops/qsa.py`의 view를 `float8_e5m2`로, quant mode 판정과 assert를 e5m2까지 넓혔다.

```
NotImplementedError: FlashAttention does not support fp8_e5m2 kv-cache on this device.
```

즉 이 포크와 SM86 조합에서는 `fp8_e4m3`는 Triton이, `fp8_e5m2`는 FlashAttention이 막는다. 8비트 KV는
이 조합에서 UNCHECKED다.

### 4.4 대안 경로의 상태

피드백이 지목한 `int8_per_token_head`(`KV_QUANT_MODE = 2`)는 이 포크에 배선돼 있지 않다. `ops/qsa.py`에는
`KV_QUANT_MODE = 1`(FP8 per-tensor) 경로만 주석과 함께 남아 있다. int8을 쓰려면 커널과 스케일 배선을
새로 넣어야 하고, 이 작업 범위에서는 실행하지 않았다.

---

## 5. 262,144 대비 비용

| 지표 | 262,144 | 경로 A | 경로 B |
|---|---:|---:|---:|
| 윈도우 | 262,144 | 409,600 | 409,600 |
| admission | `380,966` / 1.45x | `427,468` / 1.04x | `427,468` / 1.04x |
| code decode 중앙값 | 194.70 | 181.78 (−6.6%) | 185.89 (+2.3% vs A) |
| prose 워밍 중앙값 | 73.61 | 72.84 (−1.0%) | 71.90 |
| prefill 4K | 2,814.22 | 2,753.36 (−2.2%) | 2,813.41 |
| prefill 16K | 2,723.52 | 2,688.15 (−1.3%) | 2,699.81 |
| prefill 32K | 2,649.84 | 2,623.35 (−1.0%) | 2,629.04 |
| load | 218.27 | 297.25 | 305.18 |

윈도우를 56% 늘리는 비용은 decode 6.6%, prefill 1~2%다. 피드백의 "속도 저하 없음" 주장은 prefill에는
맞고 decode에는 약간 낙관적이다.

---

## 6. 자문에게 남기는 질문

1. 이 포크에서 14 GiB보다 큰 KV를 409,600 윈도우에 넣는 방법이 있는가? MTP int4로 3.47 GiB를 회수했지만
   기동 피크가 그대로 넘는다. 피크를 낮추는 설정(스테이징 순서, `expandable_segments`, 로드 포맷)이
   실제로 유효한가?
2. `int8_per_token_head` KV를 SM86에서 배선하려면 어느 정도의 커널 작업이 필요한가? Triton에서 int8
   per-token-head 스케일을 128B 정렬에 맞추는 제약이 실제 병목인가?
3. YaRN factor를 1.5625로 정확히 맞췄는데, 524,288까지 확장할 때 factor 2.0에서 단문 품질 손실이 측정
   가능한가?
4. 409,600 윈도우에서 14 GiB KV는 동시성 1.04x다. 다중 턴 에이전트 루프에서 prefix caching을 켜면
   고정 블록이 67에서 73으로 늘어 수용량이 줄어드는데, 캐시 히트 이득이 그 손실을 넘는가?
5. PLE 51 GiB 페이지 캐시가 프리필 중 축출되면 조회 지연이 1.3~2.5ms에서 5~30ms로 뛴다. 호스트 RAM
   52 GiB 여유를 pin으로 고정하는 것이 409,600 롱프리필에서 측정 가능한 이득을 주는가?

---

## 7. 아티팩트

| 파일 | 내용 |
|---|---|
| `artifacts/recipe/sep-e.json` | 경로 A 실행 상태(409,600, 14 GiB, load 297.25초) |
| `artifacts/recipe/a-sep-e-k6-open-*.json` | 경로 A decode·prefill 봉투 7건 |
| `artifacts/recipe/sep-f.json` | 경로 B 실행 상태(409,600, 14 GiB, load 305.18초) |
| `artifacts/recipe/a-sep-f-k6-open-*.json` | 경로 B decode·prefill 봉투 7건 |
| `artifacts/recipe/sep-g.json` | 경로 C 실행 상태(실패) |
| `artifacts/recipe/sep-g.server.log` | 경로 C의 Triton·FlashAttention 오류 원문 |
| `artifacts/recipe/sep-d.server.log` | 최초 409,600 강제 시도의 device split 로그 |
| `docs/06-results.md` | 409,600 절과 262,144 절의 전체 표 |
| `docs/09-context-extension-routes.ko.md` | 경로 설계와 실행 기록 |

측정 규약 전체는 `docs/01-measurement-method.md`, 산문 감속의 근거는
`docs/02-why-prose-is-slower.md`, GB10 단독과 eGPU 비교는 `docs/07-gb10-solo-vs-egpu.md`에 있다.
