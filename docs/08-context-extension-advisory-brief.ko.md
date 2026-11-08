# 8. 컨텍스트 확장: 외부 자문용 브리프 (처음 읽는 독자용 전체 맥락)

이 문서는 이 작업을 처음 접하는 외부 AI 또는 엔지니어를 위해 쓰였다. 앞의 배경을 모른 채 읽어도
"무엇을 어떤 하드웨어에서 어떤 모델로 재고 있고, 지금 어디서 막혔는가"를 파악할 수 있게 구성했다.

- 0장: 배경. 모델, 퀀트, 하드웨어, 관련 저장소, 측정 규약, 용어, 지금까지의 여정.
- 1~9장: 실측 기록. 확정 구성, 수치, 실패한 시도와 그 오류, 상류 상태, 열린 질문, 아티팩트.

모든 수치는 9장의 아티팩트와 `docs/06-results.md`에서 인용했다. 모델링이나 추측으로 채운 값은 없고,
확인이 안 된 항목은 그 자리에서 "미확인"으로 표시했다. 영어 원문은 같은 디렉터리의
`08-context-extension-advisory-brief.md`이며 수치는 동일하다.

---

## 0. 배경

### 0.1 한 문장 요약

소비자용 워크스테이션 두 장비(GB10 통합 메모리 박스 하나 + PCIe로 붙인 RTX 3090 한 장)로, 125B
파라미터 MoE 모델을 INT4로 서빙하면서 컨텍스트 윈도우를 최대한 넓히려는 실험이다. 지금은 262,144
토큰에서 확정됐고, 그 두 배인 409,600 토큰을 시도했지만 24 GiB 카드의 메모리 산술 때문에 실패했다.

### 0.2 이 문서가 답하려는 질문

1. 이 모델을 이 하드웨어에서 어디까지 밀어붙일 수 있는가. 특히 **컨텍스트 윈도우**를 얼마나 넓힐 수
   있는가.
2. 윈도우를 넓힐 때 무엇이 손실되는가. decode 속도, prefill 속도, 메모리 여유.
3. 윈도우를 더 넓히려면 어느 구성 요소를 줄이거나 옮겨야 하는가. 그리고 그 이동이 이 포크 엔진과
   현재 vLLM 빌드에서 가능한가.

### 0.3 하드웨어가 특이한 이유

이 환경은 일반적인 멀티 GPU 서버가 아니다. 두 장치의 성격이 근본적으로 다르다.

| 장치 | 성격 | 이 실험에서의 역할 |
|---|---|---|
| GB10 (SM121) | CPU와 GPU가 같은 121.7 GiB 통합 메모리를 공유. 대역폭 246 GB/s | 모델 가중치 대부분과 PLE 테이블의 페이지 캐시를 담당 |
| RTX 3090 (SM86) | 별도 24 GiB GDDR6X, 대역폭 833 GB/s. eGPU로 연결 | hot-expert 캐시와 KV 캐시를 담당. 실질적 병목 |

연결은 PCIe Gen4 x4이고 **Non-P2P**다. 두 카드가 서로 직접 메모리를 읽지 못하고 호스트 RAM을
경유한다. 실측 왕복 지연은 20KB 기준 33.14µs, `nvbandwidth`로 잰 복사 대역폭은 host→device 6.20 GB/s,
device→host 6.59 GB/s다. 이 수치는 `artifacts/recipe/sep-d.egpu.log`에 남아 있다.

3090은 소프트웨어 unlock이 적용돼 BF16 연산이 원래보다 70~80% 빨라진 상태다(Zagitta/gpu-unlock,
142~157 TFLOPS). 이 unlock은 이 레시피가 만든 것이 아니고, 이 레시피는 그 상태를 변경하지 않는다.

저장소도 특이하다. 모델의 PLE 테이블 51 GiB를 USB 외장 디스크에 두고 mmap으로 읽는다. 10 Gbps
Samsung T7 Shield와 20 Gbps USB NVMe 두 벌이 있고, 두 사본은 바이트 단위로 동일하다(총
52,259,870,014 바이트).

### 0.4 모델이 무엇인지

대상 체크포인트는 Hugging Face의
`azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`다. Qwen3.8-Flash-Next 계열을 한 사용자가
INT4로 재양자화하고 일부를 재학습한 파생 모델이다.

| 항목 | 값 |
|---|---|
| 총 파라미터 | 125B |
| 토큰당 활성 파라미터 | 4.8B (`A5B`는 "약 5B 활성"을 뜻하는 이름) |
| 구조 | MoE. routed expert 512개 유지, 토큰당 5개 선택 |
| 원본 라우팅 | 토큰당 10개 선택, 활성 6.0B |
| 양자화 | INT4 AutoRound (Intel AutoRound 계보, W4A16) |
| healing | always-on 공유 컴포넌트 1개, 37.75M 파라미터 |
| healing 데이터 | 코드·수학·일반 텍스트·툴 호출·롱컨텍스트 14,667 예제 |
| lm_head | int4 GPTQ-Marlin g128 (BF16 대비 top-1 일치 97.2%), int8 변형은 99.6% |
| draft head | BF16 top-10 유지 |
| 최대 위치 | `max_position_embeddings = 262,144` |

이 모델에는 특수 구성 요소가 몇 개 있다. 이 문서에서 계속 등장하므로 먼저 정리한다.

- **MTP (Multi-Token Prediction) draft head.** 추론 시 여러 토큰을 미리 제안하고 타깃 모델이 검증하는
  투기적 디코딩(speculative decoding)용 작은 머리다. 제안이 수락되면 한 번의 타깃 순전파로 여러 토큰이
  나온다. 이 레시피는 코드 경로에서 depth 15, 산문 경로에서 런타임 조절 depth를 쓴다.
- **QSA ring.** attention 구현 중 하나로, 링 용량이 MTP 깊이에 따라 달라진다. MTP 15에서 20행,
  MTP 5에서 12행이다. KV 페이지 크기(`block_size`)는 이 용량을 나눠야 한다.
- **GDN 층.** 상태를 유지하는 선형 attention 계열 층으로 36개가 있다.
- **PLE 테이블.** 모델 가중치와 별도로 존재하는 51 GiB int4 n-gram 조회 테이블이다. Layer 1에서만
  쓰이며 입력은 input_ids다. 가중치에 넣으면 한 장비에 들어가지 않아서 USB에 두고 mmap으로 읽는다.

### 0.5 INT4 AutoRound가 무엇인지

AutoRound는 Intel이 공개한 양자화 기법으로, 가중치를 4비트 정수로 반올림하되 반올림 오차를 학습으로
보정한다(W4A16: 가중치 4비트, 활성화 16비트). 이 체크포인트는 그 계보를 따르며, 원본 BF16 모델의
라우팅 깊이를 10에서 5로 줄이고 healing 컴포넌트를 붙인 변형이다.

중요한 점은 **expert를 삭제하지 않았다**는 것이다. 512개 routed expert는 전부 남아 있고, 토큰당 선택
개수만 줄었다. "산문 expert를 잘라냈다"는 흔한 설명은 이 체크포인트에서 사실이 아니다
(`docs/02-why-prose-is-slower.md`가 근거와 함께 정정한다).

### 0.6 관련 저장소

이 작업은 여러 외부 저장소 위에 서 있다. 처음 읽는 독자를 위해 역할과 고정 버전을 함께 적는다.

| 저장소 | 종류 | 이 작업에서의 역할 | 고정 |
|---|---|---|---|
| `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound` | Hugging Face 모델 | 서빙 대상 체크포인트 | 체크포인트 자체 |
| `azampatti/Qwen3.8-Flash-Next-Int4-FAST` | Hugging Face 모델/레시피 | 모델 작성자가 공개한 자기 레시피. 수락률 수치의 출처 | 문서 기준 |
| `azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ` | Hugging Face 모델 | 비교용 대안 체크포인트 | 체크포인트 자체 |
| `nvidia/Qwen3.8-Flash-Next-FP4-Local` | Hugging Face 모델 | 계보상 원본. 이 레시피는 적재하지 않고 참조만 한다 | — |
| `MiaAI-Lab/sparkDash` | GitHub | 측정 하네스. 모든 속도 수치의 출처 | `7c2b9638b025cd39fb43f95c225de62980f130ee` |
| `MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold` | GitHub | 품질·정확성 검증 도구 모음 (MIT) | `20e15e41f2f514a0d287710f4665ee1b50099ba41` |
| `Saren-Arterius/qwen3.8-Flash-DGX-AutoRound` | GitHub | 엔진 이미지의 기반 빌드 | 이미지 태그 `a5b-autoround-bottleneck-q2` |
| `Zagitta/gpu-unlock` | GitHub | RTX 3090 BF16 unlock. 이 레시피는 상태를 바꾸지 않음 | — |

엔진은 정품 vLLM이 아니라 포크 빌드다. `vLLM 0.1.dev20073+g8e685d198` 위에 `peer_worker.py`와
`benchmark_worker.py`를 바인드 마운트로 얹는다. 이 두 파일이 owner/helper 장치 분할, trunk 이동,
draft 배치, PLE 조회 경로를 결정한다.

측정 하네스 `sparkDash`는 포크가 아니라 **원본 바이트 그대로** 쓴다. 고정 커밋과 상류 HEAD
(`da20ac5b`, v2.77.0) 사이에서 측정 폐포에 속한 9개 파일이 바이트 단위로 동일함을 확인했고, 그
SHA-256을 `bench/sparkdash-exact/source-lock.json`에 기록했다. 이 저장소에서 상류와 다른 파일은
호출 래퍼 `runner.mjs` 하나뿐이다.

### 0.7 이 레시피가 실제로 하는 일

1. **체크포인트를 그대로 서빙한다.** expert 삭제 없음, top-5 라우팅 유지, 활성 4.8B.
2. **PLE를 가중치 밖에 둔다.** 51 GiB 테이블을 USB에서 mmap하고 `PREWARM=1`, `MADV_RANDOM=0`,
   32개 mmap worker, 512 fast rows로 읽는다. 그래서 모델이 GB10 한 대에 들어간다.
3. **포크 엔진으로 상주 투기적 디코딩을 돌린다.** 코드 경로 depth 15는 서버에 고정, 산문 경로 depth는
   RPC로 조절한다(`k6-open` = depth 6, 게이트 없음).
4. **sparkDash로 측정한다.** 모든 봉투(envelope)에 엔진 제어값을 기록해서, 수치가 항상 특정 구성에
   귀속되게 한다.

### 0.8 측정 규약

속도 수치는 MiaAI-Lab `sparkDash` 수집기가 만든다. decode는 스트리밍 토큰 타임스탬프로 계산한다.

```
decodeTps = (completionTokens - 1) / (tLast - tFirst) * 1000
```

첫 토큰은 prefill의 결과이므로 decode 구간에서 제외한다. 요청은 측정이 흔들리지 않게 고정된다:
`min_tokens = max_tokens`, `ignore_eos`, `stop: []`, 기본 400 토큰, 작업당 32 토큰 워밍업 1회.

| 워크로드 | temperature | thinking | 프롬프트 출처 |
|---|---:|---|---|
| `prose` (산문) | 0.7 | on | `DECODE_PROSE_PROMPT` — 해시맵 설명 과제 |
| `code` (코드) | 0.0 | off | `DECODE_CODE_TASKS` |

prefill은 시도마다 고유 salt 헤더를 붙여 prefix caching이 돕지 못하게 하고, TTFT에서 보정 overhead를
뺀 값으로 계산한다.

```
rate = promptTokens / (ttftMs - overheadMs)
```

overhead 보정은 TTFT의 절반을 넘지 못하고, overhead 비중이 30%를 넘으면 그 행은 `lowConfidence`로
표시된다. 보정값은 가정이 아니라 16토큰·3샘플로 측정한다. 반복은 32K 이하 3회, 131K 이하 2회, 그
이상 1회다. prefill은 동시 요청을 받지 않으므로 "C=4 prefill"이라는 수치는 이 방법에 존재하지 않는다.

### 0.9 용어집

| 용어 | 뜻 |
|---|---|
| A5B | 토큰당 활성 파라미터가 약 5B라는 이름. 실제 4.8B |
| MoE | Mixture of Experts. 토큰마다 일부 expert만 활성화하는 구조 |
| routed expert | 라우터가 토큰별로 고르는 expert. 이 모델은 512개 중 5개 |
| MTP | Multi-Token Prediction. 투기적 디코딩용 초안 머리 |
| draft / 타깃 | 초안을 제안하는 작은 경로(draft)와 검증하는 본체(target) |
| 수락률 (acceptance) | draft가 제안한 토큰 중 타깃이 받아들인 비율. decode 속도를 직접 좌우 |
| PLE | 모델 가중치와 별도인 int4 n-gram 조회 테이블. 51 GiB, Layer 1 전용 |
| QSA ring | attention의 링 버퍼. 용량이 MTP 깊이를 따름 (20행 / 12행) |
| GDN | 상태 유지형 선형 attention 계열 층 |
| block_size | KV 페이지(블록) 크기(토큰). decode 속도와 KV 메모리 효율을 동시에 좌우 |
| KV cache | 생성 중 과거 토큰의 key/value 저장소. 윈도우 크기의 주 제약 |
| admission | 엔진이 기동하며 보고하는 KV 수용량. 예: `380,966 tokens, 1.45x` |
| 1.45x | 요청한 윈도우(262,144) 대비 KV가 1.45배 들어간다는 뜻 |
| TTFT | Time To First Token. prefill이 끝나 첫 토큰이 나오기까지의 시간 |
| tok/s | 초당 토큰 수 |
| OOM | Out Of Memory. 할당 실패 |
| eGPU | 외장 GPU. 여기서는 PCIe Gen4 x4로 연결된 3090 |
| Non-P2P | 두 GPU가 서로 직접 메모리 접근을 못 하고 호스트 RAM을 경유하는 연결 |
| owner / helper | 이 포크가 정한 두 장치 역할. owner는 trunk와 KV, helper는 나머지 |
| AutoRound | Intel의 학습 기반 4비트 양자화 기법 |
| W4A16 | 가중치 4비트, 활성화 16비트 양자화 형식 |
| healing | 양자화 손실을 KL 자기증류로 보정하는 재학습 단계 |
| 봉투 (envelope) | sparkDash가 작업 1건마다 남기는 불변 JSON 기록 |

### 0.10 지금까지의 여정

| 단계 | 윈도우 | KV | block | batch | graphs | 결과 |
|---:|---:|---:|---:|---:|---|---|
| 기준 | 17,408 | 4 GiB | 160 | 8192 | PIECEWISE | 기준 레시피. admission 33,572토큰 1.93x |
| 페이지 실험 | 17,408 | 4 GiB | 1600 / 1632 | 8192 | PIECEWISE | decode 73~76 tok/s로 급락. 페이지 크기가 원인임을 분리 확인 |
| 1차 확장 | 204,800 | 12 GiB | 160 | 2048 | NONE | 정상. admission 299,323토큰 1.46x, OOM 0건 |
| 2차 확장 | 262,144 | 14 GiB | 160 | 2048 | NONE | 정상. 모델 최대치. admission 380,966토큰 1.45x |
| 3차 시도 | 409,600 | 16 GiB | 160 | 2048 | NONE | admission 481,483토큰 1.18x까지 통과 후 OOM |
| 복구 | 262,144 | 14 GiB | 160 | 2048 | NONE | 재확인. admission 380,966토큰, 로드 220.14초 |

이 과정에서 확정된 규칙 세 가지가 있다.

1. **페이지 크기가 decode를 결정한다.** MTP 깊이와 윈도우를 고정하고 block만 1600/1632로 바꾸면
   decode가 73~76 tok/s로 떨어지고, block 160에서는 192~194 tok/s가 나온다. 커널이 페이지에 할당된
   모든 행을 처리하기 때문에 큰 페이지는 일을 낭비한다. 또한 `block_size`는 QSA ring 용량을 나눠야
   한다. MTP 15에서 ring은 20행이므로 block은 20의 배수여야 하고, 1632는 MTP 5 전용이다.
2. **prefill batch는 2048이다.** 8192에서는 workspace가 KV와 함께 들어가지 않아 32K prefill에서 엔진
   코어가 죽는다. 2048에서는 261K까지 모든 크기가 통과한다.
3. **KV는 device 0에 고정된다.** 이 포크 엔진의 워커는 로드 장치와 무관하게 device 0 기준으로 KV와
   가중치를 할당한다. 그래서 KV 크기는 GB10의 121.7 GiB가 아니라 24 GiB 카드로 상한이 걸린다.

---

## 1. 하드웨어와 소프트웨어

| 항목 | 값 |
|---|---|
| 호스트 | JGX10-1, DGX Spark 급 |
| 장치 1 (owner) | RTX 3090, GPU 언락, SM86, 24,576 MiB GDDR6X, 833 GB/s |
| 장치 0 (vLLM 기본 owner) | GB10, SM121, 121.7 GiB 통합 메모리, 246 GB/s |
| 연결 | PCIe Gen4 x4, Non-P2P, 20KB 왕복 33.14µs |
| 실측 복사 대역폭 | host→device 6.20 GB/s, device→host 6.59 GB/s |
| 호스트 RAM | 121 GiB (로드 전 `MemAvailable` 약 118 GiB) |
| PLE 테이블 | 51 GiB int4 n-gram, USB 저장소에서 mmap, 두 사본 동일 |
| 엔진 이미지 | `qwen38-flash-dgx:a5b-autoround-bottleneck-q2` |
| vLLM | `0.1.dev20073` (포크 빌드) |
| PyTorch | `2.13.0+cu130` |
| Triton | `3.7.1` |

엔진은 포크다. `peer_worker.py`와 `benchmark_worker.py`를 이미지의 원본 위에 바인드 마운트한다. 포크는
작업을 owner 장치와 helper 장치로 나누고, 트렁크를 owner로 옮기며, draft head와 PLE 조회는 포크가
정한 위치에 둔다.

## 2. 확정 구성 (262,144)

```
--max-model-len 262144 --max-num-seqs 1 --kv-mib 14336 --block-size 160 \
--max-num-batched-tokens 2048 --mtp 15 --graphs NONE
```

admission 로그: `GPU KV cache size: 380,966 tokens, Maximum concurrency for 262,144 tokens per request: 1.45x`.
`engine_load_seconds 218.27`이며, 재기동 측정은 220.14초였다.

3090은 24,576 MiB 중 22,913 MiB(93%)를 쓰고, 호스트 RAM은 121.7 GiB 중 69.7 GiB를 쓴다. 카드의 실제
여유는 1.6 GiB다.

## 3. 262,144에서 측정한 수치

decode, C=1:

| 워크로드 | 속도 |
|---|---:|
| 코드 (temp 0, reasoning off) | 콜드 194.70 tok/s |
| 산문 (temp 0.7, reasoning on) | 콜드 33.18 tok/s, 워밍 후 76.77 / 73.61 tok/s |

prefill, batch 2048:

| 크기 | promptTokens | 샘플 수 | prefillTps | TTFT |
|---|---:|---:|---:|---:|
| 4K | 4,136 | 3 | 2,814.22 | 1.58 s |
| 16K | 16,423 | 3 | 2,723.52 | 6.14 s |
| 32K | 32,805 | 3 | 2,649.84 | 12.49 s |
| 200K | 200,041 | 1 | 2,331.81 | 85.90 s |
| 261K | 261,036 | 1 | 2,240.99 | 116.59 s |

204,800 구성(KV 12 GiB)은 코드 192.28, 산문 워밍 76.12 / 68.92, prefill 4K 2,844.53 / 16K 2,727.28 /
32K 2,657.05 / 200K 2,345.07(TTFT 85.41초), admission 299,323토큰 1.46x, OOM 0건이었다. 204,800에서
262,144로 올린 비용은 측정 가능한 수준이 아니었다. 모든 수치가 실행 간 잡음 안에 있다.

산문이 코드의 약 1/3인 이유는 draft 수락률이다. 모델 작성자는 T=0.5에서 전체 수락률 75%(코드 85%,
산문 63%, 롱컨텍스트 89%)를 공표했고, 포럼 카테고리 표는 CODE 82.1%, MATH 76.8%, LONGCTX 85.5%,
TOOL 68.9%, GENERAL 55.7%다. 산문은 제안된 토큰 중 약 40%가 기각되는 영역이고, thinking 경로가 그
분포에 토큰을 가장 많이 쓰기 때문에 속도가 낮게 나온다. 자세한 근거는
`docs/02-why-prose-is-slower.md`에 있다.

## 4. 실제로 실행한 사다리

| 윈도우 | KV | block | batch | graphs | 결과 |
|---:|---:|---:|---:|---|---|
| 17,408 | 4 GiB | 160 | 8192 | PIECEWISE | 기준 레시피 |
| 204,800 | 12 GiB | 160 | 2048 | NONE | 정상, admission 1.46x |
| 262,144 | 14 GiB(14,336 MiB) | 160 | 2048 | NONE | 정상, admission 1.45x |
| 409,600 | 16 GiB(16,384 MiB) | 160 | 2048 | NONE | admission 후 OOM |

## 5. 카드에서 409,600이 실패하는 이유

`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`을 주면 엔진이 `--max-model-len 409600`을 받아들이고, `--kv-mib 16384`에서
`GPU KV cache size: 481,483 tokens, Maximum concurrency for 409,600 tokens per request: 1.18x`를 보고한다.
그리고 기동 중에 죽는다.

```
memory allocation failed with OOM on device 0   (1.28 GB request, 44 MB free)
```

세 상주 객체가 함께 들어가지 않는다.

| 구성 요소 | GiB |
|---|---:|
| 3090의 draft 가중치 | 9.09 |
| 3090의 target 가중치 | 6.19 |
| KV 16 GiB | 16.00 |
| 합계 | 31.28 |

카드는 24 GiB다. 262,144에서는 같은 세 구성 요소가 계산상 29.28 GiB지만 실측 22.38 GiB다. vLLM이 로드 후
가중치 스테이징 버퍼의 일부를 해제하기 때문이며, 그래서 예약 수치는 상한선이다.

**이 문제는 KV를 옮기지 않고 해결됐다.** 체크포인트의 `config.json`에 최상위 `max_position_embeddings`가
없어서 vLLM이 `text_config` 값으로 윈도우를 유도했고, 그보다 큰 값은 거부했다. 최상위와 `text_config` 양쪽에
`max_position_embeddings: 409600`을 넣고 YaRN factor를 1.5625로 맞추면서 KV를 14 GiB로 유지하자
`GPU KV cache size: 427,468 tokens`(1.04x)로 409,600이 들어갔다. 기동 피크가 맞는 이유는 14 GiB가 OOM이
나는 값보다 2 GiB 낮기 때문이다. decode는 코드 194.70 → 181.78, 산문 워밍 73.61 → 72.84로 움직였고,
prefill 4K는 2,814.22 → 2,753.36이다. 전체 측정은 `docs/06-results.md`와
`docs/09-context-extension-routes.ko.md`에 있다.

## 6. 시도한 탈출 경로

**경로 1: draft head를 GB10으로 옮기기.** draft 요청이 helper 장치에 로드되도록 `peer_worker.py`를
수정했다. 그러나 배치 로그는 여전히 device 0에서 `primary_bytes 9,757,622,784`를 보고했고, 엔진 로그도
draft가 그대로 남았음을 확인했다. vLLM 워커는 로드 장치와 무관하게 device 0을 기준으로 KV와 가중치를
할당하므로 이 수정은 효과가 없었다.

**경로 2: prefix caching 활성화.** 이것이 없으면 모델 최대치를 넘는 윈도우는
`--mamba-block-size can only be set with --enable-prefix-caching`로 거부된다. prefix caching을 켜면 그
검사는 통과하지만 같은 CUDA OOM에 도달한다.

**경로 3: GB10을 device 0으로 만들기.** `QWEN38_VISIBLE_DEVICES`를 GB10 우선으로 두고 `QWEN38_OWNER=1`,
`QWEN38_HELPER=0`, `QWEN38_SKIP_DEVICE_CHECK=1`을 설정했다. 역할 검사는 통과하고 엔진이 모델 로드까지
도달한다. 이어서 세 오류가 순서대로 나온다.

1. `Expected all tensors to be on the same device, but got index is on cuda:0, different from other tensors on
   cuda:1`. `vocab_parallel_embedding.py`의 `embedding`에서 발생한다. 트렁크 이동이 embedding 테이블을
   owner에 두는 반면, 입력 id는 device 0에서 만들어진다.
2. `VocabParallelEmbedding`을 helper로 옮기면 이것은 해소되지만, Triton이 테이블을
   `Pointer argument cannot be accessed from Triton (cpu tensor?)`로 거부한다.
3. embedding forward를 `torch.cuda.device(input_.device)`로 감싸고 테이블을 3090에 두면 Triton은
   통과하지만, 첫 `index_select`에서 같은 장치 분리 오류가 난다. id가 embedding에 들어가기 전에 이미
   cuda:0에 존재하기 때문이다.

장치 분리는 이 호출 지점 이전에 만들어지므로, embedding 안에서는 고칠 수 없다. 이후 엔진을 기본 장치
순서로 되돌렸고, 262,144 구성을 다시 확인했다(`GPU KV cache size: 380,966 tokens`, 로드 220.14초).

## 7. 상류 상태

- vLLM의 SM121/GB10 지원 pull request(#31740)는 2026-11-01에 머지되지 않은 채 닫혔다.
- 설치된 vLLM은 태그 릴리스가 아니라 개발 빌드(`0.1.dev20073`)다.
- Triton의 "cannot be accessed from Triton" 계열 오류는 통합 메모리 자체보다, 텐서의 장치와 다른
  current device에서 커널을 launch할 때 흔히 발생한다. 경로 3의 3단계가 이 해석을 뒷받침한다.
  current device를 맞추자 Triton 오류가 사라지고 앞선 장치 분리가 드러났다.
- 이 포크의 `load_device` 변경은 무효였다. 워커가 device 0 기준으로 할당하는 규칙이 포크 바깥에 있다.

## 8. 자문에게 남기는 열린 질문

1. 이 포크에서 device 0을 옮기지 않고 KV만 통합 메모리 장치에 두면서 가중치는 3090에 남기는 지원
   경로가 있는가? 포크의 owner/helper 분리는 그런 용도로 설계되지 않았다.
2. KV를 옮기는 대신 세 상주 객체 중 하나를 줄여 409,600 윈도우에 도달할 수 있는가? 예를 들어 더 작은
   draft head, 더 좁은 KV dtype, 더 낮은 target 가중치 footprint 같은 방향이다.
3. `0.1.dev20073` 이후 릴리스된 vLLM 태그 중 KV의 device-0 할당 규칙을 바꾸거나, 통합 메모리 device-0을
   받아들이는 버전이 있는가?
4. 윈도우를 262,144로 유지할 경우, 200K를 넘지 않는 워크로드에서 여유가 1.6 GiB 더 큰 204,800 구성이
   측정 가능한 이득을 주는가?
5. 이 환경의 Non-P2P 링크(왕복 33.14µs, 6.2/6.6 GB/s)를 전제로, KV를 두 장치에 나눠 두는 샤딩이
   이득이 있는가? 아니면 호스트 RAM 경유 비용이 항상 손해인가?

## 9. 아티팩트

- `artifacts/recipe/sep-d.json` — 409,600 실행 기록(`status: failed`, `max_model_len: 409600`).
- `artifacts/recipe/sep-d.server.log` — 마지막 409,600 시도의 엔진 로그이며, 6장 경로 3의 장치 분리에서
  멈춘다.
- `artifacts/recipe/sep-d.egpu.log` — 그 시도의 PCIe 링크 상태와 AER 카운터: 3090은 Gen4 x4, 장치에서
  correctable timeout 3건, root port에서 correctable 오류 188건, fatal 오류 0건. `nvbandwidth` 복사
  대역폭도 이 파일에 있다.
- `artifacts/recipe/`에는 17,408 레시피 실행의 sparkDash 봉투(`r5f`, `r5g`)도 있으며,
  `docs/06-results.md`가 인용하는 decode·prefill 행을 담고 있다.
- `artifacts/reproduction/`, `artifacts/longprefill/`, `artifacts/storage/`, `artifacts/verify/`에는
  재현성·롱프리필·저장소 A/B·검증 결과가 들어 있다.

5장에서 인용한 `481,483` admission 줄과 `memory allocation failed` 줄은 첫 409,600 시도에서 엔진이 출력한
것이다. 그 시도의 서버 로그는 엔진을 재기동하기 전에 아카이브되지 않아서, 파일 링크 대신 라이브 실행에서
옮긴 인용으로 남겼다. 아카이브된 `sep-d.server.log`는 이후 시도의 것이며, 더 이른 지점인 embedding 장치
분리에서 멈춘다.

- `docs/06-results.md`의 "204,800 window at block 160", "262,144 window", "Forcing 409,600",
  "Swapping device 0 to the GB10" 절이 같은 수치를 주변 맥락과 함께 담고 있다.
- `docs/07-gb10-solo-vs-egpu.md`의 "Why the GB10 cannot take over KV in the eGPU configuration" 절이
  device-0 교체의 장벽을 담고 있다.
- `docs/01-measurement-method.md`가 측정 규약 전체를 담고 있고, `docs/02-why-prose-is-slower.md`가
  산문 감속의 근거를 담고 있다.
