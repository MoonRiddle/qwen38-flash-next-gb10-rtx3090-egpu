# Analysis 9 — transcription of the bundled PDF

**This file is a cleaned transcription of `GB10-and-RTX3090-decoding-optimization-9.pdf`, which sits
next to it in this directory. The PDF is authoritative.** The transcription preserves the original
Korean text and the author's own `[INFERENCE]` markers; OCR noise, broken table cells and orphaned
formula fragments were removed. Where the PDF states a modelled rather than a measured number, that is
kept as stated and is not presented as measured throughput anywhere in this repository.

An English structural summary, including which of these recipes the engine in `engine/` actually
implements, is in `../docs/03-cold-start-analysis.md`.

---

# Qwen3.8-Flash-Next (A5B-INT4-AutoRound) 짧은 워밍업(Short-Warmup) sparkDash 산문 75~90+ tok/s 즉시 달성 및 코딩·산문 격차의 구조적 해부

## 0. 하네스 /goal 즉시 복붙용 목표 명세

`/goal [JGX10-1: GB10 (SM121, 121.7GiB 통합 메모리) + GPU-Unlocked RTX 3090 eGPU (SM86, 24GiB GDDR6X, Zagitta/gpu-unlock BF16 +70~80% 가속, PCIe Gen4 x4 Non-P2P) + 10Gbps USB Samsung PSSD T7 Shield]` 환경에서 `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound` 단일 체크포인트를 대상으로, MiaAI-Lab의 sparkDash 포크 기준 '짧은 워밍업(첫 1~2회 실행)' 상태에서도 산문(Prose) 디코딩 속도가 즉시 75+ tok/s 이상(목표 80~90+ tok/s) 나오도록 대폭 개선한다.

### 최우선 달성 목표 (Priority KPI)

1. **[1순위 절대 목표]** 짧은 워밍업 상태의 sparkDash 산문(Prose: temp=0.7, thinking=ON) 디코딩 75+ tok/s 즉시 돌파. 여러 번 반복 실행하여 캐시가 달궈진 뒤에만 85 tok/s가 나오는 현상을 해결하고, 엔진 부팅 직후 짧은 워밍업(1회 짧은 실행 또는 무워밍업 첫 본 측정)에서 곧바로 산문 75+ tok/s 이상(목표 80~90+ tok/s)을 달성한다.
2. **[코딩(220 tok/s) 측정 유효성 교차 검증 및 유지]** sparkDash 코딩 측정에서 프리필 캐시 혼입이나 스트림 슬롯 계산 왜곡이 없는지 검증하고, 정당한 순수 디코딩 기준으로 코딩 180~220+ tok/s와 4K 프리필 속도를 유지한다.
3. **[짧은 워밍업 산문 병목 5대 원인(Cold-Start Killers)의 부팅 시점 선제 제거]** 런타임 워밍업에 의존하던
   - ① 행 수(M=1..6)별 조각/Helper CUDA 그래프 및 Triton 커널 지연 컴파일,
   - ② 10Gbps USB T7 Shield의 산문 n-gram PLE 페이지 폴트,
   - ③ 코딩 편향 초기 Hot Expert 미스,
   - ④ 타깃 INT4 lm_head(97.2% 일치율)와 초안 BF16 lm_head 간의 양자화 불일치,
   - ⑤ 600토큰 장문 KV/GDN 메모리 블록 분할

   을 모두 '서버 초기화(Boot-Time Pre-Bake)' 단계로 이동시켜 첫 산문 요청부터 최고 속도가 나오게 만든다.

### 멀티 서브에이전트 동시 가동 규칙

GPU 실행만 공통 잠금(Lock)으로 직렬화하고, 4개의 서브에이전트를 동시에 병렬 가동한다.

- Sub-Agent α: 부팅 시점 전 행 M=1..6 그래프·Triton 선행 캡처 & 타깃-초안 INT4 LM Head 포인터 통합
- Sub-Agent β: USB PLE 부팅 시점 RAM 고정 & 상위 25만 어휘 Fast-Rows 선적재
- Sub-Agent γ: A5B 코딩 편향 힐링 보정용 산문 수락률 패치
- Sub-Agent δ: sparkDash Cold-Run 1 vs Warm Run 3 비교 계측기 및 코딩 220 tok/s 무결성 검증

## 1. 초압축 컨텍스트 및 2대 핵심 의문에 대한 정밀 해답

이 장은 기존 아키텍처 맥락을 요약하고, 두 가지 근본 의문에 대해 실측 데이터와 모델 제작 명세서를 기반으로 답한다.

- 의문 1: 코딩 디코딩이 처음부터 220 tok/s로 빠른 것이 측정 오류인가, 아니면 높은 수락률만으로 설명되는가?
- 의문 2: INT4-AutoRound 퀀트가 코딩 위주로만 전문가를 남겨서 산문이 느린 것인가?

### 1.1 의문 1 — 코딩이 처음부터 빠른 이유

결론: 절반은 수락률과 어휘 국소성에 의한 실제 물리적 현상이고, 절반은 sparkDash 코딩 프롬프트의 특수성 때문에 반드시 점검해야 할 측정 요소다.

추측 디코딩의 속도 공식은 `TPS = E[L] / T_round` 다. 코딩 220 tok/s와 산문 68~85 tok/s가 동일한 하드웨어 파이프라인에서 나온 수치임이 이 식으로 설명된다.

- **Coding (temp=0.0, thinking=OFF):** `<think>` 추론 독백 없이 첫 토큰부터 결정론적 파이썬 코드(def, return, 4칸 들여쓰기 등)를 생성한다. 반복적 함수 패턴에서는 초안기가 제안한 5~6개 토큰이 거의 매번 전수 수락되어 라운드당 확정 토큰 수 `E[L]`이 (보너스 토큰 1개 + 초안 수락 4.5개)에 달한다. 라운드당 실행 시간이 `T_round = 25.0 ms`이면 속도는 위 식으로 계산된다.
- **Warm Prose (temp=0.7, thinking=ON, 충분히 워밍업된 후):** 동일하게 `T_round = 25.0 ms`로 빠르더라도, 고엔트로피 `<think>` 독백과 온도 0.7 샘플링 때문에 라운드당 수락 토큰 수 `E[A]`가 낮아진다.
- **Short-Warmup / Cold Prose (워밍업이 짧은 첫 1~2회 실행):** 수락 길이는 그대로인데, 2장에서 규명할 산문 전용 콜드 페널티(USB SSD의 PLE 페이지 폴트 + 부분 수락 행에 대한 지연 그래프/Triton 컴파일) 때문에 라운드당 평균 시간이 늘어난다. 그 결과 속도는 즉시 75 tok/s 아래로 추락한다.

코딩이 첫 실행부터 콜드 페널티가 없는 이유는 두 가지다.

- **극단적인 어휘·전문가 국소성:** 파이썬 코드는 def, if, for, in, return, `_`, 4칸 공백 등 고작 150~200개의 고유 토큰과 수십 개의 고정된 전문가(Hot Experts)만 반복 호출한다. 따라서 생성 시작 후 처음 15토큰(0.1초) 만에 필요한 모든 PLE n-gram 행과 MoE 전문가가 캐시에 100% 적재된다.
- **단일 그래프 경로 고정:** 코딩은 매 라운드 거의 전수 수락되어 타깃 검증 배치 크기가 항상 일정하다. 즉, 짧은 워밍업 1회 동안 캡처된 단 하나의 CUDA 그래프만 계속 재사용하므로 실행 도중 새로운 그래프 캡처나 Triton 컴파일이 발생하지 않는다.

반드시 확인해야 할 sparkDash 코딩 측정 왜곡 가능성 두 가지:

- 구버전 sparkDash 포크에서는 프리필 캐시 히트 시 프리필 토큰이나 첫 청크가 디코드 슬라이딩 윈도우 계산에 섞이거나, 반복적인 `clamp_00 ... clamp_49` 프롬프트가 비현실적으로 높은 수락률(98%)을 유발한다는 커뮤니티 지적이 있었다.
- 따라서 `first_token_ts` 이후의 순수 생성 토큰만 분자에 들어가고 있는지 검증해야 한다.

### 1.2 의문 2 — 모델 카드 해부

물리적 전문가 삭제는 아니지만 힐링 학습 데이터셋과 INT4 lm_head 양자화 과정에서 코딩/STEM 편향과 산문 불이익이 실제로 발생했음이 확인된다.

| 분석 항목 | 실제 내부 구조 | 코딩 vs 산문 속도에 미치는 영향 |
|---|---|---|
| 전문가 절삭 방식 (top-10 → top-5) | 48층 512개의 물리적 전문가는 모두 그대로 남아 있으나, 토큰당 활성 전문가 수를 10개에서 5개로 절반 절삭함 | 코딩은 소수의 문법 전문가 3~4개만으로 충분하지만, 복합적인 어휘와 뉘앙스를 조합하는 일반 산문은 6~10 순위 전문가들이 잘려 나가면서 원래 모델 대비 로짓 엔트로피가 높아짐 |
| 힐링 데이터셋의 코딩·STEM 편향 | 5개 절삭으로 떨어진 품질(41.8점)을 복구하기 위해 모든 라우팅 전문가는 동결하고 오직 Shared Expert(3,775만 파라미터)만 14,667개 예제로 KL 자기증류 학습함. 힐링 레시피 구성 비율은 수학 30% + 도구 호출 22% + 코드 22%(= STEM/코딩/도구 74%)인 반면 일반 산문은 20% | 모델 카드 공식 결과: 도구 사용과 코딩은 절삭 전과 100% 동일하게 보존(91 vs 91-92)되었으나, 일반 능력은 약 4점을 포기함(49.3 vs 51.8). Shared Expert가 코딩/도구/수학 위주로 힐링되어 코딩에서는 10-expert 교사 모델을 완벽히 모사하지만, 산문에서는 분포 복원력이 약함 |
| 미힐링 MTP 초안기와의 분포 불일치 | 타깃 모델의 Shared Expert는 top-5에 맞춰 재학습(Healed)되었으나, MTP 초안기(draft-k10)는 재학습되지 않은 원래 top-10 가중치 그대로임 | 코딩에서는 힐링된 Shared Expert가 원래 top-10 모델을 완벽히 복원하므로 미힐링 MTP 초안기와 잘 맞지만, 산문에서는 힐링이 덜 된 타깃(top-5)과 옛날 분포를 가진 MTP(top-10) 사이에 미세한 예측 괴리가 발생하여 수락률이 깎임 |
| INT4 lm_head 재양자화 | 디코딩 스텝마다 4번 읽히는 lm_head의 속도를 +8% 높이기 위해 원래 BF16 헤드를 AutoRound INT4 GPTQ-Marlin(group 128)로 강하게 재양자화함. 이 과정에서 BF16 헤드 대비 top-1 일치율이 99.6%(INT8)에서 97.2%(INT4)로 하락(2.8% 불일치) | 코딩(temp=0.0)은 1등 토큰의 확률 격차가 압도적이라 4비트 양자화 노이즈에 흔들리지 않지만, 산문(temp=0.7, thinking=ON)은 후보 단어들의 확률이 엇비슷하여 타깃의 INT4 lm_head와 MTP 초안기의 BF16 lm_head가 서로 다른 1등 토큰을 지목하는 현상(2.8% 불일치 누적)이 발생함 |

> 편집 주석: 위 표의 힐링 데이터 비율(수학 30% / 도구 22% / 코드 22% / 일반 산문 20%)은 PDF 9의 서술이며, Hugging Face 모델 카드에는 카테고리만 열거되고 비율은 기재되어 있지 않다. 1차 출처로 확인되지 않은 주장으로 취급한다.

## 2. 왜 산문은 긴 워밍업 후에만 85 tok/s가 나오고 짧은 워밍업에서는 75 tok/s 미만으로 떨어지는가 (5대 콜드 병목)

MiaAI-Lab의 공식 문서(`docs/tp4.md`)조차 "sparkDash로 벤치마크할 때 부팅 후 처음 두 번의 실행은 버려라(cold Engram row cache, first-run warm-up)"라고 명시하고 있을 만큼, Qwen3.8-Flash-Next 계열은 첫 1~2회 산문 실행에서 심각한 콜드 스타트 페널티를 겪는다.

| 콜드 병목 순위 및 명칭 | 짧은 워밍업 시 발생하는 현상 | 긴 워밍업 후 사라지는 이유 | 첫 실행 산문 속도 손실량 |
|---|---|---|---|
| Cold-1 (최대 주범): 10Gbps USB T7 Shield의 콜드 PLE 페이지 폴트 | `PREWARM=0`, `MADV_RANDOM=1`, `FAST_ROWS=512` 상태에서 첫 산문 요청이 들어오면, `<think>`와 산문 어휘의 n-gram 해시들이 아직 Linux 페이지 캐시에 없는 51.2GB PLE 파일 영역을 찌르며 느린 USB 외장 SSD에서 수백 번의 4KB 랜덤 읽기 스톨(회당 300~500µs)을 일으킴 | 산문을 여러 번 돌리면 자주 쓰이는 자연어 n-gram 페이지들이 GB10의 유휴 RAM 페이지 캐시에 자연스럽게 쌓여서 디스크 I/O가 사라짐 | −8.0 ~ −14.0 tok/s (라운드당 3~5ms 지연) |
| Cold-2: 부분 수락 행 크기에 대한 지연 그래프 캡처 & Triton JIT 컴파일 | 코딩 워밍업이나 짧은 워밍업에서는 전수 수락만 발생하여 그 경로만 워밍업됨. sparkDash 산문(temp=0.7)이 시작되면 0개 수락, 1개 수락, 2개 수락 및 다양한 Cold 행 개수가 처음 등장하면서 측정 도중에 Helper CUDA Graph 캡처와 Triton 커널 JIT 컴파일/오토튜닝이 실행됨 | 여러 번 산문을 돌리고 나면 가능한 모든 행 조합의 그래프와 Triton 커널이 캐시에 다 등록되어 스톨이 사라짐 | −6.0 ~ −12.0 tok/s (첫 실행 중 0.8~2.0초 멈칫거림) |
| Cold-3: 타깃 INT4 lm_head vs 초안 BF16 lm_head 불일치 및 미힐링 MTP의 `<think>` 초반 기각 | 타깃은 AutoRound INT4 lm_head(97.2% 일치)를 쓰고 draft-k10-compact MTP는 원본 BF16 lm_head를 써서 두 헤드의 양자화 노이즈 방향이 다름. 특히 첫 산문 요청의 `<think>` 도입부에서 양자화 불일치 + `DRAFT_SCALE=2.0` 왜곡이 겹치며 첫 토큰 즉시 기각이 집중 발생함 | 문맥이 길어지고 본문(낮은 엔트로피)으로 넘어가면 로짓 격차가 커져 불일치가 완화됨 | −5.0 ~ −9.0 tok/s (라운드당 0.25개 하락) |
| Cold-4: 초기 Hot Expert 캐시의 산문 전문가 미적재 (Cold Expert 폭증) | 부팅 직후 코딩 워밍업만 거친 상태에서는 RTX 3090의 Hot Expert 캐시에 파이썬 문법 전문가만 들어 있거나 산문 `<think>` 추론 전문가가 빠져 있어, 첫 산문 실행 내내 GB10으로 4행 전체가 전송됨 | 산문 실행이 누적되면 산문 전문가들이 캐시나 L2에 워밍업됨 | −4.0 ~ −7.0 tok/s |
| Cold-5: 128 토큰 워밍업 이후 400~600 토큰 생성 시 CUDA Allocator 블록 분할 및 클럭 승압 지연 | 짧은 워밍업(64~128토큰)만 돌리면 KV 캐시와 GDN 상태 버퍼가 128토큰까지만 할당됨. sparkDash 산문이 400~600토큰까지 길어질 때 측정 구간 한복판에서 PyTorch CUDA Allocator 세그먼트를 할당·분할하며 지터를 만듦 | 한 번 600토큰 이상 생성하고 나면 메모리 블록 풀이 확장된 채로 유지됨 | −2.0 ~ −4.0 tok/s |

## 3. 워밍업 없는 첫 실행 75~90+ tok/s (Zero-Warmup Pre-Baking)

핵심 전략은 명확하다. "벤치마크가 시작된 뒤에 워밍업되던 모든 요소를 서버 부팅 시점(run_engine.sh / peer_worker.py 초기화 직후, 벤치마크 타이머가 켜지기 전)에 100% 미리 구워놓는(Boot-Time Pre-Baking) 것"과, 체크포인트의 산문 약점(A5B 미힐링 MTP + INT4 Head 불일치)을 직접 보정하는 것이다.

### 3.1 [레시피 ZW-1] 부팅 시점 전 행 조합 그래프·Triton JIT/그래프 캡처

**원리:** sparkDash 산문 첫 실행이 느린 가장 큰 이유는, 짧은 워밍업 동안 한 번도 나왔던 적 없는 행 개수(예: 초안 1개 수락 시 타깃 검증 행, 또는 Hot Cache 행 압축 후 GB10 helper 실행 행)가 실제 산문 측정 도중에 처음 발생하면서 그때서야 GB10 helper CUDA 그래프를 캡처하거나 Triton 커널을 컴파일하기 때문이다.

**해결책 (Boot-Time Exhaustive Shape Pre-Compilation):**

1. `peer_worker.py`에서 모델 로드 완료 직후(API 서버 오픈 전 또는 벤치마크 시작 전 초기화 함수 내부에서), 가능한 모든 검증 행 크기 `M ∈ {1, 2, 3, 4, 5, 6}`과 GB10 helper 그래프 행 크기 `M_cold ∈ {1, 2, 3, 4, 5, 6}`에 대해 더미 텐서로 1회씩 강제 포워드 및 그래프 캡처를 수행한다.
2. 특히 `temperature = 0.7` 경로의 RejectionSampler 커널(Top-64 Subspace Softmax, `generate_uniform_probs`, Entropy/Block Verify)과 GDN 상태 롤백(`RecoverSSM`) 커널도 부팅 시점에 더미로 2스텝 호출하여 Triton JIT 컴파일을 부팅 단계에서 100% 끝내고 커널 캐시를 동결(Freeze)한다.
3. **소요 시간 및 효과:** 부팅 초기화 시간이 고작 1.5초 늘어나는 대신, 첫 번째 sparkDash 산문 측정에서 발생하는 JIT/그래프 캡처 멈칫거림(1~2초)이 100% 사라져 즉시 +8~12 tok/s가 상승한다 **[INFERENCE]**.

### 3.2 [레시피 ZW-2] 10Gbps USB T7 Shield의 콜드 PLE 스톨 원천 봉쇄

**원리:** MiaAI-Lab이 "콜드 Engram(PLE) 행 캐시 때문에 첫 2회 실행을 버려라"라고 한 문제를 부팅 시점 선적재로 정면 돌파한다.

**해결책 (Boot-Time PLE Pre-Warming):**

1. **RTX 3090 VRAM FAST_ROWS 512배 확장 및 부팅 시점 즉시 채우기:** `VLLM_PLE_MMAP_FAST_ROWS`를 기존 512(80KB)에서 262,144행(41.94 MiB)으로 확장하고, 요청이 들어올 때 채우는(Lazy) 것이 아니라 모델 로더가 초기화될 때 PLE 테이블의 앞쪽 262,144행(가장 빈도가 높은 상위 토큰/n-gram 영역)을 단 한 번의 연속 읽기(42MB 순차 읽기 = USB T7 Shield에서도 0.05초 소요)로 RTX 3090 VRAM에 미리 복사해 둔다.
2. **GB10 유휴 통합 메모리(~58 GiB)를 활용한 백그라운드 순차 선적재 (MADV_WILLNEED):** 혼합 모드에서 GB10은 62.8 GB만 쓰므로 약 58 GiB의 RAM이 비어 있다. 현재 설정된 `VLLM_PLE_MMAP_MADV_RANDOM=1`은 OS의 페이지 프리페치를 막아 콜드 산문에서 4KB 단위로 USB 버스를 찌르게 만든다. 부팅 시점에 PLE 파일의 핵심 구간(40~45 GiB)에 대해 `posix_fadvise(..., POSIX_FADV_WILLNEED)` 또는 `vmtouch -t`로 순차 적재를 걸어두고, Layer 0 진입 전 비동기 스트림에서 PLE 조회를 미리 발사(Off-Critical-Path Prefetch)하면 첫 번째 산문 요청의 첫 번째 토큰부터 USB 페이지 폴트가 0건이 된다.

### 3.3 [레시피 ZW-3] 타깃-초안 INT4 lm_head 포인터 공유(Zero-Drift) + 미힐링 MTP 은닉 보정

**원리:** A5B-INT4는 타깃 lm_head를 INT4 GPTQ-Marlin(BF16 대비 97.2% 일치)으로 재양자화했고, draft-k10-compact MTP는 원래의 BF16 lm_head(1.27 GB)와 미힐링 top-10 가중치를 사용한다. 타깃은 INT4 lm_head로 검증하는데 초안기가 BF16 lm_head로 초안을 뽑으면, 두 헤드 사이의 2.8% 양자화 불일치 때문에 초안기가 맞는 토큰을 뽑고도 타깃에게 기각당할 뿐만 아니라, 초안기가 매 스텝 무거운 1.27 GB BF16 헤드를 읽느라 라운드당 3~4ms를 낭비한다.

**해결책 (Shared INT4 LM Head Pointer & MTP top-k=6):**

1. 타깃과 초안기의 `lm_head` 모듈을 RTX 3090에 이미 로드된 타깃의 INT4 GPTQ-Marlin lm_head 단일 인스턴스로 100% 공유(Pointer Aliasing)시킨다 **[INFERENCE]**.
2. 이 한 줄 수정이 일으키는 3중 효과:
   - **양자화 오차 동조화로 산문 수락률 상승:** 초안기와 타깃 검증기가 1비트까지 똑같은 INT4 lm_head 가중치로 로짓을 계산하므로, 2.8%의 헤드 양자화 불일치로 인한 억울한 산문 기각이 0%로 사라진다.
   - **초안 스텝 속도 3배 가속:** 초안기가 매 스텝 읽는 헤드 크기가 1.27 GB(BF16)에서 ~0.32 GB(INT4 Marlin)로 75% 줄어들어 라운드당 2.5~3.5ms가 즉시 단축된다.
   - **RTX 3090 VRAM 1.27 GiB 공짜 해방:** 중복 로드되던 초안기용 BF16 lm_head가 사라져 1.27 GiB의 여유 VRAM이 즉시 추가로 생긴다. 이 공간으로 전체 1개 층의 512개 Expert 전체(1.21 GiB)를 RTX 3090에 통째로 더 상주시킬 수 있다.
3. **미힐링 MTP와 힐링된 타깃 Shared Expert 간의 은닉 분포 정렬:** 타깃(top-5 + 힐링된 Shared Expert)이 출력하여 MTP(top-10 미힐링)로 넘기는 폭 10,240 은닉 상태(`get_mtp_target_hidden_states()`)의 `pre_fc_norm_hidden` 입력에 미세 스케일 보정 계수를 적용하고, MTP 라우터의 top-k를 10에서 6(또는 5, INT4 Marlin 변환)으로 조율하여 산문 `<think>` 구간의 1~2번째 위치 수락률을 극대화한다.

### 3.4 [레시피 ZW-4] 부팅 시점 정적 산문 우선(Prose-First) Hot Expert 맵 고정 적재

**원리:** 워밍업이 짧을 때 산문이 느린 또 다른 이유는, 초기 Hot Expert 캐시가 비어 있거나 코딩 프롬프트 기준으로 초기화되어 있어 첫 산문 요청의 `<think>` 추론 전문가들과 서술형 전문가들이 전부 GB10(Cold)으로 빠지기 때문이다.

**해결책:**

- 산문에서 자주 쓰이는 전문가 ID 목록을 런타임에 수집하지 말고, 오프라인에서 sparkDash 산문(thinking + 서술 본문) 70% + 코딩 30% 가중치로 추출한 48층 정적 전문가 ID 테이블(`static_prose_hot_map.pt`)을 파일로 저장해 두었다가 모델 부팅 시(`t = 0`) 즉시 RTX 3090에 고정 적재한다.
- 이렇게 하면 워밍업이 0초여도 첫 번째 산문 요청의 첫 토큰부터 Hot Expert 적중률 65~72%가 즉시 발휘된다.

### 3.5 [레시피 ZW-5] temp=0.7 산문 전용 무왜곡 Top-64 부분공간 기각 샘플러 + `<think>` 구간 적응형 검증 + BF16 KV Cache

**해결책:**

1. **`VLLM_MTP_DRAFT_SCALE` 분리:** temp=0.0(코딩)에서는 기존 스케일을 유지하되, temp=0.7(산문)에서 rejection_sampler에 들어가는 `draft_probs`는 반드시 `SCALE=1.0`(Unscaled) 로짓에 타깃과 동일한 온도 `T=0.7`을 적용하여 계산하고, 248,320 전체 어휘 대신 타깃 상위 64개(`topk(64)`) 부분공간에서만 소프트맥스를 수행해 스텝당 샘플러 지연을 0.05ms 이하로 줄인다.
2. **`<think>` 구간 유연 수락 (Block / Entropy Verify):** sparkDash 산문의 초반 200~300토큰을 차지하는 `<think>` 독백 구간에서는 Entropy Verify(`posterior_threshold=0.07`, `alpha=0.3`) 또는 Block Verify를 적용해 고엔트로피 추론 접속사("Wait,", "Let's" 등)에서 발생하는 첫 위치 즉시 기각을 방지한다.
3. **BF16 KV Cache & BF16 GDN State (RecoverSSM):** C=1에서는 메모리를 거의 먹지 않는(4K 기준 120MiB) `kv_cache_dtype: bfloat16`과 `MAMBA_SSM_CACHE_DTYPE=bfloat16`을 사용하여 산문의 깊은 초안 위치 수락률을 2배로 유지하고 기각 시 GDN 상태를 포인터 스왑만으로 즉시 복구한다.

### 3.6 [레시피 ZW-6] 부팅 시점 600-토큰 KV/GDN 메모리 풀 선할당(Pre-Touch) 및 클럭 락(Persistence Mode)

**원리:** 워밍업이 64~128토큰으로 짧으면, sparkDash 산문이 129번째 토큰부터 600번째 토큰까지 생성해 나가는 도중에 PyTorch CUDA Caching Allocator가 새로운 메모리 블록을 쪼개거나 할당하며 지터를 발생시킨다.

**해결책:**

- 환경 변수에 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`를 설정하고, 부팅 시점 더미 스윕(ZW-1) 단계에서 길이 1,024토큰짜리 더미 KV/GDN 슬롯을 한 번 할당 후 해제하여 PyTorch 메모리 풀과 GPU 클럭(P0 상태)을 벤치마크 시작 전에 완전히 달궈 놓는다.

## 4. 기대 효과 종합: 짧은 워밍업 산문 68.5 → 86.0+ tok/s 도약 시뮬레이션 **[INFERENCE]**

위 6대 레시피(ZW-1 ~ ZW-6)를 적용했을 때, 워밍업이 거의 없는 첫 번째 sparkDash 산문 실행(Cold / Short-Warmup Run 1)의 라운드 시간과 수락 토큰 수 변화는 다음과 같다.

| 단계 및 적용 레시피 | 짧은 워밍업 산문 T_round (ms) | 짧은 워밍업 산문 확정 토큰 E[L] (개) | 짧은 워밍업 산문 속도 (Run 1, tok/s) | 충분히 워밍업된 산문 속도 (Run 5+, tok/s) | 코딩 속도 (Run 1, tok/s) |
|---|---:|---:|---:|---:|---:|
| 현재 상태 (긴 워밍업 전 콜드 산문) | 31.0 ms (USB 폴트 + JIT 스톨 + BF16 헤드) | 2.125 | 68.5 (< 75 미달) | 85.0 | ~220.0 |
| + ZW-1 & ZW-6 (부팅 시점 전수 그래프/Triton 캡처 + 600토큰 풀 선할당) | 27.2 ms (−3.8ms: 측정 중 JIT/할당 스톨 소멸) | 2.125 | 78.1 (1차 목표 75+ 즉시 돌파) | 85.5 | 222.0 |
| + ZW-2 & ZW-4 (부팅 시점 PLE 256K VRAM 선적재 + GB10 RAM 순차 고정 + 정적 산문 Hot Map) | 24.5 ms (−2.7ms: 첫 실행부터 USB 폴트 0건 & Hot 적중) | 2.150 | 87.8 | 88.5 | 225.0 |
| + ZW-3 & ZW-5 (타깃-초안 INT4 lm_head 포인터 공유 + 1.27GiB 추가 층 상주 + 미힐링 MTP 스케일/검증 보정) | 22.2 ms (−2.3ms: 초안 헤드 75% 경량화 & 1개 층 왕복 제거) | 2.350 (+0.20개: 헤드 양자화 일치로 기각 감소) | 105.9 (첫 실행부터 100+ 돌파) | 106.5 | 238.0+ |

> 이 표의 네 행은 이 머신에서 측정된 값이 아니라 PDF 9의 모델이다.

## 5. 짧은 워밍업 sparkDash 산문 75~90+ tok/s 최우선 달성 & 코딩 검증 마스터 지침서

아래는 PDF 9가 실행 에이전트에게 전달한 지시서의 요지다.

- 절대 1순위 목표는 sparkDash 벤치마크에서 워밍업이 짧은 상태(부팅 직후 첫 번째 실행 또는 기본 1회 짧은 워밍업 직후)에서도 산문(Prose: temp=0.7, thinking=ON) 디코딩 속도가 즉시 75+ tok/s 이상(목표 85~95+ tok/s) 나오게 만드는 것이다.
- 왜 코딩(220 tok/s)은 처음부터 빠른가: 코딩(temp=0.0, thinking=OFF)은 `<think>` 독백이 없고, 약 200개의 반복적인 파이썬 예약어/공백만 써서 시작 후 0.1초 만에 모든 PLE 행과 Hot Expert가 캐시에 다 올라오며, 매 라운드 전수 수락되어 단 하나의 CUDA 그래프만 계속 타기 때문이다. 반면 산문(temp=0.7, thinking=ON)은 첫 실행 때 ① 부분 수락(0/1/2개 및 Cold 행)이 발생하면서 측정 도중에 새로운 CUDA 그래프 캡처와 Triton 커널 JIT 컴파일이 터지고, ② 새로운 자연어 어휘가 10Gbps USB T7 Shield의 51.2GB PLE 테이블에서 랜덤 페이지 폴트를 일으키며, ③ 초기 Hot Cache에 산문 전문가가 없어서 첫 실행이 느렸던 것이다.
- azampatti/INT4-AutoRound가 코딩 전문가만 남겨서인가: 모델 카드를 확인해 보니 512개 전문가는 그대로 두고 top-10 → top-5로 절삭했는데, 그 손실을 메우기 위해 Shared Expert(37.75M)를 힐링 학습시킬 때 사용한 데이터셋의 ~74%가 코딩·수학·도구 호출이고 일반 산문은 ~20%뿐이어서 일반 산문 능력이 4점 빠졌고, 결정적으로 타깃의 lm_head를 INT4 GPTQ-Marlin(BF16 대비 97.2% 일치)으로 재양자화해 놓고 MTP 초안기(draft-k10)는 미힐링 top-10 + BF16 lm_head(1.27GB) 그대로 두었기 때문에, 고엔트로피 산문에서 타깃(INT4 헤드)과 초안기(BF16 헤드)가 2.8% 확률로 서로 다른 토큰을 지목하며 기각을 유발하고 있었다.

### 서브에이전트 지시 요약

- **Sub-Agent α [콜드 스타트 킬러 1·2호 제거]:** 부팅 시점 `M = 1..6` 및 `M_cold = 1..6` 전수 더미 스윕과 temp=0.7 RejectionSampler 커널 선행 실행(ZW-1), 600토큰 분량 더미 KV/GDN 버퍼 선할당(ZW-6), 타깃-초안 INT4 lm_head 포인터 공유(ZW-3).
- **Sub-Agent β [콜드 스타트 킬러 3·4호 제거]:** `VLLM_PLE_MMAP_FAST_ROWS=262,144` VRAM 선적재 + GB10 유휴 RAM 순차 프리로드(ZW-2, `MADV_RANDOM` 해제), 정적 산문 70% + 코딩 30% Hot Expert 테이블 부팅 시점 바인딩(ZW-4).
- **Sub-Agent γ [A5B 코딩 편향 힐링 보정용 산문 수락률 부스터]:** temp=0.7 무왜곡 Top-64 부분공간 기각 샘플링, `<think>` 구간 유연 검증(Block/Entropy Verify) 및 `</think>` 이후 자동 가속, 미힐링 MTP 입력 은닉 스케일 보정 및 MTP top-k=6, BF16 KV Cache & BF16 GDN State(ZW-5).
- **Sub-Agent δ [계측기 및 코딩 220 tok/s 무결성 감사]:** 코딩 220 tok/s 측정 무결성 팩트체크(프리필 캐시 히트 토큰이 디코드 토큰 수에 섞이지 않았는지, `T_round`, `E[L]`, 총 생성 토큰 수와 출력 텍스트 검증), 엄격한 Short-Warmup(Cold Run 1) 측정 프로토콜(서버 Ready 직후 1회의 짧은 워밍업 또는 무워밍업 첫 산문 호출의 tok/s를 최우선 합격 기준 ≥75 tok/s로 기록).

## 참고 자료 (PDF 9 기준)

1. `external-ai-advisory-20261029.md`
2. `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound` — https://huggingface.co/azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound
3. (unknown_url)
4. `MiaAI-Lab/sparkDash` — https://github.com/MiaAI-Lab/sparkDash
5. `loud1990/GLM-5.3-Flash-NVFP4-MXFP8-2x-DGX-Sparks-GitHub` — https://github.com/loud1990/GLM-5.3-Flash-NVFP4-MXFP8-2x-DGX-Sparks
6. Qwen3.8 Flash Next on one DGX Spark with TensorFold — https://mia-ai.net/models/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
7. `MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold` — https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
8. Anyone happy with just single DGX Spark? — r/LocalLLM
9. Read a speculative decoding acceptance histogram — https://fanout.sh/blog/speculative-decoding-acceptance-length-histogram
10. `DeepSeek-v4.1-Flash-DGX-Sparks/docs/tp4.md` — https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/blob/main/docs/tp4.md
11. New Deepseek 4.1 flash recipe — DGX Spark / GB10
12. sparkDash — Mia's AI Lab — https://mia-ai.net/lab/projects/sparkdash
13. `azampatti/GLM-5.3-Flash-AZ` — https://huggingface.co/azampatti/GLM-5.3-Flash-AZ
14. `myllmbox/qwen38-flash-next-recipe` — https://github.com/myllmbox/qwen38-flash-next-recipe
15. Multi-Agent Workload: Qwen3.8-Flash-Next NVFP4 on 2× DGX Spark
16. `MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark` — https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark
17. rejection_sampler — vLLM Documentation — https://docs.vllm.ai/en/stable/api/vllm/v1/sample/rejection_sampler/
18. Recipe + field notes: Qwen3.8-Flash-Next NVFP4 on 2x DGX Spark — https://github.com/beastllama/dgx-spark-qwen38-flash-next-recipe
19. I measured five ways to serve Qwen3.8 Flash Next on one DGX Spark — r/LocalLLM
20. Strata LLM — https://stratallm.org/
21. Speculative Decoding Guide — vLLM Ascend
