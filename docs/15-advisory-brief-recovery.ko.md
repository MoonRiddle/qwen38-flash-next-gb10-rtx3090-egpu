# 15. 단일 스트림 디코딩 회복: 자문용 보고서

이 문서는 사전 맥락이 없는 외부 AI나 엔지니어를 위해 작성했다. 이 문서만 읽으면 하드웨어, 모델, 측정 계약,
무엇을 측정했고 무엇이 어떤 정확한 오류로 실패했는지, 그리고 지금 작업이 어디에서 막혀 있는지 이해할 수 있다.

- 0절은 배경이다.
- 1-6절은 측정 기록이다.
- 7-9절은 열린 질문, 남은 후보, 산출물이다.

아래 모든 수치는 sparkDash native 산출물과 `docs/14-pdf-recovery-ladder.md`에서 옮겼다. 모델링한 값은 없다.
확인되지 않은 주장은 그 자리에서 표시했다.

---

## 0. 배경

### 0.1 한 문장 요약

GB10 통합 메모리 장치와 RTX 3090 eGPU에서 `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`를
340,000토큰 윈도우, 동시성 2로 서빙하면서, 동시성 2가 잃게 만든 단일 스트림 디코드 속도를 회복하는 실험이다.

### 0.2 두 목표

|작업|목표|현재 최고|격차|
|---|---:|---:|---:|
|code 디코드, C=1|180 tok/s|122.96|57.04|
|prose 디코드, C=1|70 tok/s|54.39|15.61|

code와 prose를 따로 측정하는 이유는 이 레시피에서 두 작업이 다르게 동작하기 때문이다. code 경로는 MTP 깊이를
최대로 쓰고, prose 경로의 최고 구성은 다른 깊이를 쓴다.

### 0.3 하드웨어

|장치|특성|역할|
|---|---|---|
|GB10 (SM121)|CPU와 GPU가 121.7GiB 통합 풀 공유, 246GB/s|모델 가중치, PLE 페이지 캐시|
|RTX 3090 (SM86)|별도 24GiB GDDR6X, 833GB/s, eGPU|핫 엑스퍼트 캐시, KV 캐시. 병목|

링크는 PCIe Gen4 x4 Non-P2P라 전송이 호스트 RAM을 거친다. 실측 왕복은 20KB에서 33.14us이고, 복사 대역폭은
호스트에서 장치로 6.20GB/s, 장치에서 호스트로 6.59GB/s다. 3090에는 BF16 연산을 기본보다 70-80퍼센트 빠르게
만드는 소프트웨어 언락이 걸려 있다. 이 레시피는 그 상태를 만들지 않았고 바꾸지 않는다.

### 0.4 모델

|항목|값|
|---|---|
|전체 파라미터|125B|
|토큰당 활성 파라미터|4.8B|
|구조|MoE, 512개 라우티드 엑스퍼트 전부 유지, 토큰당 5개 선택|
|양자화|INT4 AutoRound (W4A16)|
|lm_head|int4 GPTQ-Marlin g128, BF16 대비 top-1 일치 97.2퍼센트|
|드래프트 헤드|BF16 top-10|
|PLE 테이블|51GiB int4, USB에서 mmap, Layer 1에서만 사용|

PLE 테이블은 한 장치에 가중치와 함께 들어가지 않는다. 20Gbps USB NVMe에 있고 mmap으로 읽으며, 10Gbps 사본도
있다.

### 0.5 측정 계약

모든 속도는 MiaAI-Lab sparkDash native 러너의 커밋 `7c2b9638b025cd39fb43f95c225de62980f130ee`에서 나온다.
이 작업이 지키는 규칙은 다음과 같다.

- 실행 상태가 `completed`이고 `failures`가 비어 있을 때만 속도를 보고한다.
- 케이스마다 3회 실행해 중앙값을 보고한다. 순간 최고치와 동시 요청 합산 속도는 목표 달성 근거로 쓰지 않는다.
- 엔진을 한 번 띄우고 모든 케이스를 그 엔진 하나에 돌린다. 케이스마다 재기동하지 않는다.
- 예열, 반복, EOS 처리는 원본 수집기를 그대로 따른다.

### 0.6 서빙 구성

엔진은 340,000토큰 윈도우를 `--max-num-seqs 2`로 서빙한다. 이 윈도우용으로 두 개의 상주 뷰를 만든다.

|뷰|경로|비고|
|---|---|---|
|target-340k|`/work/target-340k`|YaRN factor 1.2969970703125|
|draft-340k|`/work/draft-340k`|같은 factor, 동기화됨|

`max_position_embeddings`는 340000, `rope_type`은 yarn, `original_max_position_embeddings`는 262144,
`mrope_section [11,11,10]`, `partial_rotary_factor 0.25`, `rope_theta 10000000`이다.

KV 캐시는 `fp8_e5m2`, `block_size 160`, `kv-mib 15360`이다. admission은 340,000토큰 요청을 동시성 2로
받을 때 `677,901 tokens, 1.99x`를 보고한다.

---

## 1. 동시성 2가 잃게 만든 것

이 카드 조합에서 340K 윈도우를 서빙하려면 동시성 2가 필요했다. 부작용은 단일 스트림 디코드 하락이었다. 회복
작업 전의 기준선은 code C=1 114.11, prose C=1 45.85였다. 목표는 180과 70이므로, 회복 작업은 이미 목표 아래에서
시작했다.

---

## 2. 6단계 사다리

그 기준선에 대해 단일 변수 6단계를 측정했다. 각 단계는 기대가 아니라 측정값으로 유지하거나 되돌렸다.

|단계|변경|code C=1|prose C=1|admission|판정|
|---|---|---:|---:|---|---|
|기준선|—|114.11|45.85|677,901, 1.99x|참조|
|1|`--kv-cache-dtype-skip-layers 48`|**122.96**|**50.58**|677,901, 1.99x|채택|
|2|`QWEN38_INDEX_SHARE=1`|114.92|53.62|677,901, 1.99x|혼합|
|3|mamba SSM 캐시 bf16|118.27|45.67|—|되돌림|
|4|batch 2048|108.45|52.11|—|되돌림|
|5|340K 뷰, factor 1.2969|107.12|48.75|—|되돌림|
|6|int8 per-token-head KV|—|—|OOM|되돌림|

단계 1이 유일한 순이득이다. code 7.8퍼센트, prose 10.3퍼센트 상승이며 admission 손실이 없다. 단계 2는 code를
prose로 맞바꾸며, 누적 구성의 일부로만 유지한다. 단계 3, 4, 5는 잃는 것이 얻는 것보다 많아 되돌렸다.

단계 6은 따로 적을 가치가 있다. `int8_per_token_head`를 네 개 허용 목록(`qwen38_qsa_v2.py:111` `_fp8` 튜플,
`:128`, `:227`, `:341` `torch.int8`)에 배선했다. 엔진은 기동했지만 admission이 fp8과 같은
`677,901 tokens, 1.99x`를 보고한 뒤 실패했다.

```
memory allocation failed with OOM on device 0 while trying to allocate 73400320 bytes (free: 61014016)
```

스케일 텐서가 넓은 타일이 쓸 메모리를 소비하므로 int8 per-token-head는 이 장치에서 admission을 사지 못한다.
되돌렸다.

---

## 3. 남은 격차의 원인

남은 격차는 어텐션 커널 안의 타일 폭 반감이다.

```
qwen38_ops_qsa_v2.py:926   _cast_kv_tile이 fp8 타일을 FP32를 거쳐 캐스팅한 뒤 커널이 block_n을 반감
```

fp8_e5m2 어텐션 스텝마다 fp8에서 bf16으로 캐스팅할 때 101376바이트 하드웨어 한도보다 넓은 공유 메모리 operand가
필요해 디코드 타일 폭의 약 절반을 잃는다. 이 반감을 제거한 세 번의 시도를 측정했고, 셋 다 손실이었다.

### 3.1 넓은 타일

커널 `qwen38_ops_qsa_wide.py`는 KV 타일을 bf16으로 직접 캐스팅해 반감을 제거한다. 공유 메모리에 맞추는 데
세 번의 시도가 필요했다.

```
OutOfResources: shared memory (Required: 106496, Hardware limit: 101376)
```

첫 원인은 `_qsa_mqa_paged_kernel`의 `STAGES=2`였고 `STAGES=1`로 해결했다. 둘째 원인은
`_qsa_sparse_paged_gqa_splitk_kernel`의 `num_stages=2`(`qwen38_ops_qsa_wide.py:1006`)였고 1로 해결했다.

엔진은 325.5초에 `ready`, `677,901 tokens, 1.99x`에 도달했다. 측정값은 code C=1 중앙 103.92, prose C=1 중앙
49.65였다. 단계 1 대비 code 15.5퍼센트, prose 1.8퍼센트 낮다. `num_stages=1` 파이프라인 손실이 넓은 타일 이득보다
크므로 되돌렸다.

### 3.2 청크 dot 누산

전체 헤드 차원에 걸친 단일 `tl.dot(keys, query)` 대신 `qwen38_ops_qsa_chunk.py`가 `D_CHUNK = 32` 폭의 dot을
`tl.static_range` 루프로 누산하고 `STAGES=2`를 복원한다.

```python
        scores = tl.zeros((BLOCK_N, MAX_N), dtype=tl.float32)
        for d_start in tl.static_range(0, BLOCK_D, D_CHUNK):
            d_idx = d_start + tl.arange(0, D_CHUNK)
            d_valid = d_idx < HEAD_DIM
            k_chunk = tl.load(
                k_cache_ptr
                + safe_physical_page[:, None] * stride_cache_block
                + page_offset[:, None] * stride_cache_token
                + d_idx[None, :] * stride_cache_dim,
                mask=page_valid[:, None] & d_valid[None, :],
                other=0.0,
                eviction_policy="evict_first",
            )
            q_chunk = tl.load(
                q_ptr
                + row * stride_q_row
                + heads[None, :] * stride_q_head
                + d_idx[:, None] * stride_q_dim,
                mask=(heads[None, :] < NUM_HEADS) & d_valid[:, None],
                other=0.0,
            )
            k_chunk = _cast_kv_tile(k_chunk, q_chunk, kc_scale_ptr, KC_QUANT_MODE)
            scores += tl.dot(k_chunk, q_chunk, out_dtype=tl.float32)
        scores = tl.where(heads[None, :] < NUM_HEADS, tl.maximum(scores, 0.0), 0.0)
```

엔진은 376.3초에 `ready`, `677,901 tokens, 1.99x`에 도달해 공유 메모리 상한이 `STAGES=2`를 유지한 채
사라졌다. 측정값은 code C=1 111.75 / 124.73 / 104.22, 중앙 111.75, prose C=1 28.54 / 50.88 / 49.44,
중앙 49.44였다. 단계 1 대비 code 9.1퍼센트, prose 2.3퍼센트 낮다. 청크 dot 4회가 넓은 단일 dot의 이득보다
런치와 누산기 트래픽을 더 쓴다. 되돌렸다.

이것이 가장 유용한 부정 결과다. 공유 메모리 상한이 파이프라인을 포기하지 않고도 제거 가능하다는 것을 증명했지만,
청크별 오버헤드가 여전히 넓은 타일 이득보다 크다.

### 3.3 MTP 깊이

드래프트 깊이를 15에서 12, 10으로 낮췄다. KV admission도 함께 움직였다.

|깊이|code C=1 중앙|prose C=1 중앙|admission|로드 초|
|---:|---:|---:|---|---:|
|15 (단계 1)|**122.96**|50.58|677,901, 1.99x|353.4|
|12|111.83|50.54|732,133, 2.15x|325.3|
|10|107.78|**54.39**|781,768, 2.30x|323.3|

깊이가 줄면 스텝당 드래프트 토큰이 줄어 수락 비율이 낮아지므로 code가 내려간다. prose는 깊이 10에서 단계 1
대비 7.5퍼센트 올랐다. 어느 깊이도 code에서 단계 1을 넘지 못한다.

여기서 구현 함정 하나를 발견했다. 드라이버가 `prose_max_draft_tokens`를 하드코딩한다. 깊이 10과 12는 각각
10과 12로 설정한 드라이버가 필요하고, 불일치하면 `restoration_verified`가 실패한다.

---

## 4. 유지되는 누적 구성

`sep-c2s48.sh`는 code 프로파일이며 측정된 최고 code 구성이다.

```bash
export QWEN38_MODEL_PATH=/work/target-409k-b
export QWEN38_DRAFT_PATH=/work/draft-409k-b
export QWEN38_INDEX_SHARE=0
export QWEN38_MAMBA_DTYPE=
export QWEN38_KV_SKIP_LAYERS=48
export VLLM_MTP_DRAFT_HEAD=int4
export VLLM_MTP_DRAFT_VOCAB=/work/draft-vocab-generic-65536.npy
export VLLM_MTP_DRAFT_VOCAB_CKPT=/work/draft-k10-int4
export VLLM_PLE_MMAP_MADV_RANDOM=0
export VLLM_PLE_MMAP_PREWARM=1
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=32
exec /bin/bash "$BASE/run_engine_hostnet.sh" sep-c2s48 15 NONE \
    --mode serve --state-file /work/sep-c2s48.json --served-model-name qwen38-mia-a5b \
    --protocol-profile final --max-model-len 340000 --max-num-seqs 2 --kv-mib 15360 \
    --kv-cache-dtype fp8_e5m2 --block-size 160 --max-num-batched-tokens 1024 \
    --draft-sample-method probabilistic --rejection-sampling standard \
    --reasoning-parser qwen3 --server-start-timeout 3600 \
    --prefill-sizes 4096 16384 32768 200000 261000
```

`sep-c2m10.sh`는 `sep-c2m10 10 NONE`만 다르고 prose 프로파일이다.

---

## 5. 확인하고 배제한 것

측정 전에 의심 원인 여럿을 소스에서 확인해 배제했다.

|의심|발견|
|---|---|
|`qsa_mqa_paged` KC_QUANT_MODE=0 폴백|이미 수정됨, `qwen38_qsa_v2.py:616`, `:625`, `:689`|
|타깃·드래프트 YaRN 불일치|두 뷰 모두 factor 1.5625, 이미 동기화|
|`index_share_for_mtp_iteration` 미배선|배선됨, 다만 code 프로파일은 `QWEN38_INDEX_SHARE=0`|
|드래프트 임베딩 CPU 실행|반증됨, GPU arm 105.85가 CPU arm 139.91보다 낮음|
|`--kv-cache-dtype-skip-layers` 미지원|지원됨, `attention.py:294-310`|
|`--mamba-ssm-cache-dtype` 미지원|지원됨, `config.py:801`|

---

## 6. 현재 위치

측정된 최고 구성은 code C=1 122.96, prose C=1 54.39이며 두 작업에 다른 프로파일을 쓴다. 두 목표 모두 미달이다.
남은 격차는 타일 폭 반감이고, 이를 제거한 세 번의 시도는 모두 반감 비용보다 작은 무언가를 맞바꿔 손실이었다.

|시도|맞바꾼 것|손실|
|---|---|---|
|넓은 타일|소프트웨어 파이프라인(`num_stages=1`)|code -15.5퍼센트|
|청크 dot|청크별 런치와 누산|code -9.1퍼센트|
|짧은 드래프트|수락 토큰 비율|깊이 12에서 code -9.1퍼센트|

---

## 7. 열린 질문

1. `block_n`을 넓게 유지하면서 `STAGES=2`를 동시에 지키고 dot을 청크로 나누지 않는 방법이 있는가? 청크 결과는
   공유 메모리 상한이 제거 가능함을 보여주므로, 질문은 operand를 넓은 공유 메모리 타일로 실체화하지 않고 레지스터에서
   디코드할 수 있는가다. PR #60602가 레지스터 내 바이트 디코드를 제안하며, 이 포크에서는 아직 시험하지 않았다.
2. `_cast_kv_tile`의 FP32 중간값이 꼭 필요한가? 캐스팅이 fp8에서 bf16으로 레지스터에서 바로 갈 수 있다면 공유
   메모리 operand가 파이프라인 손실 없이 반감될 수 있다.
3. 깊이 15의 수락 토큰 비율이 이미 천장 근처인가, 아니면 드래프트 헤드 어휘
   (`draft-vocab-generic-65536.npy`, `draft-k10-int4`)를 admission 손실 없이 넓힐 수 있는가?
4. code와 prose 경로가 깊이를 요청 단위로 받아 하나의 엔진을 공유할 수 있는가?

---

## 8. 우선순위 후보

1. **레지스터 내 e5m2 바이트 디코드(PR #60602 방향).** 공유 메모리 operand를 완전히 우회해 fp8 바이트를
   레지스터에서 디코드하고 넓은 `tl.dot` operand 없이 누산한다. 파이프라인이나 dot 횟수를 맞바꾸지 않고 반감을
   다루는 유일한 후보다.
2. **fp8에서 bf16 직접 캐스팅.** `_cast_kv_tile`의 FP32 중간값을 제거하고, 공유 메모리 operand가 `STAGES=2`에서
   `block_n`을 넓게 유지할 만큼 줄어드는지 측정한다.
3. **드래프트 어휘 확장.** 더 넓은 드래프트 어휘가 admission 손실 없이 깊이 15의 수락 토큰 비율을 올리는지 시험한다.
4. **요청 단위 깊이.** 깊이를 프로파일이 아니라 요청에 배선해 code와 prose가 한 엔진을 공유하게 한다.

청크 누산은 소진된 경로이므로 다시 시도하지 않는다.

---

## 9. 산출물

원격 작업 디렉터리는 `/home/user/OMP_Workspace/qwen38-gb10-3090-20261029`다.

|산출물|내용|
|---|---|
|`sep-c2.json`|기준선 상태|
|`sep-c2s48.json`|단계 1 상태, code 프로파일|
|`sep-c2is.json`|단계 2 상태|
|`sep-c2mb.json`|단계 3 상태|
|`sep-c2b2n.json`|단계 4 상태|
|`sep-c2ys.json`|단계 5 상태|
|`sep-c2wide.json`|넓은 타일 상태|
|`sep-c2chunk.json`|청크 dot 상태|
|`sep-c2m12.json`|깊이 12 상태|
|`sep-c2m10.json`|깊이 10 상태, prose 프로파일|
|`a-<prefix>-execution.json`|prefix별 sparkDash 실행 기록|
|`a-<prefix>-k6-open-<case>-0N.json`|실행별 native 산출물|

사다리 서술은 `docs/14-pdf-recovery-ladder.md`, 한국어판은 `docs/14-pdf-recovery-ladder.ko.md`다. 측정 방법은
`docs/01-measurement-method.md`다.
