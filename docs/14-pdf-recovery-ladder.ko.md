# 14. PDF 복구 사다리: 340K 동시성 2에서의 6개 단일 변수 단계

이 문서는 외부 PDF 피드백의 복구 사다리를 `sep-c2` 구성에서 실행한 기록이다. 각 단계는 누적
최선 구성에서 변수 하나만 바꾸고, 고정된 MiaAI-Lab sparkDash native 러너로 측정한 뒤 중앙값
기준으로 채택 또는 되돌림을 결정했다. 목표는 C=1 code 180 tok/s, C=1 prose 70 tok/s였다.

## 1. 기준선

`sep-c2.sh`를 변경 없이 다시 띄웠다. 상태 파일은 371.3초에 `ready`를 기록했다. 각 케이스당
디코딩 3회, 프리필 1회를 실행했다.

|케이스|1회|2회|3회|중앙|
|---|---:|---:|---:|---:|
|code C=1|114.11|108.13|115.32|114.11|
|prose C=1|34.74|45.85|59.28|45.85|
|code C=2 요청당|86.79|83.83|90.60|86.79|
|prose C=2 요청당|37.70|41.87|40.16|40.16|

프리필 작업은 세 컨텍스트 크기 모두에서 실패했다. 32,768토큰 프리필 중 엔진 코어가 종료된
것으로, `docs/12-decoding-regression.md`에 기록된 인덱서 워크스페이스 상한과 같다. 이 실패는
디코딩 비교에 영향을 주지 않는다.

## 2. PDF 주장과 포크의 실제 상태

|PDF 주장|코드 위치|검증 결과|
|---|---|---|
|`_cast_kv_tile`이 FP32 타일을 만들어 `block_n`을 반감|`qwen38_ops_qsa_v2.py:926`|확인. `:922` `if _kv_mode != 0:`가 fp8_e5m2에서 참이라 64를 32로 반감|
|`qsa_mqa_paged`가 e5m2에서 `KC_QUANT_MODE=0` 폴백|`qwen38_ops_qsa_v2.py:616`, `:625`, `:689`|이미 수정됨. `:689`가 e4m3와 e5m2 모두 1을 반환|
|타깃-드래프트 YaRN 불일치|`target-409k-b/config.json`, `draft-409k-b/config.json`|이미 동기화됨. 둘 다 `text_config.max_position_embeddings 409600`, `rope_type yarn`, `factor 1.5625`|
|`index_share_for_mtp_iteration` 미적용|`speculator.py:27-58`, `llm_base_proposer.py:580-611`, `sep-c2.sh`|배선됨, 꺼져 있음. `QWEN38_INDEX_SHARE=0`이 `set_skip_topk` 경로를 휴면 상태로 유지|
|드래프트 `embed_tokens` CPU 오프로드가 원인|`peer_worker.py:762`, `qwen38_mtp.py:262-265`|반증. GPU arm이 105.85로 CPU arm 139.91보다 느렸다|
|`--kv-cache-dtype-skip-layers` 미지원|`attention.py:294-310`|지원. 목록에 있는 레이어 인덱스면 `kv_cache_dtype = "auto"`|
|`--mamba-ssm-cache-dtype` 미지원|`config.py:801`|지원|
|int8 mode 2가 FP32 타일 없이 동작|`triton_unified_attention.py:39-58`|확인. mode 2는 쿼리 dtype으로 바로 캐스팅|

PDF의 다섯 원인 중 세 개는 이미 처리됐거나 반증됐다. 남은 두 개는 `block_n` 반감과 휴면
상태의 인덱스 공유였다.

## 3. 단계별 결과

각 단계는 C=1과 C=2에서 code 3회, prose 3회, 프리필 1회를 실행했다.

|단계|변경|code C=1 중앙|prose C=1 중앙|엔진 로드|결정|
|---|---|---:|---:|---:|---|
|기준선|없음|114.11|45.85|371.3초|기준|
|1|`--kv-cache-dtype-skip-layers 48`|**122.96**|**50.58**|353.4초|채택|
|2|`QWEN38_INDEX_SHARE=1`|114.92|53.62|337.4초|채택, code 하락|
|3|`--mamba-ssm-cache-dtype bfloat16`|118.27|45.67|299.2초|되돌림, prose 하락|
|4|`--max-num-batched-tokens 2048`|108.45|52.11|309.3초|되돌림, code 하락|
|5|340K 뷰, YaRN factor 1.2969|107.12|48.75|314.4초|되돌림, 둘 다 하락|
|6|`--kv-cache-dtype int8_per_token_head`|—|—|—|되돌림, OOM|

### 단계 1

드래프트 layer-48 KV 캐시를 bf16으로 유지한 것만으로 code 중앙이 기준선 대비 7.8퍼센트,
prose 중앙이 10.3퍼센트 올랐다. admission은 340,000토큰 요청에 677,901토큰, 1.99x로 그대로였고,
bf16 레이어가 측정 가능한 풀 비용을 만들지 않았다.

### 단계 2

`index_share_for_mtp_iteration`을 켜자 prose 중앙이 3.04 tok/s 올랐지만 code 중앙이 8.04 tok/s
내렸다. 단일 최고 code 실행은 131.58로 전체 단계 중 가장 높았으나 중앙이 따라오지 않았다. 누적
구성은 여전히 기준선보다 code가 높고 prose 상승이 명확해 채택했다.

### 단계 3

bfloat16 GDN 상태 캐시는 단계 2 대비 code를 2.9퍼센트 올리고 prose를 14.8퍼센트 내렸다. 계획의
contingency는 prose 회귀를 상태 정밀도 손실로 보고 되돌리라고 했으므로 되돌렸다.

### 단계 4

mamba 변경 없이 2048토큰 프리필 청크를 복원하자 단계 2 대비 code가 5.6퍼센트, prose가 2.8퍼센트
내렸다. OOM은 없었지만 처리량이 떨어져 1024 청크를 유지했다.

### 단계 5

`factor 1.5625`의 409,600토큰 뷰를 `factor 1.2969970703125`의 340,000토큰 뷰로 교체했다. code와
prose가 단계 2 대비 6.8퍼센트, 9.1퍼센트 내렸다. 더 넓은 정적 YaRN factor가 디코딩 하락의
원인이 아니었고, 409,600 뷰를 유지했다.

### 단계 6

int8은 엔진 기동 전에 네 개의 허용 목록에 걸렸다.

1. `--kv-cache-dtype int8`은 유효한 CLI 값이 아니다. 허용 이름은 `int8_per_token_head`다.
2. `qwen38_qsa_v2.py:128`과 `:227`이 dtype 문자열을 거부했다.
3. `qwen38_qsa_v2.py:341`이 `torch.int8` 저장 타입을 거부했다.
4. `:111`의 `_fp8` 튜플이 부모 `FlashAttentionImpl` 초기화 전에 dtype을 중립화하지 않아
   `FlashAttention does not support int8_per_token_head kv-cache on this device`가 발생했다.

네 곳을 모두 넓힌 뒤 엔진은 `GPU KV cache size: 677,901 tokens, 1.99x`에 도달했고 곧바로
`memory allocation failed with OOM on device 0 while trying to allocate 73400320 bytes
(free: 61014016)`를 냈다. per-token-head 스케일 텐서 비용이 admission 증가분보다 커서 되돌렸고,
`fp8_e5m2`와 layer-48 스킵 조합이 최종 구성으로 남았다.

## 4. 최종 구성

```
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

`run_engine_hostnet.sh`는 이제 `QWEN38_QSA_OPS_FILE`을 인식해 공유 마운트 블록을 고치지 않고도
대체 ops 파일을 마운트할 수 있다.

## 5. 넓은 타일 시도

`block_n` 반감은 `_cast_kv_tile`이 fp8 타일을 스케일 곱 전에 fp32로 승격하기 때문에 존재한다.
쿼리 dtype으로 바로 캐스팅하는 로컬 대체 함수를 `QWEN38_QSA_OPS_FILE`로 마운트하고 반감을
제거해 `block_n`을 64로 유지했다.

공유 메모리 상한이 세 번 나타났다.

1. `_cast_kv_tile`이 bf16 타일을 돌려줬지만 mqa paged 커널은 여전히 106496바이트를 요청했다.
   `STAGES=2` 소프트웨어 파이프라인이 타일 풋프린트를 2배로 만들었기 때문이다. `STAGES`를 1로
   낮춰 mqa 커널이 들어갔다.
2. 다음 상한은 `qwen38_ops_qsa_wide.py:1006`의 `_qsa_sparse_paged_gqa_splitk_kernel`이었다. 이
   커널도 `num_stages=2`로 돌았다. 1로 낮추자 splitk 커널이 들어갔고 엔진은 325.5초에 `ready`,
   `677,901 tokens, 1.99x`에 도달했다.
3. 엔진은 code C=1 98.81 / 109.35 / 103.92, 중앙 103.92, prose C=1 31.87 / 49.71 / 49.65, 중앙
   49.65를 측정했다.

넓은 타일은 단계 1 대비 code 15.5퍼센트, prose 1.8퍼센트를 잃었다. 소프트웨어 파이프라인을
제거하는 비용이 넓은 타일의 이득보다 크므로 되돌렸다. 48 같은 2의 거듭제곱이 아닌 중간 폭은
Triton 블록 상수가 2의 거듭제곱이어야 해서 쓸 수 없고, 32는 이미 쓰던 반감 값이다. 따라서
넓은 타일 방향은 101376바이트 한도 안에서 2단 파이프라인을 유지하는 커널이 필요하며, 이는
PDF가 말한 PR #60602의 레지스터 내 바이트 디코드 방향이다.

## 6. 다른 축

넓은 타일이 실패한 뒤 두 방향을 단계 1과 비교 측정했다.

### 6.1 청크 누산

전체 헤드 차원에 걸친 단일 `tl.dot(keys, query)` 대신 `D_CHUNK = 32` 폭의 dot을
`tl.static_range` 루프로 누산한다. 각 청크의 공유 메모리 operand는 전체 폭 타일의 4분의 1이라
`STAGES=2`가 `block_n` 반감 없이 101376바이트 한도에 들어간다. 엔진은 376.3초에 `ready`,
`677,901 tokens, 1.99x`에 도달해 공유 메모리 상한이 사라졌다.

측정값은 code C=1 111.75 / 124.73 / 104.22, 중앙 111.75, prose C=1 28.54 / 50.88 / 49.44,
중앙 49.44였다. 단계 1 대비 code 9.1퍼센트, prose 2.3퍼센트 낮다. 청크 dot 4회가 넓은 단일
dot의 이득보다 런치와 누산기 트래픽을 더 쓰므로 되돌렸다.

### 6.2 MTP 깊이

드래프트 깊이를 15에서 12, 10으로 낮췄다. KV admission도 함께 움직였다. 깊이 15에서
677,901토큰 1.99x, 12에서 732,133토큰 2.15x, 10에서 781,768토큰 2.30x다.

|깊이|code C=1 중앙|prose C=1 중앙|admission|
|---:|---:|---:|---|
|15 (단계 1)|**122.96**|50.58|677,901, 1.99x|
|12|111.83|50.54|732,133, 2.15x|
|10|107.78|**54.39**|781,768, 2.30x|

깊이가 줄면 스텝당 드래프트 토큰이 줄어 수락 비율이 낮아지므로 code가 내려간다. prose는 깊이
10에서 단계 1 대비 7.5퍼센트 올랐다. 어느 깊이도 code에서 단계 1을 넘지 못하므로 깊이 15는
code 기본값으로 남고, 깊이 10은 prose 프로파일이 된다.

## 7. 목표 대비 결과

최종 구성은 code C=1 122.96, prose C=1 50.58에 도달했다. 목표는 180과 70이었다. 남은 격차의
핵심은 `qwen38_ops_qsa_v2.py:926`의 `block_n` 반감으로, fp8_e5m2 어텐션 스텝마다 디코드 타일
폭의 절반을 잃는다. 이를 제거한 세 번의 측정이 모두 손실이었다. 파이프라인을 포기한 넓은
타일, dot을 청크로 나눈 방식, 드래프트를 줄인 방식이다. 청크 누산 결과는 공유 메모리 상한이
`STAGES=2`를 유지한 채 제거 가능하다는 것을 보여주지만, 청크별 오버헤드가 넓은 타일 이득보다
크다는 가장 유용한 부정 결과다. int8 per-token-head는 스케일 텐서가 넓은 타일이 쓸 메모리를
소비해 이 장치에서는 격차를 닫지 못했다.

순이득은 프로파일 분리 하나다. code 작업은 깊이 15의 `sep-c2s48.sh`, prose 작업은 깊이 10의
`sep-c2m10.sh`로 KV 풀이 15퍼센트 더 크다.

## 8. 재현

```bash
ssh -i ~/.ssh/<key> -o BatchMode=yes user@<spark-host> \
  'cd /home/user/OMP_Workspace/qwen38-gb10-3090-20261029 && ./sep-c2s48.sh'
```

```bash
ssh -i ~/.ssh/<key> -o BatchMode=yes user@<spark-host> \
  'cd /home/user/OMP_Workspace/qwen38-gb10-3090-20261029 && python3 sep-c2s48_driver.py'
```

드라이버는 기존 `a-sep-c2s48-execution.json`을 덮어쓰지 않으므로, 이전 실행 디렉터리와 실행
파일을 먼저 지워야 한다.

## 7. 측정 출처

이 문서의 모든 수치는 고정된 sparkDash native 러너 커밋
`7c2b9638b025cd39fb43f95c225de62980f130ee`에서 나왔다. 각 작업의 `status`, `failures`, 행별
`meanDecodeTps`는 해당 `a-<prefix>-execution.json`에 기록된다. `samples: 0`을 보고하는 프리필
행은 32,768토큰 엔진 코어 종료이며 처리량 근거로 쓰지 않는다.
