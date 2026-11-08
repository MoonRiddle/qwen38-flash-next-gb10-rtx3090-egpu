# 복구 경로 실측: int8 KV, 드래프트 임베딩 포인터, YaRN 1.35, 단일 스트림 QSA

이 문서는 외부 자문 PDF의 미시도 경로 네 개를 단계 1 구성(`sep-c2s48.sh`) 위에서
단일 변수로 측정한 기록이다. 측정은 고정 커밋 sparkDash native 러너
(`7c2b9638b025cd39fb43f95c225de62980f130ee`)로 C=1·C=2, 디코딩 3회, 프리필 1회를
실행했다.

성공 판정은 실행 기록이 `status completed`, `failures`가 비어 있음, 디코딩 3회의
중앙값이 존재하는 경우에만 인정한다.

## 1. side-scale int8 per-token-head KV 캐시

앞선 int8 시도는 `get_fp8_kv_cache_data_type`가 `int8_per_token_head`를 거부하는
FlashAttention 쓰기 경로에서 멈췄다. 이번에는 그 쓰기를 Triton per-token-head
reshape로 바꾸고, float32 스케일 뷰를 KV 패딩 안에 할당하고, QSA 커널이 점수와
값에 스케일을 곱하도록 배선했다.

엔진은 기동했고 admission은 fp8과 정확히 같은 `677,901 tokens, 1.99x`였다. 실행 기록은
`failed`다. 프리필 작업이 `prefill samples short`를 돌려줬고(세 크기 중 하나만 완료),
`restoration_verified`는 false다.

| 구성 | code C=1 | prose C=1 | admission | 로드 초 |
|---|---:|---:|---|---:|
| 단계 1 `sep-c2s48` | **122.96** | **50.58** | 677,901, 1.99x | 353.4 |
| int8 `sep-c2i8s` | 18.47 | 16.90 | 677,901, 1.99x | 301.2 |

code는 83.8퍼센트, prose는 63.2퍼센트 하락했다. 원인은 두 가지다. `_cast_kv_tile`의
mode 2 분기가 여전히 단순 캐스팅이라 int8 바이트가 디퀀트 없이 bf16 값으로 점곱에
들어간다. 커널도 매 타일마다 스케일을 개별 로드·곱셈하므로 MTP 수용률이 무너진다.
int8 축은 이 포크에서 회수할 수 없다.

## 2. 드래프트 임베딩 포인터

`peer_worker.load_model`은 드래프트 임베딩을 명시적으로 CPU로 옮긴다. 이번 실행은
`QWEN38_DRAFT_EMBED_SHARE=1`로 그 이동을 건너뛰고, 러너 생성 뒤 타깃 임베딩을
바인딩하도록(`peer_inventory`의 `_bind_pending_draft_embed`) 배선했다.

| 구성 | code C=1 | prose C=1 | 로드 초 | 상태 |
|---|---:|---:|---:|---|
| 단계 1 `sep-c2s48` | **122.96** | **50.58** | 353.4 | failed (prefill samples short) |
| 드래프트 임베딩 `sep-c2e` | 115.84 | 48.13 | 361.5 | failed |

code는 단계 1의 94.2퍼센트, prose는 95.2퍼센트였다. 이어서 프리필 작업이 RTX 3090을
소진했다: `Tried to allocate 128.00 MiB`, `23.50 GiB memory in use`. 엔진이 죽었고
`restoration_verified`는 false다. 임베딩을 24 GiB owner에 상주시키는 것은 VRAM 예산과
맞지 않는다.

## 3. Layer 48 라우티드 전문가 INT4

공개 체크포인트는 이미 `quantization_config`에 `quant_method: gptq`, `bits: 4`,
`group_size: 128`을 담고 있고, 동적 규칙은 attention·shared expert·PLE·임베딩·gate만
제외한다. 따라서 Layer 48 라우티드 전문가는 이미 INT4이며, 이 경로는 변환이 필요 없어
드롭한다.

## 4. YaRN factor 1.35 뷰

`build_135_views.py`는 같은 가중치 위에 `factor: 1.35`,
`max_position_embeddings: 340000`인 `/work/target-135`와 `/work/draft-135` 심링크 뷰를
만든다.

마운트 계약에 주의해야 한다. `run_engine_hostnet.sh`는 `$BASE`를 `/work`에 바인딩하므로
뷰는 호스트 `/work`가 아니라 `$BASE` 아래에 만들어야 한다.

| 구성 | code C=1 | prose C=1 | admission | 로드 초 |
|---|---:|---:|---|---:|
| 단계 1 `sep-c2s48` | **122.96** | **50.58** | 677,901, 1.99x | 353.4 |
| YaRN 1.35 `sep-c2y35` | 96.18 | 49.70 | 677,901, 1.99x | 340.3 |

code는 21.8퍼센트, prose는 1.7퍼센트 하락했다. 이어서 프리필에서 엔진이 연결 거부로
죽었고 `restoration_verified`는 false다. 1.35 배율은 측정된 처리량 하락과 프리필 실패를
냈지만, 그 원인을 이 실행으로 확정할 수는 없다.

## 5. 단일 스트림 QSA 패스트패스

`qwen38_ops_qsa_fast.py`는 `q.shape[0] <= 16`일 때 `target_splits = 1`로 둔다. 이는 한
스트림의 MTP 드래프트 배치 상한이다. `num_splits == 1`이면 fp32 partial 버퍼와 merge
커널이 컴파일에서 제거된다. C=2는 행 수가 두 배라 분할 경로를 유지한다.

| 구성 | code C=1 | prose C=1 | admission | 로드 초 |
|---|---:|---:|---|---:|
| 단계 1 `sep-c2s48` | **122.96** | **50.58** | 677,901, 1.99x | 353.4 |
| 단일 스트림 `sep-c2f` | 106.93 | 49.62 | 677,901, 1.99x | 341.4 |

code는 13.0퍼센트, prose는 1.9퍼센트 하락했다. split 병렬성이 merge 오버헤드보다 크다는
뜻이다. 프리필에서 다시 RTX 3090 128MiB 할당에 OOM으로 엔진이 죽었다.

## 6. 채택 조합

새 경로 네 개가 모두 되돌림이므로 채택 조합은 code에 `sep-c2s48.sh`(드래프트 깊이 15),
prose에 `sep-c2m10.sh`(드래프트 깊이 10) 그대로다. code 122.96, prose 54.39로 목표
180과 70에 미달이며, 남은 격차의 핵심은 `block_n` 반감과 MTP 수용률이다.

## 7. 장문 윈도에서의 동시 디코딩

409,600 윈도는 `427,468 tokens, 1.04x`를 허용하므로 완전한 요청 두 개가 들어가지
않는다. 그 윈도에서 동시성을 보려고 `sep-g`(`sep-f`에 `--max-num-seqs 4`)를 C=2와
C=4로 3회 반복했다.

| 조건 | 작업 | C | 요청당 | 합계 | 실패 스트림 |
|---|---|---:|---:|---:|---:|
| C=2 | code | 2 | 149.00 | **260.81** | 0 |
| C=2 | prose | 2 | 44.17 | **86.60** | 0 |
| C=4 | code | 4 | 91.82 | **302.98** | 0 |
| C=4 | prose | 4 | 30.53 | **115.28** | 0 |

중요한 한계: sparkDash native 러너는 `contextLength`를 프리필 전용으로만 쓴다
(`runner.mjs:222`). 따라서 이 디코딩 행들은 러너 고정 짧은 프롬프트, 즉 입력 약
67토큰에 강제 출력 400토큰을 쓴다(두 스트림 작업의 `totalPrefillTokens 133`). 이
행들은 동시성을 측정하며 200,000토큰이나 100,000토큰 프롬프트를 측정하지 않는다.
합계 처리량을 "400k 윈도를 나누면 처리량이 오른다"는 근거로 읽으면 안 되며, 그
주장은 철회한다.

같은 엔진의 프리필은 4K 2,814.48, 16K 2,698.71, 32K 2,621.51이다.

측정 기록은 `status completed`, `failures []`, `restoration_verified` true, 13개
작업이다.

## 재현

스크립트는 원격 작업 디렉터리에 있다.

- `sep-c2i8s.sh` + `sep-c2i8s_driver.py`
- `sep-c2e.sh` + `sep-c2e_driver.py`
- `sep-c2y35.sh` + `sep-c2y35_driver.py`
- `sep-c2f.sh` + `sep-c2f_driver.py`
- `sep-g.sh` + `sep-g_driver.py` (200k x 2, 100k x 4 분할)

뷰는 엔진 이미지 안에서 `build_135_views.py`를
`-v $BASE:/work -v /srv/vault:/srv/vault`로 실행해 만든다.
