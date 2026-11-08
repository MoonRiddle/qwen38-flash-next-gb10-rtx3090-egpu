# 자문 검토 브리프: GB10 + RTX 3090 eGPU 장문 측정

이 브리프는 `docs/14-pdf-recovery-ladder.md`와 `docs/16-recovery-paths-results.md`에
기록된 측정 묶음의 독립 검토를 요청한다. 질문은 수치를 신뢰할 수 있는지, 그리고
보고가 실제로 측정한 것과 시도만 한 것을 구분하는지다.

## 0. 환경

- 헬퍼: NVIDIA GB10(SM121), 121.7 GiB 통합 메모리.
- 오너: RTX 3090 eGPU(SM86), 24 GiB GDDR6X, PCIe Gen4 x4, non-P2P, unlock BF16.
- 모델: `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`와 MTP 드래프트
  (`draft-k10-int4`), GPTQ 4-bit, `group_size 128`, `desc_act false`.
- 엔진: 포크 vLLM `0.1.dev20073+g8e685d198`, 이미지
  `qwen38-flash-dgx:a5b-autoround-bottleneck-q2`.
- 드라이버·unlock·전력 설정은 건드리지 않았다. sudo도 쓰지 않았다.

## 1. 측정 계약

- 러너: MiaAI-Lab sparkDash native 러너를 커밋
  `7c2b9638b025cd39fb43f95c225de62980f130ee`로 고정했다. 로컬 수집기 대체는 없다.
- 디코딩: `maxTokens 400`, 한 작업에 C=1과 C=2, 3회 반복.
- 프리필: 설정마다 4K·16K·32K로 한 작업.
- 보고 디코딩 값: 3회의 요청당 `medianDecodeTps` 중앙값.
- 보고 합계 값: 3회의 `aggregateDecodeTps` 중앙값.
- 성공 판정은 실행 기록이 `status completed`, `failures` 비어 있음,
  `restoration_verified` true일 때만 인정한다.
- 순간 최고치와 동시 요청 합산은 단일 스트림 근거로 쓰지 않는다.

## 2. 실행 기록 상태

| 기록 | 상태 | 실패 | restoration_verified | 비고 |
|---|---|---|---:|---|
| `sep-c2` | failed | 3 | false | 340,000 디코딩 완료, 프리필 short |
| `sep-c2s48` | failed | 3 | false | 기준 디코딩 완료, 프리필 short |
| `sep-c2i8s` | failed | 1 | false | 디코딩 완료 후 프리필에서 엔진 사망 |
| `sep-c2e` | failed | 3 | false | 프리필에서 RTX 3090 OOM |
| `sep-c2y35` | failed | 3 | false | 프리필에서 엔진 연결 거부 |
| `sep-c2f` | failed | 3 | false | 프리필에서 RTX 3090 OOM |
| `sep-g` | completed | 0 | true | 13개 작업 |

실패한 여섯 기록에도 각각 디코딩 3회가 들어 있다. 아래 디코딩 중앙값은 그 반복에서
나온다. 실패 열은 감추지 않고 그대로 두어, 실패한 실행을 완료된 것처럼 제시하지
않는다.

함께 드러난 것: `a-sep-c2-k6-open-prefill-01.json`은 세 프리필 크기 모두에서
`prefillTps 0`, `samples 0`, `error "fetch failed"`를 돌려줬다. 따라서
`docs/11-concurrency-two.md`의 340,000 프리필 수치는 그 기록에서 추적되지 않는다.

## 3. 복구 사다리 결과 (디코딩, C=1)

기준은 `sep-c2s48.sh`다. skip-layers 48, index share 0, batch 1024, `fp8_e5m2`,
409K 뷰, 드래프트 깊이 15.

| 구성 | code C=1 | prose C=1 | admission | 로드 초 | 결정 |
|---|---:|---:|---|---:|---|
| 기준 `sep-c2s48` | **122.96** | **50.58** | 677,901, 1.99x | 353.4 | 유지 |
| int8 side-scale `sep-c2i8s` | 18.47 | 16.90 | 677,901, 1.99x | 301.2 | 되돌림 |
| 드래프트 임베딩 `sep-c2e` | 115.84 | 48.13 | — | 361.5 | 되돌림 |
| YaRN 1.35 `sep-c2y35` | 96.18 | 49.70 | 677,901, 1.99x | 340.3 | 되돌림 |
| 단일 스트림 `sep-c2f` | 106.93 | 49.62 | 677,901, 1.99x | 341.4 | 되돌림 |

Layer 48 라우티드 전문가는 공개 체크포인트에서 이미 INT4라서 그 경로는 실행 없이
드롭했다.

## 4. 장문 분할 결과 (디코딩)

409,600 윈도는 `427,468 tokens, 1.04x`를 허용하므로 완전한 요청 두 개가 들어가지
않는다. 엔진 `sep-g`는 `sep-f`에 `--max-num-seqs 4`를 더한 것이고 KV는 14 GiB이며,
C=2와 C=4로 3회 반복했다.

한계: sparkDash native 러너는 `contextLength`를 프리필 전용으로만 쓴다. 따라서 이
디코딩 행들은 고정 짧은 프롬프트(입력 약 67토큰, 강제 출력 400토큰)를 쓴다. 이 행들은
동시성을 측정하며 200k나 100k 프롬프트를 측정하지 않는다.

| 조건 | 작업 | C | 요청당 | 합계 | 실패 스트림 |
|---|---|---:|---:|---:|---:|
| C=2 | code | 2 | 149.00 | 260.81 | 0 |
| C=2 | prose | 2 | 44.17 | 86.60 | 0 |
| C=4 | code | 4 | 91.82 | 302.98 | 0 |
| C=4 | prose | 4 | 30.53 | 115.28 | 0 |

같은 엔진의 프리필은 4K 2,814.48, 16K 2,698.71, 32K 2,621.51이다.

## 5. README에 쓴 헤드라인 수치

| 윈도 | 작업 | C=1 | C=2 요청당 | C=2 합계 | 구성 |
|---|---|---:|---:|---:|---|
| 409,600 단일 | code | 185.89 | — | — | `sep-f`, MTP int4, KV 14 GiB |
| 409,600 단일 | prose | 67.28 | — | — | `sep-e`, config extension, KV 14 GiB |
| 340,000 | code | 114.11 | 86.79 | 163.46 | `sep-c2`, KV 15 GiB |
| 340,000 | prose | 45.85 | 40.16 | 78.35 | `sep-c2`, KV 15 GiB |

C=1 값은 3회 반복의 중앙값이다. `sep-e` prose 반복은 27.34, 67.28, 72.84였으므로
72.84는 관측 최댓값이고 중앙값이 아니다. README는 이제 67.28을 쓴다. `sep-c2` code
반복은 114.11, 108.13, 115.32였으므로 139.91은 중앙값이 아니라 단일 회차다. README는
이제 114.11을 쓴다. 409,600 두 행은 서로 다른 구성에서 나왔다. code는 MTP int4에서
185.89로 최고이고, prose도 MTP int4가 70.93으로 단순 config extension의 67.28보다 높다.

## 6. 검토 요청

1. 고정 러너에서 3회 중앙값이 이 잡음 수준에서 디코딩 차이를 실제라고 부르기에
   충분한가. `sep-g`의 prose C=1 편차는 30.80에서 51.97로 넓었다.
2. 프리필이 뒤에서 죽어 `status`가 `failed`인 기록의 디코딩 중앙값을, 실패를 수치
   옆에 표시하는 조건으로 보고하는 것이 정당한가.
3. 400,000 토큰 윈도를 200k x 2와 100k x 4로 나누는 것이 409,600 요청 하나와 같은
   일을 측정하는가, 아니면 작업 자체가 달라져 합계 비교가 오해를 낳는가.
4. "두 요청이 들어가지 않는다"는 주장의 근거로 엔진 로그의 admission 비율
   (`677,901 tokens, 1.99x`, `427,468 tokens, 1.04x`)을 쓰는 것이 맞는가.
5. 보고에서 실제 관측을 과장한 부분이나, 드러나야 할 실패 시도를 빠뜨린 부분이
   있는가.

## 7. 재현

원격 작업 디렉터리 `/home/user/OMP_Workspace/qwen38-gb10-3090-20261029`:

- `sep-c2i8s.sh` + `sep-c2i8s_driver.py`
- `sep-c2e.sh` + `sep-c2e_driver.py`
- `sep-c2y35.sh` + `sep-c2y35_driver.py`
- `sep-c2f.sh` + `sep-c2f_driver.py`
- `sep-g.sh` + `sep-g_driver.py`
- `build_135_views.py` (엔진 이미지 안에서 `-v $BASE:/work -v /srv/vault:/srv/vault`로 실행)

실행 기록은 `a-<prefix>-execution.json`, 작업별 산출은 `a-<prefix>-k6-open-*.json`이다.
