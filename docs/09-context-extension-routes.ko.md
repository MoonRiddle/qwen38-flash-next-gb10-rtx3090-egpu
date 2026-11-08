# 9. 409,600 확장 경로 (외부 피드백 반영)

이 문서는 외부 피드백 PDF("409K 컨텍스트 확장 및 병목 최적화")를 이 저장소의 실측 위에 올려 검증한
기록이다. 피드백은 3단계 경로(A·B·C)를 제안했고, 이 문서는 각 경로의 근거, 이 저장소에서 확인된 사실,
그리고 실제 실행 결과를 구분해서 적는다.

## 9.1 피드백의 요지

| 경로 | 주장 | 전제 조건 |
|---|---|---|
| A | `--kv-mib 14336`(14 GiB)을 유지한 채 `--max-model-len 409600`만 주면 409,600이 들어간다 | 체크포인트 `config.json`에 YaRN과 확장된 `max_position_embeddings`를 넣어야 함 |
| B | MTP 드래프트의 512개 expert를 BF16에서 int4로 바꾸면 3090에서 3.47 GiB가 회수된다 | `quantize_mtp_experts_int4.py` 적용 |
| C | QSA 메인 KV를 8비트로 낮추면 KV 수용량이 1.72~1.86배가 된다 | SM86 전용 `fp8_e5m2`(#60491) 또는 `int8_per_token_head` 패치 |

피드백이 지적한 핵심은 "409,600은 메모리 총량 부족이 아니라 **블록 산술과 기동 피크** 때문"이라는 점이다.

## 9.2 블록 산술 주장과 이 저장소의 대응

피드백은 하이브리드 KV 할당기를 역산해 다음을 주장한다.

- 36개 GDN 계층의 상태는 컨텍스트 길이와 무관한 고정 블록 67개(프리픽스 캐싱 OFF, 약 3.08 GiB)를 차지한다.
- 어텐션 블록 1개는 1,792토큰을 담는다.
- 따라서 요청당 블록 수는 `⌈L/1792⌉ + 67`이고, 409,600은 `229 + 67 = 296`블록이다.
- 311블록(14 GiB)이면 `311 / 296 = 1.0507x`로 들어간다.

이 저장소의 기존 실측 두 점은 이 산식과 일치한다.

| 구성 | KV | 요청당 블록(산출) | 동시성(산출) | 기동 로그(실측) |
|---|---:|---:|---:|---|
| 204,800 | 12 GiB | 182 | 1.4615x | 299,323 tokens, 1.46x |
| 262,144 | 14 GiB | 214 | 1.4533x | 380,966 tokens, 1.45x |
| 409,600 | 16 GiB | 302 | 1.1755x | 481,483 tokens, 1.18x |

즉 409,600에서 16 GiB가 실패한 이유는 수용량이 아니라 **기동 피크**다. 262,144의 정상 상태는 22,913 MiB
(여유 1,663 MiB)지만 가중치 스테이징 피크는 약 23,707 MiB이고, 여기에 KV를 2,048 MiB 더하면 25,755 MiB가
되어 24,576 MiB를 넘는다.

## 9.3 경로 A: 설정만으로 409,600

이 저장소에서 확인된 `config.json` 구조는 피드백의 진단과 일치한다.

- 최상위에 `max_position_embeddings`가 **없다**.
- `text_config.max_position_embeddings`만 262144다.
- `text_config.rope_parameters.rope_type`은 `default`다.

그래서 경로 A는 다음 순서로 진행했다.

1. 체크포인트의 읽기 전용 원본은 그대로 두고, `target-409k`와 `draft-409k` 래퍼 디렉터리를 만들었다.
   각 래퍼는 나머지 파일을 심볼릭 링크하고 `config.json`만 실제 사본으로 둔다.
2. `config.json`의 최상위와 `text_config` 양쪽에 `max_position_embeddings: 409600`을 넣고,
   `rope_parameters`를 `rope_type: yarn`, `factor: 1.5625`(= 409600 / 262144),
   `original_max_position_embeddings: 262144`, `mrope_interleaved: true`, `mrope_section: [11, 11, 10]`,
   `partial_rotary_factor: 0.25`, `rope_theta: 10000000`으로 설정했다.
3. `sep-e.sh`에서 `--max-model-len 409600 --kv-mib 14336 --block-size 160 --max-num-batched-tokens 2048
   --mtp 15 --graphs NONE`으로 기동한다.

이 과정에서 이 포크(vLLM `0.1.dev20073`)에서 추가로 필요한 수정이 드러났다. `Qwen4ExpConfig`는
`transformers`의 `standardize_rope_params()`가 `PretrainedConfig.__init__`보다 먼저 실행되는 경로에서
`self.max_position_embeddings`를 읽는데, 그 시점에는 속성이 아직 없다. 그래서 `Qwen3_8FlashNextConfig.__init__`
첫머리에서 `kwargs`의 `max_position_embeddings`를 먼저 `self`에 설정하도록 모듈을 패치하고,
`run_engine_hostnet.sh`에 그 파일을 읽기 전용으로 바인드 마운트했다.

## 9.4 경로 B: MTP expert INT4

MTP 드래프트 계층의 routed expert 1,536텐서(512 expert × 3 proj)는 `model_extra_tensors.safetensors`에
BF16으로 남아 있다. 변환 도구는 `quantize_mtp_int4.py`이며, GPTQ qweight·scales·qzeros로 다시 패킹한다.

이 저장소의 체크포인트 `quantization_config`는 이미 도구가 요구하는 조건을 만족한다.

```
quant_method gptq  bits 4  group_size 128  sym True  desc_act False
dynamic includes "-:.*layers\.48\..*"
```

다만 이 체크포인트의 `model.safetensors.index.json`은 `metadata.total_size`(175,330,224,632)가 실제 인덱스
텐서 바이트 합(74,914,614,008)과 다르다. 도구는 이 불일치를 오류로 처리하므로, 그 검사만 경고로 낮춘
사본(`quantize_mtp_int4_relaxed.py`)으로 변환한다.

## 9.5 경로 C: QSA 8비트 KV

이 포크의 `qsa.py`는 `supported_kv_cache_dtypes = ["auto", "bfloat16"]`만 허용한다. 8비트 KV를 쓰려면
SM86에서 동작하는 패치가 필요하다. GB10용 `fp8_e4m3` 경로는 SM86에서 Triton 명령어 미지원으로 작동하지
않으므로, `fp8_e5m2` 또는 `int8_per_token_head` 계열이 대상이다. 이 저장소는 아직 이 경로를 실행하지
않았고, 아래 실행 기록에도 포함하지 않는다.

## 9.6 262,144 초과 시 런타임 함정

피드백이 지적한 세 가지는 경로 A가 성공하더라도 별도로 확인해야 한다.

| 함정 | 내용 | 이 저장소의 조치 |
|---|---|---|
| 262,145번째 토큰 | MTP 드래프트의 RoPE 테이블이 262,144행에 고정되어 OOB read로 크래시 | `config.json` 확장으로 드래프트도 같은 값을 상속하게 함 |
| YaRN factor | 정적 스케일링이라 과대 factor는 단문 품질을 떨어뜨림 | 409,600에 정확히 맞는 1.5625 사용 |
| QSA 인덱서 버퍼 | 청크마다 증가하는 로짓 버퍼가 캐시 할당기에 누적 | 현행 `_LOGITS_WORKSPACE_BYTES` 고정 워크스페이스라 영향 없음 |

## 9.7 실행 기록

### 9.7.1 경로 A 기동

```
GPU KV cache size: 427,468 tokens, Maximum concurrency for 409,600 tokens per request: 1.04x
engine_load_seconds 297.25
```

3090은 22,985 MiB(여유 1,591 MiB), 호스트 RAM 여유 52 GiB였다. OOM은 발생하지 않았다.

이 값은 피드백의 산출(311블록, 요청당 296블록, 1.0507x)과 방향이 같고 크기는 조금 작다. 이 포크의 실제
블록 수는 로그의 427,468토큰에서 역산하면 409,600 × 1.04 = 425,984토큰 상당이므로, 고정 블록과 어텐션
블록의 비율이 피드백 산식과 미세하게 다르다. 어느 쪽이든 결론은 같다: 14 GiB KV로 409,600이 들어간다.

### 9.7.2 409,600 decode (C=1, 400토큰 고정)

| 작업 | 회차 | median tok/s |
|---|---|---:|
| code | 01 | 177.80 |
| code | 02 | 197.91 |
| code | 03 | 181.78 |
| prose | 01 (콜드) | 27.34 |
| prose | 02 | 67.28 |
| prose | 03 | 72.84 |

code 중앙값은 181.78, prose 워밍 중앙값은 72.84다. 모든 행이 `streamsOk == 1`, `streamsFailed == 0`이다.

### 9.7.3 409,600 prefill (batch 2048)

| 크기 | promptTokens | 샘플 | prefillTps | TTFT |
|---|---:|---:|---:|---:|
| 4K | 4,134 | 3 | 2,753.36 | 1.64 s |
| 16K | 16,419 | 3 | 2,688.15 | 6.24 s |
| 32K | 32,805 | 3 | 2,623.35 | 12.64 s |

### 9.7.4 262,144 대비 변화

| 지표 | 262,144 | 409,600 | 변화 |
|---|---:|---:|---:|
| code decode (중앙값) | 194.70 | 181.78 | −6.6% |
| prose 워밍 (중앙값) | 76.77 / 73.61 | 72.84 | −1.0% |
| prefill 4K | 2,814.22 | 2,753.36 | −2.2% |
| prefill 16K | 2,723.52 | 2,688.15 | −1.3% |
| prefill 32K | 2,649.84 | 2,623.35 | −1.0% |

즉 윈도우를 56% 늘리는 비용은 decode 6.6%, prefill 1~2%다. 피드백의 "속도 저하 없음(노이즈 범위 내)"
주장은 prefill에는 맞고 decode에는 약간 낙관적이다.

### 9.7.5 262,144 경계 통과

피드백이 지적한 첫 번째 함정은 262,145번째 토큰에서 MTP 드래프트의 RoPE 테이블이 범위를 벗어나
`CUDA error: an illegal memory access`로 엔진 코어가 죽는 현상이다. 경로 A의 `config.json` 확장이 이
함정을 실제로 막는지 확인했다.

요청은 262,200토큰을 목표로 만든 프롬프트였고, 실제 토큰 수는 262,209였다. 16토큰을 강제 생성했다.

```
prompt_tokens: 262,209
completion_tokens: 16
finish_reason: length
ok: true
```

262,144를 넘은 위치에서 생성이 정상 종료했고, 엔진 코어는 죽지 않았다. 즉 드래프트가 확장된
`max_position_embeddings`와 YaRN 테이블을 상속했다는 뜻이다.

### 9.7.6 경로 B 기동

MTP routed expert 1,536텐서를 int4 GPTQ-Marlin g128 RTN으로 변환한 뷰(`target-409k-b`, `draft-409k-b`)를
`--kv-mib 17408`로 기동했다. 세 번의 시도가 있었다.

첫 실패는 링크 경로였다. 뷰의 심볼릭 링크가 컨테이너에 마운트되지 않는 `/models/...`를 가리켜서
`Couldn't instantiate the backend tokenizer`로 멈췄다. 링크를 `/srv/vault/...` 절대 경로로 바꾸자 풀렸다.

두 번째 실패는 디렉터리 권한이었다. 변환 산출물 디렉터리가 `drwx------`(700)라 엔진이 tokenizer를 읽지
못했다. `chmod 755`로 열자 진행했다.

17 GiB KV는 admission까지 통과하고 기동 피크에서 OOM이 났다.

```
GPU KV cache size: 518,185 tokens, Maximum concurrency for 409,600 tokens per request: 1.27x
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.27 GiB.
GPU 0 has a total capacity of 23.61 GiB of which 116.19 MiB is free.
```

16,384 MiB에서도 OOM이 났다.

```
GPU KV cache size: 487,946 tokens, Maximum concurrency for 409,600 tokens per request: 1.19x
torch.OutOfMemoryError: Tried to allocate 1.19 GiB. 1004.19 MiB free.
```

14,336 MiB에서 기동에 성공했다.

```
GPU KV cache size: 427,468 tokens, Maximum concurrency for 409,600 tokens per request: 1.04x
engine_load_seconds 305.18
```

즉 MTP expert를 int4로 줄여 3.47 GiB를 회수했지만, 그 여유를 KV로 다시 밀어넣으면 기동 피크가 다시
넘는다. 409,600 윈도우에서 14 GiB가 이 카드의 실질 상한이다.

### 9.7.7 경로 B decode와 prefill

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

경로 A 대비: code 중앙값 181.78 → 185.89 (+2.3%), prose 워밍 72.84 → 71.90 (−1.3%), prefill 4K
2,753.36 → 2,813.41 (+2.2%). 피드백이 예상한 +2~4% 범위 안이다.

### 9.7.8 경로 C: 8비트 KV

이 포크의 `qsa.py`는 `fp8_e4m3`를 이미 허용하므로 `--kv-cache-dtype fp8_e4m3`로 먼저 시도했다. admission은
통과했지만 Triton이 거부했다.

```
GPU KV cache size: 677,415 tokens, Maximum concurrency for 409,600 tokens per request: 1.65x
triton.compiler.errors.CompilationError
ValueError("type fp8e4nv not supported in this architecture. The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')")
```

피드백이 말한 SM86 제약이 그대로 재현됐다. 수용량 1.65x는 피드백의 673,997~819,200 예상과 일치한다.

그래서 `fp8_e5m2` 경로로 패치했다. `qsa.py`의 두 dtype 검사 목록과 `ops/qsa.py`의 view·quant mode·assert를
e5m2까지 넓혔다. 결과는 세 번째 차단이었다.

```
NotImplementedError: FlashAttention does not support fp8_e5m2 kv-cache on this device.
```

즉 이 포크와 SM86 조합에서는 `fp8_e4m3`는 Triton이, `fp8_e5m2`는 FlashAttention이 막는다.

그런데 이 FlashAttention 차단은 커널 한계가 아니라 포크의 누락이었다. `qsa.py`의 `_fp8` 튜플에
`fp8_e5m2`가 없어 부모 `FlashAttentionImpl.__init__`에 그대로 전달됐고, `forward_qsa`의 uint8 뷰는
모드와 무관하게 `float8_e4m3fn`으로 고정돼 있었으며, storage dtype 검사에는 e5m2가 빠져 있었다. 이 네 곳을
고친 `qwen38_qsa_v2.py`·`qwen38_ops_qsa_v2.py`를 마운트한 뒤 `fp8_e5m2`가 실제로 동작했다.

| KV dtype | `--kv-mib` | admission | 결과 |
|---|---:|---|---|
| bf16 | 14,336 | 427,468 / 1.04x | ready |
| fp8_e5m2 | 14,336 | 677,415 / 1.65x | ready |
| fp8_e5m2 | 15,616 | 738,180 / 1.80x | ready |
| fp8_e5m2 | 15,584 | 696,000 / 2.00x (`--max-model-len 348000`) | 75 MB 할당 부족 |
| fp8_e5m2 | 17,408 | 823,701 / 2.01x | 1.23 GiB 할당 OOM |

`fp8_e4m3`는 여전히 Triton이 막는다. 피드백이 지목한 `int8_per_token_head` 경로는 이 포크에 배선돼 있지
않다.

### 9.7.9 복구

경로 C 종료 후 경로 A 엔진을 다시 띄웠다. `GPU KV cache size: 427,468 tokens`, `1.04x`,
`engine_load_seconds 233.22`로 409,600 윈도우가 정상 동작함을 확인했다.

### 9.7.10 결론

| 경로 | 결과 | 비용 |
|---|---|---|
| A | 409,600 기동 성공, decode −6.6%, prefill −1~2% | 설정만 |
| B | 409,600 기동 성공, code +2.3%, prefill +2.2% | MTP expert int4 변환 |
| C (e4m3) | Triton 차단 | 커널 미지원 |
| C (e5m2) | 409,600에서 1.80x 기동 성공, 2.00x는 할당 피크 17 MiB 부족 | 포크 4곳 패치 |

409,600은 경로 A만으로 달성된다. 경로 B는 같은 윈도우에서 속도를 조금 더 올린다. 경로 C의 `fp8_e5m2`는
포크 누락 4곳을 고치면 동작하고, KV 수용량을 1.586배로 늘린다. 다만 동시성 2.0x는 KV 할당 피크에서 막혀
`--max-model-len`을 348,000으로 줄여야 하고, 그마저도 17 MiB가 부족해 별도 트림이 필요하다.
