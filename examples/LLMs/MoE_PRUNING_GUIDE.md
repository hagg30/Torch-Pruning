# MoE 모델 프루닝 가이드

## 개선 사항

이 버전의 Torch-Pruning은 MoE (Mixture of Experts) 모델을 안전하게 프루닝할 수 있도록 다음과 같은 개선사항이 적용되었습니다:

### 1. **사이클 방지 (Cycle Prevention)**
- `onpath` 세트를 사용한 entering/exit 패턴으로 순환 의존성 감지
- `visited` 세트로 이미 방문한 노드 추적
- `MAX_EXPANSIONS` 한도로 무한 확장 방지 (기본값: 200,000)

### 2. **MoE Expert 경계 차단**
- 서로 다른 expert 간의 dependency를 자동으로 차단
- 예: `experts.0`과 `experts.1` 간의 dependency는 차단됨
- 각 expert는 독립적으로 프루닝됨

### 3. **Shared Expert 및 Router 지원**
- `shared_expert`, `shared_experts`: 모든 expert와 연결 가능
- `gate`, `router`: 모든 expert와 연결 가능
- 이들 모듈은 expert 경계 검사에서 제외됨

### 4. **투명한 연산 최적화**
- ElementWise, Reshape, View, Transpose 등의 연산에서
- 인덱스 변화가 없으면 확장을 중단하여 성능 향상

## 지원 모델 구조

### 일반 MoE 구조
```
model.layers.X.mlp.experts.0.gate_proj
model.layers.X.mlp.experts.0.up_proj
model.layers.X.mlp.experts.0.down_proj
model.layers.X.mlp.experts.1.gate_proj
...
model.layers.X.mlp.experts.N.gate_proj
```

### Shared Expert 포함 구조
```
model.layers.X.mlp.shared_expert.gate_proj
model.layers.X.mlp.shared_expert.up_proj
model.layers.X.mlp.shared_expert.down_proj
model.layers.X.mlp.experts.0.gate_proj
...
```

### Router/Gate 포함 구조
```
model.layers.X.mlp.gate  (또는 router)
model.layers.X.mlp.experts.0.*
...
```

## 사용 방법

### 1. 기본 사용법

```python
import torch
import torch_pruning as tp
from transformers import AutoModelForCausalLM, AutoTokenizer

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-MoE-7B",
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-MoE-7B")

# 더미 입력 생성
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# Pruner 생성
importance = tp.importance.GroupMagnitudeImportance(p=2, group_reduction='mean')

pruner = tp.pruner.BasePruner(
    model,
    example_inputs=inputs.input_ids,
    importance=importance,
    global_pruning=False,
    output_transform=lambda x: x.logits,
    pruning_ratio=0.5,  # 50% 프루닝
    ignored_layers=[model.lm_head],
    round_to=4,
)

# 프루닝 실행
pruner.step()

# 모델 설정 업데이트
model.config.hidden_size = model.lm_head.in_features
for name, m in model.named_modules():
    if name.endswith("self_attn"):
        if hasattr(m, "q_proj"):
            m.hidden_size = m.q_proj.out_features
            m.num_heads = m.hidden_size // m.head_dim
    elif name.endswith("mlp"):
        if hasattr(m, "gate_proj"):
            m.hidden_size = m.gate_proj.in_features

# 모델 저장
model.save_pretrained("./pruned_model")
tokenizer.save_pretrained("./pruned_model")
```

### 2. 테스트 스크립트 사용

```bash
# MoE 모델 테스트
python test_moe_pruning.py \
    --model Qwen/Qwen2.5-MoE-7B \
    --pruning_ratio 0.5

# 캐시 디렉토리 지정
python test_moe_pruning.py \
    --model Qwen/Qwen2.5-MoE-7B \
    --cache_dir ./model_cache \
    --pruning_ratio 0.3
```

### 3. 기존 LLM 프루닝 스크립트 사용

```bash
python prune_llm.py \
    --model Qwen/Qwen2.5-MoE-7B \
    --pruning_ratio 0.5 \
    --save_model ./pruned_moe_model
```

## 문제 해결

### 1. "Dependency expansion overflow" 경고
이는 모델이 매우 복잡하거나 순환 의존성이 있을 수 있음을 의미합니다.

**해결책:**
- `MAX_EXPANSIONS` 값을 늘리거나 줄여보세요
- `graph.py`의 291번 줄 `MAX_EXPANSIONS = 200000`을 수정

### 2. Expert 간 잘못된 프루닝
Expert들이 서로 영향을 받는다면, expert 경계 감지가 제대로 작동하지 않는 것입니다.

**해결책:**
- 모델의 expert 구조를 확인하세요:
```python
for name, _ in model.named_modules():
    if 'expert' in name:
        print(name)
```
- `_get_expert_boundary()` 함수를 모델 구조에 맞게 조정하세요

### 3. Shared Expert가 프루닝되지 않음
Shared expert는 의도적으로 특별하게 처리됩니다.

**해결책:**
- Shared expert를 프루닝하려면 `_get_expert_boundary()` 함수에서
- `'shared_expert'` 키워드 체크를 제거하세요

## 디버깅

### Dependency Graph 시각화
```python
# graph.py에서 verbose=True 설정
pruner = tp.pruner.BasePruner(
    model,
    ...,
    verbose=True,  # 경고 메시지 출력
)
```

### Expert 경계 확인
```python
# 모델의 expert 구조 출력
for name, module in model.named_modules():
    if 'expert' in name or 'gate' in name or 'router' in name:
        print(f"{name}: {type(module).__name__}")
```

## 알려진 제한사항

1. **매우 큰 MoE 모델**: expert 수가 매우 많은 경우 (> 256) 메모리 사용량이 높을 수 있습니다.
2. **커스텀 MoE 구조**: 표준과 다른 expert 명명 규칙을 사용하는 경우 `_get_expert_boundary()` 수정 필요
3. **동적 라우팅**: 런타임에 expert를 동적으로 선택하는 경우 추가 처리 필요

## 지원

문제가 발생하면 다음 정보와 함께 이슈를 제출하세요:
- 모델 이름 및 설정
- 프루닝 파라미터
- 오류 메시지 및 스택 트레이스
- `test_moe_pruning.py`의 실행 결과


