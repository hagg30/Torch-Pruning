"""
MoE 모델 프루닝 테스트 스크립트
개선된 dependency graph로 MoE 모델을 안전하게 프루닝합니다.
"""

import torch
import torch_pruning as tp
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse


def test_moe_pruning(model_name, cache_dir=None, pruning_ratio=0.5):
    """
    MoE 모델을 프루닝하고 dependency graph가 정상 작동하는지 확인
    
    Args:
        model_name: HuggingFace 모델 이름 (예: "Qwen/Qwen2.5-MoE-7B")
        cache_dir: 모델 캐시 디렉토리
        pruning_ratio: 프루닝 비율
    """
    print(f"Loading model: {model_name}")
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 더미 입력 생성
    text = "Hello, how are you?"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    print("\n" + "="*50)
    print("모델 정보:")
    print(f"  - Hidden size: {model.config.hidden_size}")
    print(f"  - Num layers: {model.config.num_hidden_layers}")
    
    # MoE 정보 출력
    has_experts = False
    num_experts = 0
    for name, module in model.named_modules():
        if '.experts.' in name or '.expert.' in name:
            has_experts = True
            if '.experts.0' in name or '.expert.0' in name:
                # Count experts in first layer
                parent_name = '.'.join(name.split('.')[:-2])
                num_experts = sum(1 for n, _ in model.named_modules() 
                                if n.startswith(parent_name + '.experts.') or n.startswith(parent_name + '.expert.'))
                break
    
    if has_experts:
        print(f"  - MoE 모델 감지: Expert 수 = {num_experts}")
    else:
        print("  - 일반 Transformer 모델")
    print("="*50 + "\n")
    
    # Dependency graph 빌드
    print("Building dependency graph...")
    try:
        importance = tp.importance.GroupMagnitudeImportance(p=2, group_reduction='mean')
        
        pruner = tp.pruner.BasePruner(
            model,
            example_inputs=inputs.input_ids,
            importance=importance,
            global_pruning=False,
            output_transform=lambda x: x.logits,
            pruning_ratio=pruning_ratio,
            ignored_layers=[model.lm_head],
            round_to=4,
        )
        
        print("✓ Dependency graph 빌드 성공!")
        print(f"✓ 프루닝 비율: {pruning_ratio}")
        
        # 프루닝 실행
        print("\n프루닝 실행 중...")
        pruner.step()
        
        print("✓ 프루닝 완료!")
        
        # 프루닝 후 모델 정보
        print("\n" + "="*50)
        print("프루닝 후 모델 정보:")
        new_hidden_size = model.lm_head.in_features
        print(f"  - Hidden size: {model.config.hidden_size} → {new_hidden_size}")
        
        # 파라미터 수 계산
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  - Total parameters: {num_params:,}")
        print("="*50 + "\n")
        
        # 추론 테스트
        print("추론 테스트 중...")
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=20,
                do_sample=False,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"생성된 텍스트: {generated_text}")
        print("\n✓ 모든 테스트 통과!")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 오류 발생: {type(e).__name__}")
        print(f"  메시지: {str(e)}")
        
        # 스택 트레이스 출력
        import traceback
        print("\n스택 트레이스:")
        traceback.print_exc()
        
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoE 모델 프루닝 테스트")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace 모델 이름 (예: Qwen/Qwen2.5-MoE-7B)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="모델 캐시 디렉토리",
    )
    parser.add_argument(
        "--pruning_ratio",
        type=float,
        default=0.5,
        help="프루닝 비율 (0.0 ~ 1.0)",
    )
    
    args = parser.parse_args()
    
    success = test_moe_pruning(
        model_name=args.model,
        cache_dir=args.cache_dir,
        pruning_ratio=args.pruning_ratio,
    )
    
    exit(0 if success else 1)

