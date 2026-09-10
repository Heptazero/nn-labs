"""e：测量独立于模型，模型不会接收 target。"""
import torch

# [观测·数据] 二值 exact recall：N 个位置必须逐元素完全一致
def e1_exact_recall(final_state: torch.Tensor, target: torch.Tensor) -> bool:
    return bool(torch.equal(final_state.to(torch.int8), target.to(torch.int8)))


# [观测·数据] overlap=(1/N)Σ_i s_i ξ_i；连续输出也有定义
def e2_overlap(final_state: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        torch.mean(final_state.to(torch.float64) * target.to(torch.float64)).item()
    )


# [观测·数据] 用终态与全部记忆的内积排序，返回最相似记忆 ID
def e3_top1_memory(final_state: torch.Tensor, memories: torch.Tensor) -> int:
    # [中介变量] memories:(P,N)@state:(N,)→scores:(P,)
    scores = memories.to(torch.float64) @ final_state.to(torch.float64)
    return int(torch.argmax(scores).item())


# [观测·分类] 先区分连续/离散语义，再判目标、错误记忆、反记忆或虚假固定点
def e4_attractor_class(
    final_state: torch.Tensor,
    target_id: int,
    memories: torch.Tensor,
    status: str,
    output_kind: str,
) -> str:
    top1_id = e3_top1_memory(final_state, memories)
    # [边界] 连续终态不是离散吸引子，只报告它 Top-1 指向谁
    if output_kind == "continuous":
        return "continuous_target" if top1_id == target_id else "continuous_other"
    # [边界] 未到 fixed/one_step 的状态不能事后硬分成某个吸引子
    if status not in {"fixed", "one_step"}:
        return "nonconverged"
    final = final_state.to(torch.int8)
    binary_memories = memories.to(torch.int8)
    # [判断] 广播比较 P 条记忆，找出与终态完全相同的存储项
    matches = torch.all(binary_memories == final.unsqueeze(0), dim=1)
    matched_ids = torch.nonzero(matches, as_tuple=False).flatten().tolist()
    if target_id in matched_ids:
        return "target"
    if matched_ids:
        return "wrong_memory"
    # [判断] 反记忆单列，避免把全局翻转混入普通 spurious fixed point
    inverse_matches = torch.all(binary_memories == -final.unsqueeze(0), dim=1)
    if bool(torch.any(inverse_matches)):
        return "inverse_memory"
    return "spurious_fixed" if status == "fixed" else "one_step_nonmemory"

def retrieve_measured(model, cue, *, target, update_seed, max_sweeps, observer=None):
    """兼容旧实验的误差轨迹；target 仅进入此处的外部观察器。"""
    errors = []
    def record(step, state, energy, diagnostics):
        signed = torch.where(state >= 0, 1.0, -1.0)
        errors.append(float((signed != target.to(signed.device)).double().mean().item()))
        if observer is not None:
            observer(step, state, energy, diagnostics)
    result = model.retrieve(cue, update_seed=update_seed, max_sweeps=max_sweeps, observer=record)
    result.error_trace = errors
    return result
