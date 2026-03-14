import os
import sys
import yaml
import json
import random
import math
import argparse
from typing import Dict, List, Optional, Tuple, Any

current_dir = os.path.dirname(os.path.abspath(__file__))
rms_norm_xgboost_dir = os.path.join(current_dir, "rms_norm_xgboost")
sys.path.append(rms_norm_xgboost_dir)

try:
    from rms_norm_xgboost import predict_xgboost
except ImportError:
    print(f"Error: Could not import predict_xgboost from {rms_norm_xgboost_dir}")
    sys.exit(1)

RMS_NORM_YAML_PATH = "/root/LLM-Compiler/InfiniCore/scripts/profile/rms_norm/rms_norm.yaml"
MODEL_PATH = os.path.join(rms_norm_xgboost_dir, "xgboost_model.json")
META_PATH = os.path.join(rms_norm_xgboost_dir, "xgboost_model_meta.json")
HARDWARE_NAME = "NVIDIA 4090"
WORK_LOAD = {
    "batch_size": 8,
    "sequence_length": 512,
    "hidden_size": 4096,
    "type": "prefill",
}


class Node:
    def __init__(self, schedule: Dict, workload: Dict, parent: Optional["Node"] = None, performance: float = 0.0):
        self.schedule = schedule
        self.workload = workload
        self.parent = parent
        self.performance = performance
        self.visits = 0
        self.total_reward = 0.0
        self.children = []

    def uct(self, exploration_weight: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        if self.parent is None:
            return 0.0
        avg_reward = self.total_reward / self.visits
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        return avg_reward + exploration

    def to_dict(self):
        return {
            "schedule": self.schedule,
            "performance": self.performance,
        }


class LLMClient:
    def __init__(self, schedule_space: Dict):
        self.schedule_space = schedule_space
        self._client = None

    def _normalize_value(self, key: str, value: Any) -> Any:
        allowed = self.schedule_space[key]
        if value in allowed:
            return value
        if any(isinstance(a, bool) for a in allowed):
            if isinstance(value, str):
                low = value.strip().lower()
                if low == "true":
                    return True
                if low == "false":
                    return False
            if isinstance(value, (int, float)):
                if int(value) == 1:
                    return True
                if int(value) == 0:
                    return False
        return value

    def generate_optimization_suggestion(self, prompt: str) -> Dict:
        def extract_json_object(text: str) -> Dict:
            raw = (text or "").strip()
            if not raw:
                raise ValueError("Empty LLM output")
            if raw.startswith("```"):
                parts = raw.split("```")
                if len(parts) >= 2:
                    raw = parts[1].strip()
                    if raw.lower().startswith("json"):
                        raw = raw[4:].strip()
            try:
                obj = json.loads(raw)
                if not isinstance(obj, dict):
                    raise ValueError("LLM output JSON is not an object")
                return obj
            except Exception:
                start = raw.find("{")
                end = raw.rfind("}")
                if start == -1 or end == -1 or end <= start:
                    raise
                obj = json.loads(raw[start : end + 1])
                if not isinstance(obj, dict):
                    raise ValueError("LLM output JSON is not an object")
                return obj

        def validate_suggestion(obj: Dict) -> Dict:
            filtered: Dict[str, Any] = {}
            for k, _ in self.schedule_space.items():
                if k not in obj:
                    continue
                v = self._normalize_value(k, obj[k])
                if v in self.schedule_space[k]:
                    filtered[k] = v
            return filtered

        api_key = os.environ.get("AIHUBMIX_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("AIHUBMIX_BASE_URL") or "https://aihubmix.com/v1"
        model = os.environ.get("AIHUBMIX_MODEL") or "gpt-5.1"
        temperature_str = os.environ.get("AIHUBMIX_TEMPERATURE", "0.7")
        reasoning_effort = (os.environ.get("AIHUBMIX_REASONING_EFFORT") or "none").strip().lower()

        if not api_key:
            raise RuntimeError("api key is not set")

        try:
            import openai
        except Exception:
            raise RuntimeError("openai module not found")

        if self._client is None:
            self._client = openai.OpenAI(api_key=api_key, base_url=base_url)

        try:
            temperature = float(temperature_str)
        except Exception:
            temperature = 0.7

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if reasoning_effort and reasoning_effort != "none":
            kwargs["reasoning_effort"] = reasoning_effort

        try:
            response = self._client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            print("[DEBUG] LLM raw output:", content)
            suggestion = validate_suggestion(extract_json_object(content))
            if suggestion:
                print(f"LLM suggestion: {suggestion}")
                return suggestion
            raise ValueError("LLM output JSON is not a valid schedule suggestion")
        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {e}")


def _list_kv_to_dict(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for item in entries:
        if isinstance(item, dict):
            result.update(item)
    return result


def load_config(yaml_path: str) -> Tuple[Dict, Dict]:
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_schedule = config["schedule"]
    raw_workload = config["workload"]
    schedule_space = _list_kv_to_dict(raw_schedule)
    workload_space = _list_kv_to_dict(raw_workload)
    return schedule_space, workload_space


def get_random_workload(workload_config: Dict) -> Dict:
    w_type = random.choice(list(workload_config.keys()))
    params = workload_config[w_type]
    batch_size = random.choice(params["batch_size"])
    sequence_length = random.choice(params["sequence_length"])
    hidden_size = random.choice(params["hidden_size"])
    return {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "hidden_size": hidden_size,
        "type": w_type,
    }


def get_random_schedule(schedule_space: Dict) -> Dict:
    schedule = {}
    for k, v in schedule_space.items():
        schedule[k] = random.choice(v)
    return schedule


def predict(schedule: Dict, workload: Dict) -> float:
    features = {
        "batch_size": workload["batch_size"],
        "seq_len": workload["sequence_length"],
        "hidden_size": workload["hidden_size"],
        "block_size_value": schedule.get("block_size_value", 64),
        "num_warps": schedule.get("num_warps", 4),
        "num_stages": schedule.get("num_stages", 2),
        "use_vectorized_application": schedule.get("use_vectorized_application", False),
    }

    try:
        score = predict_xgboost(features, model_path=MODEL_PATH, meta_path=META_PATH)
        print(f"[DEBUG] predict_xgboost score {score}")
        return score
    except Exception:
        return -1.0


def format_rms_norm_workload(workload: Dict) -> str:
    b = workload.get("batch_size")
    s = workload.get("sequence_length")
    h = workload.get("hidden_size")
    t = workload.get("type", "unknown")
    if b is None or s is None or h is None:
        return json.dumps(workload, ensure_ascii=False)
    return f"type={t}, X({b}*{s}*{h})"


def save_json(path: str, data: Any) -> None:
    abs_path = os.path.abspath(path)
    dir_path = os.path.dirname(abs_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    tmp_path = abs_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, abs_path)


def construct_prompt(node: Node, schedule_space: Dict) -> str:
    history = []
    curr = node
    while curr:
        history.append(curr.to_dict())
        curr = curr.parent
    history.reverse()

    required_keys = list(schedule_space.keys())
    example = {k: (schedule_space[k][0] if schedule_space[k] else None) for k in required_keys}

    prompt = f"""
你是一名算子调优专家，现在需要根据以下信息，对GEMM算子进行调优。
Current Hardware: {HARDWARE_NAME}
Workload: {format_rms_norm_workload(node.workload)}
Search Space: {json.dumps(schedule_space)}

History of optimizations (from initial to current):
"""
    for i, item in enumerate(history):
        prompt += f"Step {i}: Config={json.dumps(item['schedule'])}, Score={item['performance']}\n"

    prompt += f"""
Specific Requirements:
性能分数代表对某个候选配置在对应问题上“更优”的相对评分，分数越大，排序越靠前。根据以上信息，分析当前算子配置(Last Step)的性能分数，判断是否需要进行调优。如果需要，根据你的先验知识，分析性能分数与其他变体的差异，识别性能变化的来源。根据分析结果，生成新的调优建议。
生成的调优建议必须是json格式,且必须包含以下实例json中包含的所有参数:
```
{json.dumps(example, ensure_ascii=False, indent=2)}
```
注意，生成的调优建议中只能包含实例json中包含的参数，不能包含其他参数。你的回答只能包含该json,且只能包含一个json模块，严禁包含其他内容或思考过程。
"""
    return prompt


def expand(node: Node, llm_client: LLMClient, schedule_space: Dict) -> Node:
    prompt = construct_prompt(node, schedule_space)
    suggestion = llm_client.generate_optimization_suggestion(prompt)
    new_schedule = node.schedule.copy()
    new_schedule.update(suggestion)
    performance = predict(new_schedule, node.workload)
    child_node = Node(new_schedule, node.workload, parent=node, performance=performance)
    node.children.append(child_node)
    return child_node


def backpropagate(node: Node, reward: float):
    curr = node
    while curr:
        curr.visits += 1
        curr.total_reward += reward
        curr = curr.parent


def run_mcts(iterations: int, verbose: bool = True, output_path: Optional[str] = None, random_workload: bool = False):
    print("Loading configuration...")
    schedule_space, workload_config = load_config(RMS_NORM_YAML_PATH)

    workload = get_random_workload(workload_config) if random_workload else WORK_LOAD
    print(f"Target Workload: {workload}")

    initial_schedule = get_random_schedule(schedule_space)
    initial_score = predict(initial_schedule, workload)
    root = Node(initial_schedule, workload, performance=initial_score)
    print(f"Initial Schedule: {initial_schedule}")
    print(f"Initial Score: {initial_score:.4f}")

    llm_client = LLMClient(schedule_space)
    best_node = root

    print(f"Starting MCTS for {iterations} iterations...")
    for i in range(iterations):
        node = root
        depth = 0
        while node.children:
            node = max(node.children, key=lambda n: n.uct())
            depth += 1

        child = expand(node, llm_client, schedule_space)
        backpropagate(child, child.performance)

        if child.performance > best_node.performance:
            best_node = child
            if verbose:
                print(f"Iter {i + 1}: New Best Score: {best_node.performance:.4f} (Depth: {depth + 1})")
        elif verbose and (i + 1) % 10 == 0:
            print(f"Iter {i + 1}: Best Score: {best_node.performance:.4f}")

    print("\nOptimization Finished.")
    print(f"Best Score: {best_node.performance:.4f}")
    print(f"Best Schedule: {json.dumps(best_node.schedule, indent=2)}")

    if output_path:
        result = {
            "hardware": HARDWARE_NAME,
            "workload": workload,
            "iterations": iterations,
            "best_score": best_node.performance,
            "best_schedule": best_node.schedule,
        }
        save_json(output_path, result)
        print(f"Best result saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize RMSNorm operator using MCTS and LLM.")
    parser.add_argument("--iterations", type=int, default=20, help="Number of MCTS iterations")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(current_dir, "best_rms_norm_result.json"),
        help="Path to save best result JSON",
    )
    parser.add_argument("--random-workload", action="store_true", help="Sample workload from YAML config")
    args = parser.parse_args()

    run_mcts(args.iterations, args.verbose, args.output, args.random_workload)
