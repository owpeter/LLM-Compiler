import os
import sys
import yaml
import json
import random
import math
import argparse
from typing import Dict, List, Optional, Tuple, Any

# Add gemm_xgboost to path to import predict_xgboost
current_dir = os.path.dirname(os.path.abspath(__file__))
gemm_xgboost_dir = os.path.join(current_dir, "gemm_xgboost")
sys.path.append(gemm_xgboost_dir)

try:
    from gemm_xgboost import predict_xgboost
except ImportError:
    print(f"Error: Could not import predict_xgboost from {gemm_xgboost_dir}")
    sys.exit(1)

# Constants
GEMM_YAML_PATH = "LLM-Compiler/InfiniCore/scripts/profile/gemm/gemm.yaml"
MODEL_PATH = os.path.join(gemm_xgboost_dir, "xgboost_model.json")
META_PATH = os.path.join(gemm_xgboost_dir, "xgboost_model_meta.json")
HARDWARE_NAME = "NVIDIA 4090"  # Placeholder
WORK_LOAD = {
    "m": 512, 
    "n": 1024, 
    "k": 2048, 
}

class Node:
    def __init__(self, schedule: Dict, workload: Dict, parent: Optional['Node'] = None, performance: float = 0.0):
        self.schedule = schedule
        self.workload = workload
        self.parent = parent
        self.performance = performance
        self.visits = 0
        self.total_reward = 0.0
        self.children = []
        
    def uct(self, exploration_weight: float = 1.414) -> float:
        if self.visits == 0:
            return float('inf')
        if self.parent is None:
            return 0.0
        # UCT = avg_reward + c * sqrt(ln(parent_visits) / visits)
        # Normalize reward if possible, but here we use raw performance score
        avg_reward = self.total_reward / self.visits
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        return avg_reward + exploration

    def to_dict(self):
        return {
            "schedule": self.schedule,
            "performance": self.performance
        }

class LLMClient:
    """
    LLM client for generating optimization suggestions.
    """
    def __init__(self, schedule_space: Dict):
        self.schedule_space = schedule_space
        self._client = None

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
            for k, v in obj.items():
                if k not in self.schedule_space:
                    continue
                allowed = self.schedule_space[k]
                if v in allowed:
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
            import openai  # type: ignore
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
            else:
                raise ValueError("LLM output JSON is not a valid schedule suggestion")
        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {e}")

def load_config(yaml_path: str) -> Tuple[Dict, Dict]:
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
        
    return config['schedule'], config['workload']

def get_random_workload(workload_config: Dict) -> Dict:
    
    w_type = random.choice(list(workload_config.keys()))
    params = workload_config[w_type]
    
    m = random.choice(params['m'])
    n = random.choice(params['n'])
    k = random.choice(params['k'])
    
    return {"m": m, "n": n, "k": k, "type": w_type}

def get_random_schedule(schedule_space: Dict) -> Dict:
    schedule = {}
    for k, v in schedule_space.items():
        schedule[k] = random.choice(v)
    return schedule

def predict(schedule: Dict, workload: Dict) -> float:
    # Construct features dict
    features = {
        "m": workload["m"],
        "n": workload["n"],
        "k": workload["k"],
        "block_m": schedule.get("block_m", 16),
        "block_n": schedule.get("block_n", 16),
        "block_k": schedule.get("block_k", 32),
        "unroll": schedule.get("unroll", 1),
        "num_warps": schedule.get("num_warps", 4),
        "num_stages": schedule.get("num_stages", 2),
    }
    
    try:
        score = predict_xgboost(features, model_path=MODEL_PATH, meta_path=META_PATH)
        print(f"[DEBUG] predict_xgboost score {score}")
        return score
    except Exception as e:
        # print(f"Prediction failed: {e}")
        return -1.0 # Penalize invalid configs

def format_gemm_workload(workload: Dict) -> str:
    m = workload.get("m")
    n = workload.get("n")
    k = workload.get("k")
    if m is None or n is None or k is None:
        return json.dumps(workload, ensure_ascii=False)
    shape_desc = f"A({m}*{k}), B({k}*{n}), C({m}*{n})"
    return shape_desc

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
    """
    Constructs the prompt for the LLM as specified in the requirements.
    """
    
    # Collect ancestor info
    history = []
    curr = node
    while curr:
        history.append(curr.to_dict())
        curr = curr.parent
    history.reverse() # Root to current
    
    prompt = f"""
你是一名算子调优专家，现在需要根据以下信息，对GEMM算子进行调优。
Current Hardware: {HARDWARE_NAME}
Workload: {format_gemm_workload(node.workload)}
Search Space: {json.dumps(schedule_space)}

History of optimizations (from initial to current):
"""
    for i, item in enumerate(history):
        prompt += f"Step {i}: Config={json.dumps(item['schedule'])}, Score={item['performance']}\n"
        
    prompt += """
Specific Requirements:
性能分数代表对某个候选配置在对应问题上“更优”的相对评分，分数越大，排序越靠前。根据以上信息，分析当前算子配置(Last Step)的性能分数，判断是否需要进行调优。如果需要，根据你的先验知识，分析性能分数与其他变体的差异，识别性能变化的来源。根据分析结果，生成新的调优建议。
生成的调优建议必须是json格式,且必须包含以下实例json中包含的所有参数:
```
{
    "block_m": 64,
    "block_n": 64,
    "block_k": 32,
    "unroll": 1,
    "num_stages": 2,
    "num_warps": 8,
}
```
注意，生成的调优建议中只能包含实例json中包含的参数，不能包含其他参数。你的回答只能包含该json,且只能包含一个json模块，严禁包含其他内容或思考过程。
"""
    return prompt

def expand(node: Node, llm_client: LLMClient, schedule_space: Dict) -> Node:
    # 1. Construct Prompt
    prompt = construct_prompt(node, schedule_space)
    
    # 2. Call LLM
    suggestion = llm_client.generate_optimization_suggestion(prompt)
    
    # 3. Apply suggestion to create new schedule
    new_schedule = node.schedule.copy()
    new_schedule.update(suggestion)
    
    # 4. Predict performance (Simulation)
    performance = predict(new_schedule, node.workload)
    
    # 5. Create new node
    child_node = Node(new_schedule, node.workload, parent=node, performance=performance)
    node.children.append(child_node)
    
    return child_node

def backpropagate(node: Node, reward: float):
    curr = node
    while curr:
        curr.visits += 1
        curr.total_reward += reward
        curr = curr.parent

def run_mcts(iterations: int, verbose: bool = True, output_path: Optional[str] = None):
    # 1. Load Config
    print("Loading configuration...")
    schedule_space, workload_config = load_config(GEMM_YAML_PATH)
    
    # 2. Define Workload
    workload = WORK_LOAD
    print(f"Target Workload: {workload}")
    
    # 3. Initialize Root (p0)
    initial_schedule = get_random_schedule(schedule_space)
    initial_score = predict(initial_schedule, workload)
    root = Node(initial_schedule, workload, performance=initial_score)
    print(f"Initial Schedule: {initial_schedule}")
    print(f"Initial Score: {initial_score:.4f}")
    
    llm_client = LLMClient(schedule_space)
    
    best_node = root
    
    # 4. MCTS Loop
    print(f"Starting MCTS for {iterations} iterations...")
    for i in range(iterations):
        # Selection
        node = root
        depth = 0
        while node.children:
            # Simple UCT selection
            node = max(node.children, key=lambda n: n.uct())
            depth += 1
            
        # Expansion & Simulation
        # In this workflow, we use LLM to expand the leaf node
        child = expand(node, llm_client, schedule_space)
        
        # Backpropagation
        backpropagate(child, child.performance)
        
        # Track best
        if child.performance > best_node.performance:
            best_node = child
            if verbose:
                print(f"Iter {i+1}: New Best Score: {best_node.performance:.4f} (Depth: {depth+1})")
        elif verbose and (i+1) % 10 == 0:
             print(f"Iter {i+1}: Best Score: {best_node.performance:.4f}")

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
    parser = argparse.ArgumentParser(description="Optimize GEMM operator using MCTS and LLM.")
    parser.add_argument("--iterations", type=int, default=100, help="Number of MCTS iterations")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(current_dir, "best_gemm_result.json"),
        help="Path to save best result JSON",
    )
    args = parser.parse_args()
    
    run_mcts(args.iterations, args.verbose, args.output)
