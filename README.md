# GraphPRM

Code and data for KDD 2025 Research Track Anonymous Submission: "Rewarding Graph Reasoning Process makes LLMs more Generalized Reasoners"

## Dataset and Model Weight Link

**Full dataset can also be accessed at:** [GraphSilo](https://huggingface.co/datasets/GraphPRM/GraphSilo), [GraphSilo-Test](https://huggingface.co/datasets/GraphPRM/GraphSilo-Test) (Anonymous Repository)
**Full GraphPRM model weight can be accessed at:** [GraphPRM-1.5B](https://huggingface.co/GraphPRM/GraphPRM-1.5B), [GraphPRM-7B](https://huggingface.co/GraphPRM/GraphPRM-7B) (Anonymous Repository)

## Key File Descriptions

### `data/`

- `GraphSilo/`: Training set for GraphPRM model (containing step-wise labels from "Task-oriented Trajectories" and "Monte Carlo Estimation").

- `GraphSilo_test/`: Test set of 13 graph tasks in GraphSilo.
  - `[graph_task].jsonl`: Test samples for corresponding graph tasks.
  - `GraphSilo_test_in_domain.jsonl`: Test samples for 10 in-domain graph tasks (that used to train GraphPRM): Degree, Clustering Coefficient, Jaccard, Common Connectivity, Diameter, Page Rank, MST, Maximum Flow, Predecessor.
  - `GraphSilo_test_out_domain.jsonl`: Test samples for 3 out-domain graph tasks (that not used to train GraphPRM): BFS, Neighbor, Cycle.
  - `GraphSilo_test.jsonl`: All test samples including 13 graph tasks.

### `prm/`

- `code/finetune_qwen_SFT.py`: Codes for SFT training GraphPRM with step-wise labels from GraphSilo.
- `config/deepspeed_config_stage3.json`: Configuration for deepspeed stage3 training.

### `reason/`

- `llm_service/create_service_graph.sh`: Script to start LM and RM services.

### `scripts/`

- `eval/best_of_N.sh`: Perform inference-time computation via Best-of-N strategy with GraphPRM.
- `eval/beam_search.sh`: Perform inference-time computation via Beam Search strategy with GraphPRM.

## Usage Instructions

### Installation

```
conda create -n GraphPRM python=3.10
conda activate GraphPRM
pip install -r requirements.txt
pip3 install  "fschat[model_worker,webui]"
pip install -U pydantic
cd envs/MATH/latex2sympy
pip install -e .
cd -
```

### Download Models

Before running the project, please ensure that all required base models are downloaded to directory `hugging_cache`.

1. Download base LLM models: `Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct, Qwen2.5-Math-7B-Instruct, LLaMA3.1-8B-Instruct, Gemma2-9B-Instruct`
2. Download GraphPRM models: `GraphPRM-7B`

To download these models, please refer to the [Hugging Face model downloading tutorial](https://huggingface.co/docs/hub/models-downloading) for step-by-step guidance on downloading models from the Hugging Face Hub.

### Start LM & RM Services

Before running inference, please modify the following variables in the script at `reason/llm_service/create_service.sh` to set the appropriate base models:

- `$MODEL_BASE`: Set this to the directory where the models are stored.
- `$POLICY_MODEL_NAME`: Set this to the name of the policy model.
- `$VALUE_MODEL_NAME`: Set this to the name of the graph reward model.
- `$NUM_LM_WORKER`: Set this to the number of language model (LM) workers to start.
- `$NUM_RM_WORKER`: Set this to the number of reward model (RM) workers to start.

Then it prepares and runs inference using different techniques.

For example, to start the LM and RM services for scaling inference-time computing with GraphPRM, run the following command:
```bash
sh reason/llm_service/create_service.sh
```

To kill the server processes, recommend using the following command:
```bash
tmux kill-session -t {Your Session Name} # default is `GraphPRM`
```

### Run GraphPRM Self-supervised Finetuning
```bash
cd prm/code

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 finetune_qwen_SFT.py
                                             --model_path $YOUR_MODEL_PATH \
                                             --data_path $YOUR_DATA_FOLDER_PATH
```

### Perform Inference-time Computation with GraphPRM

#### Best-of-N
```bash
export PYTHONPATH=$(pwd)

sh scripts/eval/cot_rerank.sh

# Key parameters:
# --LM Qwen2.5-7B-Instruct                        # The name of Policy Model
# --RM GraphPRM-7B                                # The name of Reward Model
# --temperature 0.7                               # The temperature hyper-parameter during generation
# --num_sequence 8                                # The number of generated samples during generation
# --max_new_tokens 2048                           # Max new token number during generation
# --test_set_path dataset/GraphSilo_test.jsonl    # The path to test data file

```

#### Beam Search
```bash
export PYTHONPATH=$(pwd)

sh scripts/eval/beam_search.sh

# Key parameters:
# --LM Qwen2.5-7B-Instruct                        # The name of Policy Model
# --RM GraphPRM-7B                                # The name of Reward Model
# --temperature 0.7                               # The temperature hyper-parameter during generation
# --num_sequence 2                                # The number of samples to remain per step
# --tree_max_width 4                              # The number of generated samples per step during generation
# --tree_max_depth 50                             # Max step number
# --max_new_tokens 2048                           # Max new token number during generation
# --test_set_path dataset/GraphSilo_test.jsonl    # The path to test data file

```