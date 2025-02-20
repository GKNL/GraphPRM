python reason/evaluation/evaluate.py \
    --LM Qwen2.5-7B-Instruct \
    --RM GraphPRM-7B \
    --task_name MATH \
    --temperature 0.7 \
    --num_sequence 8 \
    --max_new_tokens 2048 \
    --save_dir eval_results_GraphSilo \
    --method best_of_n \
    --num_worker 32 \
    --controller_addr http://0.0.0.0:28777 \
    --test_task GraphSilo_test_all \
    --test_set_path dataset/GraphSilo_test.jsonl

