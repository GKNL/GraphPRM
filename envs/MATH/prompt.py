

# Qwen
COT_EXAMPLES = None
COT_TASK_DESC = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>"
PROBLEM_FORMAT_STR = """<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"""

# # # LLaMA
# COT_EXAMPLES = None
# COT_TASK_DESC = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|eot_id|>"
# PROBLEM_FORMAT_STR = """<|start_header_id|>user<|end_header_id|>\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"""

SEP = "\n\n"
