
# inference

CUDA_VISIBLE_DEVICES=0,1 python inference.py \
--dataset structprobe \
--second_dataset structprobe_permute \
--prompt_type llama2 \
--model_name llm \
--llm_model_name 7b_chat \
--seed 42 \
--project structprobe \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-all/llama2-infer \
--llm_num_virtual_tokens 8 \
--patience 3 

# inference

CUDA_VISIBLE_DEVICES=0,1 python inference.py \
--dataset wtq_orig \
--second_dataset wtq_permute \
--prompt_type llama2 \
--model_name llm \
--llm_model_name 7b_chat \
--seed 42 \
--project wtq_orig \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-all/llama2-infer \
--llm_num_virtual_tokens 8 \
--patience 3 

# inference

CUDA_VISIBLE_DEVICES=0,1 python inference.py \
--dataset fetaqa \
--prompt_type llama2 \
--model_name llm \
--llm_model_name 7b_chat \
--seed 42 \
--project fetaqa \
--max_txt_len 768 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-all/llama2-infer \
--llm_num_virtual_tokens 8 \
--patience 3 

# inference

CUDA_VISIBLE_DEVICES=0,1 python inference.py \
--dataset hitab \
--prompt_type llama2 \
--model_name llm \
--llm_model_name 7b_chat \
--seed 42 \
--project hitab \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-all/llama2-infer \
--llm_num_virtual_tokens 8 \
--patience 3 

# inference

CUDA_VISIBLE_DEVICES=0,1 python inference.py \
--dataset wikisql \
--prompt_type llama2 \
--model_name llm \
--llm_model_name 7b_chat \
--seed 42 \
--project wikisql \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-all/llama2-infer \
--llm_num_virtual_tokens 8 \
--patience 3 
