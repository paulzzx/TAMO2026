
# inference

CUDA_VISIBLE_DEVICES=0 python inference.py \
--dataset structprobe \
--second_dataset structprobe_permute \
--prompt_type tablellama \
--model_name llm \
--llm_model_name table_llama_7b \
--seed 42 \
--project structprobe \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-all/tablellama-infer \
--llm_num_virtual_tokens 8 \
--patience 3 

# inference

CUDA_VISIBLE_DEVICES=0 python inference.py \
--dataset wtq_orig \
--prompt_type tablellama \
--model_name llm \
--llm_model_name table_llama_7b \
--seed 42 \
--project wtq_orig \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-all/tablellama-infer \
--llm_num_virtual_tokens 8 \
--patience 3 

# inference

CUDA_VISIBLE_DEVICES=0 python inference.py \
--dataset fetaqa \
--prompt_type tablellama \
--model_name llm \
--llm_model_name table_llama_7b \
--seed 42 \
--project fetaqa \
--max_txt_len 768 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-all/tablellama-infer \
--llm_num_virtual_tokens 8 \
--patience 3 

# inference

CUDA_VISIBLE_DEVICES=0 python inference.py \
--dataset hitab \
--prompt_type tablellama \
--model_name llm \
--llm_model_name table_llama_7b \
--seed 42 \
--project hitab \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-all/tablellama-infer \
--llm_num_virtual_tokens 8 \
--patience 3 

# inference

CUDA_VISIBLE_DEVICES=0 python inference.py \
--dataset wikisql \
--prompt_type tablellama \
--model_name llm \
--llm_model_name table_llama_7b \
--seed 42 \
--project wikisql \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-all/tablellama-infer \
--llm_num_virtual_tokens 8 \
--patience 3 

