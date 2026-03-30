
# lora + pure text

CUDA_VISIBLE_DEVICES=0,1 python table_train.py \
--dataset structprobe \
--second_dataset structprobe_permute \
--prompt_type llama2 \
--model_name llm \
--llm_model_name 7b \
--seed 42 \
--project structprobe \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-all/llama2-lora \
--llm_num_virtual_tokens 8 \
--patience 3 \
--llm_frozen False

# lora + pure text

CUDA_VISIBLE_DEVICES=0,1 python table_train.py \
--dataset wtq_orig \
--prompt_type llama2 \
--model_name llm \
--llm_model_name 7b \
--seed 42 \
--project wtq_orig \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-all/llama2-lora \
--llm_num_virtual_tokens 8 \
--patience 3 \
--llm_frozen False

# lora + pure text

CUDA_VISIBLE_DEVICES=0,1 python table_train.py \
--dataset fetaqa \
--prompt_type llama2 \
--model_name llm \
--llm_model_name 7b \
--seed 42 \
--project fetaqa \
--max_txt_len 768 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-all/llama2-lora \
--llm_num_virtual_tokens 8 \
--patience 3 \
--llm_frozen False

# lora + pure text

CUDA_VISIBLE_DEVICES=0,1 python table_train.py \
--dataset hitab \
--prompt_type llama2 \
--model_name llm \
--llm_model_name 7b \
--seed 42 \
--project hitab \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-all/llama2-lora \
--llm_num_virtual_tokens 8 \
--patience 3 \
--llm_frozen False

# lora + pure text

CUDA_VISIBLE_DEVICES=0,1 python table_train.py \
--dataset wikisql \
--prompt_type llama2 \
--model_name llm \
--llm_model_name 7b \
--seed 42 \
--project wikisql \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-all/llama2-lora \
--llm_num_virtual_tokens 8 \
--patience 3 \
--llm_frozen False


# lora + tamo

CUDA_VISIBLE_DEVICES=0,1 python table_train.py \
--dataset structprobe \
--second_dataset structprobe_permute \
--prompt_type llama2 \
--model_name table_hypergraph_llm \
--llm_model_name 7b \
--seed 42 \
--project structprobe \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-all/llama2-lora \
--llm_num_virtual_tokens 8 \
--gnn_model_name hyper \
--gnn_num_layers 1 \
--patience 3 \
--llm_frozen False

# lora + tamo

CUDA_VISIBLE_DEVICES=0,1 python table_train.py \
--dataset wtq_orig \
--prompt_type llama2 \
--model_name table_hypergraph_llm \
--llm_model_name 7b \
--seed 42 \
--project wtq_orig \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-all/llama2-lora \
--llm_num_virtual_tokens 8 \
--gnn_model_name hyper \
--gnn_num_layers 1 \
--patience 3 \
--llm_frozen False

# lora + tamo

CUDA_VISIBLE_DEVICES=0,1 python table_train.py \
--dataset fetaqa \
--prompt_type llama2 \
--model_name table_hypergraph_llm \
--llm_model_name 7b \
--seed 42 \
--project fetaqa \
--max_txt_len 768 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-all/llama2-lora \
--llm_num_virtual_tokens 8 \
--gnn_model_name hyper \
--gnn_num_layers 1 \
--patience 3 \
--llm_frozen False

# lora + tamo

CUDA_VISIBLE_DEVICES=0,1 python table_train.py \
--dataset hitab \
--prompt_type llama2 \
--model_name table_hypergraph_llm \
--llm_model_name 7b \
--seed 42 \
--project hitab \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-all/llama2-lora \
--llm_num_virtual_tokens 8 \
--gnn_model_name hyper \
--gnn_num_layers 1 \
--patience 3 \
--llm_frozen False

# lora + tamo

CUDA_VISIBLE_DEVICES=0,1 python table_train.py \
--dataset wikisql \
--prompt_type llama2 \
--model_name table_hypergraph_llm \
--llm_model_name 7b \
--seed 42 \
--project wikisql \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-all/llama2-lora \
--llm_num_virtual_tokens 8 \
--gnn_model_name hyper \
--gnn_num_layers 1 \
--patience 3 \
--llm_frozen False

