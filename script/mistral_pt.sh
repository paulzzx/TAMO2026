
# inference

CUDA_VISIBLE_DEVICES=0,1 python inference.py \
--dataset wtq_orig \
--prompt_type mistral \
--model_name mistral \
--llm_model_name mistral_7b_instruct \
--seed 42 \
--project wtq_orig \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-all/mistral-infer \
--llm_num_virtual_tokens 8 \
--patience 3 

# frozen + pure text

CUDA_VISIBLE_DEVICES=0,1 python table_train.py \
--dataset wtq_orig \
--prompt_type mistral \
--model_name pt_mistral \
--llm_model_name mistral_7b \
--seed 42 \
--project wtq_orig \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-all/mistral-pt \
--llm_num_virtual_tokens 8 \
--patience 3

# frozen + tamo

CUDA_VISIBLE_DEVICES=0,1 python table_train.py \
--dataset wtq_orig \
--prompt_type mistral \
--model_name table_hypergraph_mistral \
--llm_model_name mistral_7b \
--seed 42 \
--project wtq_orig \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-all/mistral-pt \
--llm_num_virtual_tokens 8 \
--gnn_model_name hyper \
--gnn_num_layers 1 \
--patience 3
