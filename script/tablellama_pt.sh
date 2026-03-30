
# inference

CUDA_VISIBLE_DEVICES=0,1 python inference.py \
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

# frozen + pure text

CUDA_VISIBLE_DEVICES=0,1 python table_train.py \
--dataset wtq_orig \
--second_dataset wtq_permute \
--prompt_type llama2 \
--model_name pt_llm \
--llm_model_name table_llama_7b \
--seed 42 \
--project wtq_orig \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-all/tablellama-pt \
--llm_num_virtual_tokens 8 \
--patience 3

# frozen + tamo

CUDA_VISIBLE_DEVICES=0,1 python table_train.py \
--dataset wtq_orig \
--second_dataset wtq_permute \
--prompt_type llama2 \
--model_name table_hypergraph_llm \
--llm_model_name table_llama_7b \
--seed 42 \
--project wtq_orig \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-all/tablellama-pt \
--llm_num_virtual_tokens 8 \
--gnn_model_name hyper \
--gnn_num_layers 1 \
--patience 3