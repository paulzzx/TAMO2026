CUDA_VISIBLE_DEVICES=0 python -m src.dataset.preprocess.structProbe_hyper
CUDA_VISIBLE_DEVICES=0 python -m src.dataset.preprocess.hitab_hyper
CUDA_VISIBLE_DEVICES=0 python -m src.dataset.preprocess.wtq_hyper
CUDA_VISIBLE_DEVICES=0 python -m src.dataset.preprocess.wikisql_hyper
CUDA_VISIBLE_DEVICES=0 python -m src.dataset.preprocess.fetaqa_hyper