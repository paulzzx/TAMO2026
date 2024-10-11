import datasets
from src.global_path import global_path

wikisql = datasets.load_dataset('wikisql')
wikisql.save_to_disk(f'{global_path}/dataset/wikisql/wikisql')

wtq = datasets.load_dataset('wikitablequestions')
wtq.save_to_disk(f'{global_path}/dataset/wtq/wikitablequestions')

fetaqa = datasets.load_dataset('DongfuTingle/FeTaQA')
fetaqa.save_to_disk(f'{global_path}/dataset/fetaqa/fetaqa')

hitab = datasets.load_dataset("kasnerz/hitab")
hitab.save_to_disk(f'{global_path}/dataset/hitab/hitab')