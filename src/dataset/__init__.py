from src.dataset.wtq import WTQDatasetOrig, WTQDatasetPermute
from src.dataset.wikisql import WikiSQLDataset
from src.dataset.fetaqa import FetaQADataset, FetaQADatasetNohc
from src.dataset.structProbe import StructProbeDataset, StructProbeDatasetPermute
from src.dataset.hitab import HiTabDataset

load_dataset = {
    'wtq_orig': WTQDatasetOrig,
    'wtq_permute': WTQDatasetPermute,
    'wikisql': WikiSQLDataset,
    'fetaqa': FetaQADataset,
    'fetaqa_nohc': FetaQADatasetNohc,
    'structprobe': StructProbeDataset,
    'structprobe_permute': StructProbeDatasetPermute,
    'hitab': HiTabDataset,
}
