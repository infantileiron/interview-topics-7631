"""规模压测：三万篇、两万词词典、每篇至多二十个有效词，256MiB 内存约束验证。

在独立临时目录生成合成语料（与源码目录隔离），调用离线批量检索，
校验峰值 RSS 与输出指标。纯标准库。

用法（backend 目录下）：
  PYTHONPATH=. python3 tests/bench_scale.py
"""
import csv
import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.batch_retrieval import run_batch_retrieval  # noqa: E402

DOCS_PER_CORPUS = 10_000
CORPUS_COUNT = 3
TOTAL_DOCS = DOCS_PER_CORPUS * CORPUS_COUNT      # 30000
ACTIVE_TERMS = 19_500
STOPWORDS = 500
LEXICON_SIZE = ACTIVE_TERMS + STOPWORDS          # 20000
MAX_VALID_TOKENS = 20
MEMORY_CAP_MIB = 256.0


def write_lexicon(path):
    terms = [f't{i:05d}' for i in range(ACTIVE_TERMS)]
    stopwords = [f's{i:03d}' for i in range(STOPWORDS)]
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump({'terms': terms + stopwords, 'stopwords': stopwords},
                  handle, ensure_ascii=False)
    return terms, stopwords


def write_corpora(data_dir, terms, stopwords):
    rng = random.Random(20261115)
    corpora = []
    for c in range(CORPUS_COUNT):
        cid = f'corpus-{c:02d}'
        path = data_dir / f'{cid}.csv'
        with open(path, 'w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['doc_id', 'text'])
            for d in range(DOCS_PER_CORPUS):
                valid_n = rng.randint(1, MAX_VALID_TOKENS)
                stop_n = rng.randint(0, 3)
                tokens = [rng.choice(terms) for _ in range(valid_n)]
                tokens += [rng.choice(stopwords) for _ in range(stop_n)]
                rng.shuffle(tokens)
                # 插入词典外噪声，验证 unknown_characters 计数路径
                if d % 97 == 0:
                    tokens.append('未知词xyz')
                writer.writerow([f'{cid}-doc-{d:05d}', ' '.join(tokens)])
        corpora.append({'corpus_id': cid, 'csv_path': str(path)})
    return corpora


def main():
    data_dir = Path(tempfile.mkdtemp(prefix='br-scale-data-'))
    spool_dir = Path(tempfile.mkdtemp(prefix='br-scale-spool-'))
    try:
        lexicon_path = data_dir / 'lexicon.json'
        terms, stopwords = write_lexicon(lexicon_path)
        print(f'合成数据目录: {data_dir}')
        t0 = time.perf_counter()
        corpora = write_corpora(data_dir, terms, stopwords)
        print(f'语料生成用时: {time.perf_counter() - t0:.2f}s '
              f'({TOTAL_DOCS} 篇, 词典 {LEXICON_SIZE} 词)')
        with open(lexicon_path, encoding='utf-8') as handle:
            lexicon = json.load(handle)

        started = time.perf_counter()
        result = run_batch_retrieval(
            corpora, lexicon, batch_size=128,
            spool_dir=str(spool_dir),
            output_path=str(data_dir / 'result.json'),
            csv_output_path=str(data_dir / 'neighbors.csv'))
        wall = time.perf_counter() - started
        summary = result['summary']

        checks = []
        checks.append(('输入文档=30000',
                       summary['input_document_count'] == TOTAL_DOCS))
        checks.append(('有效文档=30000',
                       summary['valid_document_count'] == TOTAL_DOCS))
        checks.append(('每篇返回3近邻',
                       all(len(x['neighbors']) == 3
                           for x in result['neighbors'])))
        checks.append(('近邻均来自其他语料',
                       all(n['corpus_id'] != x['corpus_id']
                           for x in result['neighbors']
                           for n in x['neighbors']
                           if n['doc_id'] is not None)))
        checks.append(('定位键含corpus_id与doc_id',
                       all(set(['corpus_id', 'doc_id']).issubset(n)
                           for x in result['neighbors'] for n in x['neighbors'])))
        checks.append(('词典规模=20000', len(lexicon['terms']) == LEXICON_SIZE))
        checks.append((f"峰值RSS {summary['peak_rss_mb']:.1f}MiB "
                       f"<= {MEMORY_CAP_MIB:.0f}MiB",
                       summary['peak_rss_mb'] <= MEMORY_CAP_MIB))
        checks.append(('输出neighbors条数=30000',
                       len(result['neighbors']) == TOTAL_DOCS))

        print('\n================ 规模压测结果 ================')
        for label, ok in checks:
            print(f"[{'PASS' if ok else 'FAIL'}] {label}")
        print('----------------------------------------------')
        print(f"总墙钟用时(含结果写盘) : {wall:.2f}s")
        print(f"服务自计时             : {summary['elapsed_seconds']:.2f}s")
        for phase, seconds in summary['phase_seconds'].items():
            print(f'  - {phase:<20}: {seconds:.2f}s')
        print(f"峰值内存 RSS           : {summary['peak_rss_mb']:.2f} MiB "
              f"(上限 {MEMORY_CAP_MIB:.0f} MiB)")
        print(f"临时稀疏落盘           : {summary['spool_bytes'] / 1e6:.1f} MB")
        print(f"统一词表规模/df条目    : {summary['vocabulary_size']}")
        print(f"隔离文档数             : {summary['isolated_count']}")
        print('==============================================')
        return 0 if all(ok for _, ok in checks) else 1
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)
        shutil.rmtree(spool_dir, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
