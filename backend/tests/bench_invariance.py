"""三万篇规模下的确定性验证：改变输入顺序与批大小，结果必须完全一致；
同时构造空词向量文档并验证隔离清单。纯标准库。

用法：PYTHONPATH=. python3 tests/bench_invariance.py
"""
import csv
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.services.batch_retrieval import run_batch_retrieval  # noqa: E402

PER_CORPUS = 5_000
TERMS = [f't{i:05d}' for i in range(2_000)]
STOPS = [f's{i:03d}' for i in range(50)]
LEXICON = {'terms': TERMS + STOPS, 'stopwords': STOPS}


def build_rows(seed):
    rng = random.Random(seed)
    rows = []
    for d in range(PER_CORPUS):
        if d % 500 == 7:  # 周期性空词向量文档（停用词/未知词）
            rows.append({'doc_id': f'doc-{d:05d}',
                         'text': '未知噪声 ' + ' '.join(rng.choices(STOPS, k=3))})
            continue
        n = rng.randint(1, 20)
        rows.append({'doc_id': f'doc-{d:05d}',
                     'text': ' '.join(rng.choices(TERMS, k=n))})
    return rows


def write_csv(path, rows):
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['doc_id', 'text'])
        writer.writeheader()
        writer.writerows(rows)


def main():
    work = Path(tempfile.mkdtemp(prefix='br-invariance-'))
    try:
        corpora, shuffled_corpora = [], []
        for c in range(3):
            rows = build_rows(seed=100 + c)
            p1 = work / f'c{c}-a.csv'
            write_csv(p1, rows)
            p2 = work / f'c{c}-b.csv'
            write_csv(p2, list(reversed(rows)))  # 文档顺序反转
            corpora.append({'corpus_id': f'corpus-{c:02d}', 'csv_path': str(p1)})
            shuffled_corpora.insert(0, {
                'corpus_id': f'corpus-{c:02d}', 'csv_path': str(p2)})

        r1 = run_batch_retrieval(corpora, LEXICON, batch_size=37,
                                 spool_dir=str(work / 'sp1'))
        r2 = run_batch_retrieval(shuffled_corpora, LEXICON, batch_size=129,
                                 spool_dir=str(work / 'sp2'))

        def fingerprint(result):
            return {
                'neighbors': json.dumps(result['neighbors'],
                                        sort_keys=True, ensure_ascii=False),
                'isolated': json.dumps(result['isolated'],
                                       sort_keys=True, ensure_ascii=False),
                'vocabulary': json.dumps(result['vocabulary'],
                                         ensure_ascii=False),
                'df': json.dumps(result['df']),
                'idf': json.dumps(result['idf']),
            }

        f1, f2 = fingerprint(r1), fingerprint(r2)
        ok = True
        for key in f1:
            same = f1[key] == f2[key]
            print(f"[{'PASS' if same else 'FAIL'}] 顺序/批大小不变: {key}")
            ok &= same

        isolated_keys = sorted((x['corpus_id'], x['doc_id'])
                               for x in r1['isolated'])
        expected_isolated = sorted(
            (f'corpus-{c:02d}', f'doc-{d:05d}')
            for c in range(3) for d in range(PER_CORPUS) if d % 500 == 7)
        iso_ok = isolated_keys == expected_isolated
        print(f"[{'PASS' if iso_ok else 'FAIL'}] 空词向量隔离 "
              f"{len(isolated_keys)} 篇，原因单列")
        ok &= iso_ok
        reason_ok = all(x['reason'] == '无有效受控词项（空词向量）'
                        for x in r1['isolated'])
        print(f"[{'PASS' if reason_ok else 'FAIL'}] 隔离原因字段一致")
        ok &= reason_ok

        valid = r1['summary']['valid_document_count']
        print(f"有效文档 {valid} / 输入 {r1['summary']['input_document_count']} "
              f"/ 隔离 {r1['summary']['isolated_count']}")
        print(f"峰值RSS {r1['summary']['peak_rss_mb']:.1f} MiB, "
              f"用时 {r1['summary']['elapsed_seconds']:.2f}s")
        # 抽样人工核对一条近邻结构
        sample = r1['neighbors'][0]
        print('样例:', json.dumps(sample, ensure_ascii=False)[:240])
        return 0 if ok else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
