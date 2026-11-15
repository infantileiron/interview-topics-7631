"""离线批量检索测试：直接计算 oracle、顺序/分批不变性、空词向量与并列规则。

运行（后端目录下，纯标准库）：
  PYTHONPATH=. python3 -m unittest tests.test_batch_retrieval -v
"""
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from src.services.batch_retrieval import (  # noqa: E402
    LongestMatchTokenizer, prepare_lexicon, run_batch_retrieval,
    _spool_decode, _spool_encode,
)
from src.services.tokenize import tokenize as original_tokenize  # noqa: E402

LEXICON = {
    'terms': ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta',
              'eta', 'theta', 'iota', 'kappa', '的'],
    'stopwords': ['的'],
}


def direct_tfidf_neighbors(corpora, lexicon, limit=3):
    """纯 Python 直接计算 oracle：复用既有分词，构造稠密 TF-IDF 后两两余弦。

    仅用于小样本对照（测试内允许稠密矩阵），生产批量路径不构造它们。
    近邻只取其他语料，并列规则与批量实现一致：
    round(cos, 12) 同值按 (corpus_id, doc_id) 升序，零相似补齐到三篇。
    """
    terms, term_set, stop_set = prepare_lexicon(lexicon)
    docs = []  # (corpus_id, doc_id, counts, token_count, unknown)
    isolated = []
    for corpus in corpora:
        for row in corpus['rows']:
            tok = original_tokenize(row['text'], lexicon['terms'],
                                    lexicon['stopwords'])
            if not tok['tokens']:
                isolated.append((corpus['corpus_id'], row['doc_id']))
                continue
            docs.append((corpus['corpus_id'], row['doc_id'],
                         tok['counts'], len(tok['tokens']),
                         tok['unknown_characters']))
    vocabulary = sorted({t for _, _, counts, _, _ in docs for t in counts})
    n = len(docs)
    df = {t: sum(1 for _, _, counts, _, _ in docs if t in counts)
          for t in vocabulary}
    idf = {t: math.log((1 + n) / (1 + df[t])) + 1.0 for t in vocabulary}
    vectors, norms = [], []
    for _, _, counts, token_count, _ in docs:
        vec = [(j, counts.get(t, 0) / token_count * idf[t])
               for j, t in enumerate(vocabulary) if counts.get(t)]
        vectors.append(vec)
        norms.append(math.sqrt(sum(w * w for _, w in vec)))

    def cosine(a, b):
        da = dict(vectors[a])
        return sum(w * da.get(j, 0.0) for j, w in vectors[b]) / \
            (norms[a] * norms[b])

    result = {}
    for i, (cid, did, _, _, _) in enumerate(docs):
        scored = []
        for j, (cjd, djd, _, _, _) in enumerate(docs):
            if j == i or cjd == cid:
                continue
            scored.append((cjd, djd, cosine(i, j)))
        scored.sort(key=lambda item: (-round(item[2], 12), item[0], item[1]))
        chosen = [{'corpus_id': c, 'doc_id': d, 'similarity': s,
                   'zero_similarity': s == 0.0}
                  for c, d, s in scored[:limit]]
        picked = {(c['corpus_id'], c['doc_id']) for c in chosen}
        # 零相似补齐
        others = sorted(((cjd, djd) for j, (cjd, djd, _, _, _) in enumerate(docs)
                         if cjd != cid and (cjd, djd) not in picked))
        while len(chosen) < limit and others:
            c, d = others.pop(0)
            chosen.append({'corpus_id': c, 'doc_id': d,
                           'similarity': 0.0, 'zero_similarity': True})
        result[(cid, did)] = chosen
    return result, vocabulary, df, idf, sorted(isolated), docs


def build_corpora():
    return [
        {'corpus_id': 'batchA', 'rows': [
            {'doc_id': 'a1', 'text': 'alpha beta gamma alpha 的 delta'},
            {'doc_id': 'a2', 'text': 'gamma gamma delta epsilon'},
            {'doc_id': 'a3', 'text': 'zeta eta theta iota kappa'},
            {'doc_id': 'a4', 'text': '只有停用词 的 的 的'},
        ]},
        {'corpus_id': 'batchB', 'rows': [
            {'doc_id': 'b1', 'text': 'alpha alpha beta delta 的'},
            {'doc_id': 'b2', 'text': 'zeta eta theta iota'},
            {'doc_id': 'b3', 'text': 'gamma delta epsilon epsilon'},
            {'doc_id': 'b4', 'text': '无任何受控词项 xyz 123'},
        ]},
        {'corpus_id': 'batchC', 'rows': [
            {'doc_id': 'c1', 'text': 'alpha beta gamma delta epsilon zeta'},
            {'doc_id': 'c2', 'text': 'iota kappa eta theta'},
        ]},
    ]


class BatchRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='br-test-')
        self.corpora = build_corpora()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---- 分词器与既有实现语义一致 ----
    def test_tokenizer_matches_original(self):
        trie = LongestMatchTokenizer(LEXICON['terms'], set(LEXICON['stopwords']))
        samples = [
            'alpha beta gamma alpha 的 delta',
            'alphabetagamma', 'ALPHA  Beta　的',  # 全角空格/大小写
            'zetaeta未知iota xyz κ', '的的的', '',
            'epsilon123theta', 'delta delta, delta.',
        ]
        for text in samples:
            got = trie.tokenize(text)
            want = original_tokenize(text, LEXICON['terms'],
                                     LEXICON['stopwords'])
            self.assertEqual(got['tokens'], want['tokens'], text)
            self.assertEqual(got['counts'], want['counts'], text)
            self.assertEqual(got['unknown_characters'],
                             want['unknown_characters'], text)
            self.assertEqual(got['stopword_matches'],
                             want['stopword_matches'], text)

    def test_spool_roundtrip(self):
        counts = {'alpha': 2, 'beta': 1, '的': 3}
        line = _spool_encode('c', 'd', 6, 2, counts)
        cid, did, token_count, unknown, back = _spool_decode(line + '\n')
        self.assertEqual((cid, did, token_count, unknown, back),
                         ('c', 'd', 6, 2, counts))

    # ---- 小样本与直接计算一致 ----
    def test_matches_direct_oracle(self):
        result = run_batch_retrieval(self.corpora, LEXICON, batch_size=2)
        oracle, vocabulary, df, idf, isolated, docs = \
            direct_tfidf_neighbors(self.corpora, LEXICON)
        self.assertEqual(result['vocabulary'], vocabulary)
        self.assertEqual(result['df'], [df[t] for t in vocabulary])
        for got, want_idf in zip(result['idf'],
                                 (idf[t] for t in vocabulary)):
            self.assertTrue(math.isclose(got, want_idf, rel_tol=1e-12,
                                         abs_tol=1e-12))
        self.assertEqual(len(result['neighbors']), len(docs))
        for item in result['neighbors']:
            key = (item['corpus_id'], item['doc_id'])
            expected = oracle[key]
            self.assertEqual(len(item['neighbors']), 3)
            for got, want in zip(item['neighbors'], expected):
                self.assertEqual(got['corpus_id'], want['corpus_id'])
                self.assertEqual(got['doc_id'], want['doc_id'])
                self.assertTrue(math.isclose(
                    got['similarity'], want['similarity'],
                    rel_tol=1e-12, abs_tol=1e-12),
                    f'{key} vs {want}')
                self.assertEqual(got.get('zero_similarity', False),
                                 want.get('zero_similarity', False))
        self.assertEqual([(x['corpus_id'], x['doc_id'])
                          for x in result['isolated']], isolated)
        for item in result['isolated']:
            self.assertEqual(item['reason'], '无有效受控词项（空词向量）')

    # ---- 输入顺序改变，结果不变 ----
    def test_input_order_invariance(self):
        base = run_batch_retrieval(self.corpora, LEXICON, batch_size=3)
        shuffled = [
            {'corpus_id': 'batchC', 'rows': list(reversed(self.corpora[2]['rows']))},
            {'corpus_id': 'batchA', 'rows': list(reversed(self.corpora[0]['rows']))},
            {'corpus_id': 'batchB', 'rows': list(reversed(self.corpora[1]['rows']))},
        ]
        other = run_batch_retrieval(shuffled, LEXICON, batch_size=5)
        self.assertEqual(
            json.dumps(base['neighbors'], sort_keys=True, ensure_ascii=False),
            json.dumps(other['neighbors'], sort_keys=True, ensure_ascii=False))
        self.assertEqual(base['vocabulary'], other['vocabulary'])
        self.assertEqual(base['df'], other['df'])
        self.assertEqual(base['idf'], other['idf'])
        self.assertEqual(
            json.dumps(base['isolated'], sort_keys=True, ensure_ascii=False),
            json.dumps(other['isolated'], sort_keys=True, ensure_ascii=False))

    # ---- 批大小不改变结果 ----
    def test_batch_size_invariance(self):
        first = run_batch_retrieval(self.corpora, LEXICON, batch_size=1)
        for size in (2, 7, 100):
            other = run_batch_retrieval(self.corpora, LEXICON, batch_size=size)
            self.assertEqual(
                json.dumps(first['neighbors'], sort_keys=True,
                           ensure_ascii=False),
                json.dumps(other['neighbors'], sort_keys=True,
                           ensure_ascii=False))

    # ---- 同语料文档绝不互为近邻 ----
    def test_never_same_corpus(self):
        result = run_batch_retrieval(self.corpora, LEXICON)
        for item in result['neighbors']:
            for neighbor in item['neighbors']:
                if neighbor['doc_id'] is not None:
                    self.assertNotEqual(neighbor['corpus_id'],
                                        item['corpus_id'])

    # ---- 零相似补齐：无共同词项时仍给三篇 ----
    def test_zero_similarity_fill(self):
        corpora = [
            {'corpus_id': 'x', 'rows': [
                {'doc_id': 'x1', 'text': 'alpha alpha alpha'},
                {'doc_id': 'x2', 'text': 'beta beta beta'}]},
            {'corpus_id': 'y', 'rows': [
                {'doc_id': 'y1', 'text': 'gamma gamma gamma'},
                {'doc_id': 'y2', 'text': 'delta delta delta'},
                {'doc_id': 'y3', 'text': 'epsilon epsilon epsilon'}]},
        ]
        result = run_batch_retrieval(corpora, LEXICON)
        for item in result['neighbors']:
            self.assertEqual(len(item['neighbors']), 3)
            if item['corpus_id'] == 'x':
                self.assertTrue(all(n['zero_similarity']
                                    for n in item['neighbors']))
                self.assertEqual([n['doc_id'] for n in item['neighbors']],
                                 ['y1', 'y2', 'y3'])

    # ---- 其他语料不足三篇：显式空位移交 ----
    def test_insufficient_other_corpus_slots(self):
        corpora = [
            {'corpus_id': 'x', 'rows': [{'doc_id': 'x1', 'text': 'alpha'}]},
            {'corpus_id': 'y', 'rows': [{'doc_id': 'y1', 'text': 'beta'}]},
        ]
        result = run_batch_retrieval(corpora, LEXICON)
        neighbors = result['neighbors'][0]['neighbors']
        self.assertEqual(len(neighbors), 3)
        self.assertEqual(neighbors[0]['doc_id'], 'y1')
        self.assertTrue(neighbors[0]['zero_similarity'])
        self.assertIsNone(neighbors[1]['doc_id'])
        self.assertIsNone(neighbors[2]['doc_id'])
        self.assertIn('不足', neighbors[1]['reason'])

    # ---- 并列：12 位舍入后同值按定位键升序 ----
    def test_tie_break_by_canonical_key(self):
        # 两个词项在全部有效文档同 df，构造对称文档使余弦完全相等
        corpora = [
            {'corpus_id': 'p', 'rows': [
                {'doc_id': 'p1', 'text': 'alpha beta'}]},
            {'corpus_id': 'q', 'rows': [
                {'doc_id': 'q9', 'text': 'alpha beta'},
                {'doc_id': 'q1', 'text': 'alpha beta'},
                {'doc_id': 'q5', 'text': 'alpha beta'}]},
        ]
        result = run_batch_retrieval(corpora, LEXICON)
        docs = [n['doc_id'] for n in result['neighbors'][0]['neighbors']]
        self.assertEqual(docs, ['q1', 'q5', 'q9'])
        sims = {n['doc_id']: n['similarity']
                for n in result['neighbors'][0]['neighbors']}
        self.assertTrue(all(math.isclose(v, 1.0) for v in sims.values()))

    # ---- 全空语料失败，部分空只隔离 ----
    def test_all_empty_raises(self):
        corpora = [{'corpus_id': 'x', 'rows': [
            {'doc_id': 'x1', 'text': 'xyz'}, {'doc_id': 'x2', 'text': '的 的'}]}]
        with self.assertRaises(ValueError):
            run_batch_retrieval(corpora, LEXICON)

    def test_partial_isolation_keeps_running(self):
        result = run_batch_retrieval(self.corpora, LEXICON)
        self.assertEqual(result['summary']['isolated_count'], 2)
        self.assertEqual(result['summary']['valid_document_count'], 8)

    # ---- 词典校验与既有口径一致 ----
    def test_lexicon_validation(self):
        with self.assertRaises(ValueError):
            prepare_lexicon({'terms': ['a b'], 'stopwords': []})
        with self.assertRaises(ValueError):
            prepare_lexicon({'terms': ['a', 'a'], 'stopwords': []})
        with self.assertRaises(ValueError):
            prepare_lexicon({'terms': ['a'], 'stopwords': ['b']})
        with self.assertRaises(ValueError):
            prepare_lexicon({'terms': ['a'], 'stopwords': ['a']})

    # ---- 重复编号拒绝 ----
    def test_duplicate_ids_rejected(self):
        bad = [{'corpus_id': 'x', 'rows': [
            {'doc_id': 'd1', 'text': 'alpha'},
            {'doc_id': 'd1', 'text': 'beta'}]}]
        with self.assertRaises(ValueError):
            run_batch_retrieval(bad, LEXICON)
        bad_corpus = [
            {'corpus_id': 'x', 'rows': [{'doc_id': 'd1', 'text': 'alpha'}]},
            {'corpus_id': 'x', 'rows': [{'doc_id': 'd2', 'text': 'beta'}]},
        ]
        with self.assertRaises(ValueError):
            run_batch_retrieval(bad_corpus, LEXICON)

    # ---- CSV 流式输入与行内输入等价 ----
    def test_csv_path_input_equivalent(self):
        csv_corpora = []
        for corpus in self.corpora:
            path = os.path.join(self.tmp, f"{corpus['corpus_id']}.csv")
            with open(path, 'w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['doc_id', 'text'])
                writer.writeheader()
                writer.writerows(corpus['rows'])
            csv_corpora.append({'corpus_id': corpus['corpus_id'],
                                'csv_path': path})
        inline = run_batch_retrieval(self.corpora, LEXICON)
        streamed = run_batch_retrieval(csv_corpora, LEXICON)
        self.assertEqual(
            json.dumps(inline['neighbors'], sort_keys=True, ensure_ascii=False),
            json.dumps(streamed['neighbors'], sort_keys=True,
                       ensure_ascii=False))

    # ---- CLI 端到端，输出文件与指标 ----
    def test_cli_end_to_end(self):
        paths = []
        for corpus in self.corpora:
            path = os.path.join(self.tmp, f"{corpus['corpus_id']}.csv")
            with open(path, 'w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['doc_id', 'text'])
                writer.writeheader()
                writer.writerows(corpus['rows'])
            paths.append((corpus['corpus_id'], path))
        lexicon_path = os.path.join(self.tmp, 'lexicon.json')
        with open(lexicon_path, 'w', encoding='utf-8') as handle:
            json.dump(LEXICON, handle, ensure_ascii=False)
        out_json = os.path.join(self.tmp, 'result.json')
        out_csv = os.path.join(self.tmp, 'neighbors.csv')
        # -I 会忽略 PYTHONPATH 与工作目录，-m 依赖 cwd 定位 scripts/src 包；
        # 输入均为本测试自行生成的临时文件，非外部下载数据。
        cmd = [sys.executable, '-m', 'scripts.batch_retrieve']
        for cid, path in paths:
            cmd += ['--corpus', f'{cid}={path}']
        cmd += ['--lexicon', lexicon_path, '--output', out_json,
                '--csv-output', out_csv, '--batch-size', '4',
                '--spool-dir', os.path.join(self.tmp, 'spool')]
        proc = subprocess.run(cmd, cwd=BACKEND_DIR, capture_output=True,
                              text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn('峰值内存', proc.stdout)
        with open(out_json, encoding='utf-8') as handle:
            result = json.load(handle)
        self.assertEqual(len(result['neighbors']), 8)
        self.assertIn('elapsed_seconds', result['summary'])
        self.assertIn('peak_rss_mb', result['summary'])
        with open(out_csv, encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 8 * 3)
        self.assertIn('neighbor_corpus_id', rows[0])

    # ---- 未使用词项与统一 IDF 跨语料口径 ----
    def test_unused_terms_and_global_idf(self):
        result = run_batch_retrieval(self.corpora, LEXICON)
        used = set(result['vocabulary'])
        self.assertEqual(result['unused_terms'],
                         sorted(set(LEXICON['terms'])
                                - set(LEXICON['stopwords']) - used))
        # IDF 使用全部语料有效文档总数 N=8
        n = result['summary']['valid_document_count']
        for term, df_value, idf_value in zip(result['vocabulary'],
                                             result['df'], result['idf']):
            self.assertTrue(math.isclose(
                idf_value, math.log((1 + n) / (1 + df_value)) + 1.0))


if __name__ == '__main__':
    unittest.main(verbosity=2)
