"""离线批量相似访谈检索（跨语料、流式、内存有界）。

设计约束（见离线批量检索需求）：
- 接收多份语料与检索专用词典，统一计算词频(TF)与逆文档频率(IDF)。
- 每篇有效文档只在“其他语料”中返回最相近的三篇，用 (corpus_id, doc_id) 共同定位；
  沿用既有口径：余弦相似度、相似度 12 位四舍五入后同值按编号排序，零相似也补齐到三篇。
- 不构造稠密词向量、不构造全量 N×N 两两矩阵；倒排索引 + 查询分批，内存随
  批大小与稀疏非零项有界。三万篇、两万词、每篇至多二十个有效词可在 256MiB 内分批完成。
- 纯标准库实现，不依赖 NumPy。
- 分词语义与 src/services/tokenize.py 完全一致（NFKC/lower 后最长词表匹配，
  同长度按词典序——本实现以“按长度分组的集合查找”复现同一规则）。

注意：本模块为新增离线入口，不改动既有 TF-IDF/NMF 分解配额、结果结构与任何已有文件。
"""
import array
import csv
import io
import json
import math
import os
import re
import resource
import tempfile
import time
import unicodedata

REASON_EMPTY = '无有效受控词项（空词向量）'
REASON_ZERO_FILL = '无共同有效词项，零相似补齐'
REASON_NO_OTHER_CORPUS = '其他语料有效文档不足，空位补齐'
NEIGHBOR_LIMIT = 3
_TERM_RE = re.compile(r'[a-z0-9一-龥]{1,16}')
_UNIT_SEP = '\x1f'
_REC_SEP = '\x1e'


# ---------------------------------------------------------------- 词典与分词
def normalize_term(term):
    return unicodedata.normalize('NFKC', term).strip().lower()


def prepare_lexicon(lexicon):
    """规范化并校验检索词典，规则与 src/services/lexicon.py 保持一致。"""
    terms = [normalize_term(t) for t in lexicon['terms']]
    stopwords = [normalize_term(s) for s in lexicon['stopwords']]
    if any(not _TERM_RE.fullmatch(t) for t in terms):
        raise ValueError('词项须为1—16个中英文字母/数字，无空格标点')
    if len(set(terms)) != len(terms) or len(set(stopwords)) != len(stopwords):
        raise ValueError('规范化后词项或停用词重复')
    stop_set = set(stopwords)
    if not stop_set.issubset(set(terms)):
        raise ValueError('停用词须来自受控词表')
    if not set(terms) - stop_set:
        raise ValueError('移除停用词后词表为空')
    return terms, set(terms), stop_set


class LongestMatchTokenizer:
    """最长词表匹配分词器，语义等价于 src/services/tokenize.py:tokenize。

    原实现每步对 sorted(terms, key=(-len, term)) 做 startswith 线性扫描；
    本实现按词长从大到小在集合中查找同长度候选，命中即最长匹配，
    同长度仅可能命中一个字符串（集合成员判定），规则与结果完全一致，
    但复杂度由 O(文本长 × 词表长) 降为 O(文本长 × 词长种类)，可支撑两万词词典。
    """

    def __init__(self, terms, stop_set):
        by_length = {}
        for term in terms:
            by_length.setdefault(len(term), set()).add(term)
        # 长度从大到小：保证先命中更长词；同长度集合查找等价词典序下的唯一命中
        self._lengths = sorted(by_length, reverse=True)
        self._by_length = [by_length[length] for length in self._lengths]
        self._stop = stop_set

    def tokenize(self, text):
        text = unicodedata.normalize('NFKC', text).lower()
        tokens = []
        unknown = 0
        stopped = 0
        i = 0
        n = len(text)
        lengths = self._lengths
        groups = self._by_length
        stop = self._stop
        while i < n:
            matched = None
            for k, length in enumerate(lengths):
                if length <= n - i and text[i:i + length] in groups[k]:
                    matched = text[i:i + length]
                    break
            if matched is not None:
                if matched in stop:
                    stopped += 1
                else:
                    tokens.append(matched)
                i += len(matched)
            else:
                if text[i].isalnum():
                    unknown += 1
                i += 1
        counts = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        return {'tokens': tokens, 'counts': counts,
                'unknown_characters': unknown, 'stopword_matches': stopped}


# ---------------------------------------------------------------- 输入校验
def _validate_identifier(value, label):
    value = (value or '').strip()
    if not value:
        raise ValueError(f'{label}不能为空')
    if any(ch in value for ch in ('\t', '\n', '\r', _UNIT_SEP, _REC_SEP)):
        raise ValueError(f'{label}不能包含制表符/换行/控制字符')
    return value


def _iter_corpus_rows(corpus):
    """统一语料输入：{'corpus_id', 'rows': [{'doc_id','text'}, ...]} 或
    {'corpus_id', 'csv_path'}（流式，内存更省）。"""
    corpus_id = _validate_identifier(corpus.get('corpus_id'), 'corpus_id')
    if 'rows' in corpus:
        for row in corpus['rows']:
            yield corpus_id, (row.get('doc_id') or '').strip(), (row.get('text') or '').strip()
        return
    path = corpus.get('csv_path')
    if not path:
        raise ValueError('语料须提供 rows 或 csv_path')
    with open(path, 'r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {'doc_id', 'text'}.issubset(reader.fieldnames):
            raise ValueError(f'语料{corpus_id}的CSV需要doc_id,text列')
        for row in reader:
            yield corpus_id, (row.get('doc_id') or '').strip(), (row.get('text') or '').strip()


def _spool_encode(corpus_id, doc_id, token_count, unknown, counts):
    parts = [f'{term}={count}' for term, count in sorted(counts.items())]
    return _UNIT_SEP.join([corpus_id, doc_id, str(token_count),
                           str(unknown), _REC_SEP.join(parts)])


def _spool_decode(line):
    corpus_id, doc_id, token_count, unknown, packed = line.rstrip('\n').split(_UNIT_SEP)
    counts = {}
    if packed:
        for piece in packed.split(_REC_SEP):
            term, count = piece.split('=', 1)
            counts[term] = int(count)
    return corpus_id, doc_id, int(token_count), int(unknown), counts


# ---------------------------------------------------------------- 主流程
def run_batch_retrieval(corpora, lexicon, *, batch_size=64, spool_dir=None,
                        output_path=None, csv_output_path=None):
    """离线批量跨语料相似访谈检索。

    参数
    ----
    corpora: 多个语料，元素为 {'corpus_id': str, 'rows': [...]} 或
             {'corpus_id': str, 'csv_path': str}；每个 doc_id 在语料内唯一。
    lexicon: {'terms': [...], 'stopwords': [...]}，检索专用受控词典。
    batch_size: 查询分批大小，控制峰值内存；结果与批大小无关。
    spool_dir: 临时稀疏记录目录，None 时使用系统临时目录。
    output_path / csv_output_path: 可选结果落盘路径（JSON / CSV）。

    返回可 JSON 序列化的结果字典，含 neighbors、isolated、df/idf 与性能指标。
    """
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size 须为正整数')
    if not corpora:
        raise ValueError('至少需要一份语料')
    started = time.perf_counter()
    phases = {}
    terms, term_set, stop_set = prepare_lexicon(lexicon)
    tokenizer = LongestMatchTokenizer(terms, stop_set)

    # ---- Pass 1：流式分词，稀疏记录落盘，累计 df；不把全文与稠密向量留在内存 ----
    phase_start = time.perf_counter()
    work_dir = spool_dir or tempfile.mkdtemp(prefix='batch-retrieval-')
    os.makedirs(work_dir, exist_ok=True)
    spool_path = os.path.join(work_dir, 'sparse_spool.tsv')
    seen_corpus = set()
    df = {}
    isolated = []
    # 清单只保存定位与轻量计数；偏移量用于 Pass2/3 随机回读
    manifest_corpus = []
    manifest_doc = []
    manifest_offset = array.array('q')
    manifest_tokens = array.array('i')
    manifest_unknown = array.array('i')
    total_input_docs = 0
    with open(spool_path, 'w', encoding='utf-8') as spool:
        for corpus in corpora:
            corpus_id = _validate_identifier(corpus.get('corpus_id'), 'corpus_id')
            if corpus_id in seen_corpus:
                raise ValueError(f'corpus_id 重复：{corpus_id}')
            seen_corpus.add(corpus_id)
            seen_doc = set()
            for cid, doc_id, text in _iter_corpus_rows(corpus):
                if not doc_id:
                    raise ValueError(f'语料{cid}存在空 doc_id')
                if doc_id in seen_doc:
                    raise ValueError(f'语料{cid}的 doc_id 重复：{doc_id}')
                if not text:
                    raise ValueError(f'{cid}/{doc_id} 文本为空')
                seen_doc.add(doc_id)
                total_input_docs += 1
                tok = tokenizer.tokenize(text)
                if not tok['tokens']:
                    isolated.append({
                        'corpus_id': cid, 'doc_id': doc_id, 'reason': REASON_EMPTY,
                        'unknown_characters': tok['unknown_characters'],
                        'stopword_matches': tok['stopword_matches'],
                    })
                    continue
                offset = spool.tell()
                spool.write(_spool_encode(
                    cid, doc_id, len(tok['tokens']),
                    tok['unknown_characters'], tok['counts']) + '\n')
                manifest_corpus.append(cid)
                manifest_doc.append(doc_id)
                manifest_offset.append(offset)
                manifest_tokens.append(len(tok['tokens']))
                manifest_unknown.append(tok['unknown_characters'])
                for term in tok['counts']:
                    df[term] = df.get(term, 0) + 1
    n_valid = len(manifest_doc)
    if n_valid == 0:
        raise ValueError('全部文档为空词向量，不能计算TF-IDF/余弦；请复核词表与停用词')
    phases['tokenize_spool'] = time.perf_counter() - phase_start

    # 规范序（corpus_id, doc_id）：所有后续索引、倒排链与输出均与输入顺序无关
    phase_start = time.perf_counter()
    canonical_order = sorted(range(n_valid),
                             key=lambda k: (manifest_corpus[k], manifest_doc[k]))
    vocabulary = sorted(df)
    term_id = {term: j for j, term in enumerate(vocabulary)}
    n = n_valid
    idf = [math.log((1 + n) / (1 + df[term])) + 1.0 for term in vocabulary]
    idf_by_term = dict(zip(vocabulary, idf))

    # ---- Pass 2：按规范序回读稀疏记录，构建倒排索引（term -> [(doc, 权重)]）----
    corpus_ids = []
    corpus_index = {}
    doc_corpus = array.array('i')       # 规范文档下标 -> 语料下标
    doc_ids = [None] * n
    unknowns = array.array('i')
    norms = array.array('d')
    postings_idx = [array.array('i') for _ in vocabulary]
    postings_w = [array.array('d') for _ in vocabulary]
    per_corpus_docs = {}
    with open(spool_path, 'r', encoding='utf-8') as spool:
        for new_idx, old_idx in enumerate(canonical_order):
            spool.seek(manifest_offset[old_idx])
            _, _, token_count, unknown, counts = _spool_decode(spool.readline())
            cid = manifest_corpus[old_idx]
            did = manifest_doc[old_idx]
            if cid not in corpus_index:
                corpus_index[cid] = len(corpus_ids)
                corpus_ids.append(cid)
                per_corpus_docs[corpus_index[cid]] = []
            cidx = corpus_index[cid]
            doc_corpus.append(cidx)
            doc_ids[new_idx] = did
            unknowns.append(unknown)
            per_corpus_docs[cidx].append(new_idx)
            square_sum = 0.0
            for term, count in counts.items():
                weight = (count / token_count) * idf_by_term[term]
                square_sum += weight * weight
                tid = term_id[term]
                postings_idx[tid].append(new_idx)
                postings_w[tid].append(weight)
            norms.append(math.sqrt(square_sum))
    # 语料下标也按 corpus_id 排序，保证零相似补齐顺序确定
    sorted_corpus_idx = sorted(range(len(corpus_ids)), key=lambda k: corpus_ids[k])
    phases['build_postings'] = time.perf_counter() - phase_start

    # ---- Pass 3：查询分批，借倒排索引只累计共享词项，绝不生成稠密/全量矩阵 ----
    phase_start = time.perf_counter()
    neighbors_by_key = {}
    spool_handle = open(spool_path, 'r', encoding='utf-8')
    try:
        for batch_start in range(0, n, batch_size):
            batch = list(range(batch_start, min(batch_start + batch_size, n)))
            # term_id -> [(批内查询位, 查询权重)]；词项按 id 排序，浮点累加序与输入无关
            query_terms = {}
            for local, gi in enumerate(batch):
                spool_handle.seek(manifest_offset[canonical_order[gi]])
                _, _, token_count, _, counts = _spool_decode(spool_handle.readline())
                for term, count in counts.items():
                    wi = (count / token_count) * idf_by_term[term]
                    query_terms.setdefault(term_id[term], []).append((local, wi))
            accumulators = [dict() for _ in batch]
            own_corpus = [doc_corpus[gi] for gi in batch]
            for tid in sorted(query_terms):
                qlist = query_terms[tid]
                p_idx = postings_idx[tid]
                p_w = postings_w[tid]
                for k in range(len(p_idx)):
                    j = p_idx[k]
                    wj = p_w[k]
                    cj = doc_corpus[j]
                    for local, wi in qlist:
                        if cj != own_corpus[local]:
                            bucket = accumulators[local]
                            bucket[j] = bucket.get(j, 0.0) + wi * wj
            for local, gi in enumerate(batch):
                neighbors_by_key[(corpus_ids[doc_corpus[gi]], doc_ids[gi])] = \
                    _select_neighbors(gi, accumulators[local], doc_corpus,
                                      own_corpus[local], doc_ids, corpus_ids,
                                      sorted_corpus_idx, per_corpus_docs, norms)
    finally:
        spool_handle.close()
    phases['retrieve_neighbors'] = time.perf_counter() - phase_start

    # ---- 组装规范序输出 ----
    neighbors = [neighbors_by_key[(corpus_ids[doc_corpus[gi]], doc_ids[gi])]
                 for gi in range(n)]
    isolated.sort(key=lambda item: (item['corpus_id'], item['doc_id']))
    unused_terms = sorted((term_set - stop_set) - set(vocabulary))
    spool_bytes = os.path.getsize(spool_path)
    elapsed = time.perf_counter() - started
    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # Linux: KiB
    result = {
        'summary': {
            'corpus_count': len(corpus_ids),
            'input_document_count': total_input_docs,
            'valid_document_count': n_valid,
            'isolated_count': len(isolated),
            'vocabulary_size': len(vocabulary),
            'neighbor_limit': NEIGHBOR_LIMIT,
            'batch_size': batch_size,
            'elapsed_seconds': elapsed,
            'phase_seconds': phases,
            'peak_rss_mb': peak_rss_kb / 1024.0,
            'spool_bytes': spool_bytes,
        },
        'corpora': [
            {'corpus_id': corpus_ids[cidx],
             'valid_document_count': len(per_corpus_docs[cidx])}
            for cidx in sorted_corpus_idx
        ],
        'vocabulary': vocabulary,
        'df': [df[term] for term in vocabulary],
        'idf': idf,
        'unused_terms': unused_terms,
        'neighbors': neighbors,
        'isolated': isolated,
        'method': {
            'tf': '词频/该文档有效非停用token总数',
            'idf': 'ln((1+N)/(1+df))+1，N 为全部语料有效文档总数',
            'similarity': '稀疏倒排累计的TF-IDF行L2归一余弦；仅其他语料参与',
            'tie_break': '相似度12位四舍五入后同值，按(corpus_id, doc_id)升序',
            'zero_fill': '不足三篇时用其他语料零相似文档补齐，仍不足则空位移交',
            'memory': '无稠密词向量、无全量两两矩阵；倒排索引+分批累计，spool临时落盘',
        },
    }
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as handle:
            json.dump(result, handle, ensure_ascii=False)
    if csv_output_path:
        _write_neighbor_csv(result, csv_output_path)
    return result


def _select_neighbors(gi, scores, doc_corpus, own_cidx, doc_ids, corpus_ids,
                      sorted_corpus_idx, per_corpus_docs, norms):
    """沿用既有余弦与并列规则选前三；排除本语料，零相似补齐。"""
    entries = []
    selected = set()
    # 与 src/services/tfidf.py:cosine_neighbors 相同的 12 位舍入并列口径，
    # 跨语料定位键扩展为 (corpus_id, doc_id)；先算余弦再按舍入值排序。
    cosine = {j: dot / (norms[gi] * norms[j]) for j, dot in scores.items()}
    ordered = sorted(
        cosine.items(),
        key=lambda item: (-round(item[1], 12),
                          corpus_ids[doc_corpus[item[0]]], doc_ids[item[0]]))
    for j, sim in ordered:
        if len(entries) >= NEIGHBOR_LIMIT:
            break
        entries.append({
            'corpus_id': corpus_ids[doc_corpus[j]],
            'doc_id': doc_ids[j],
            'similarity': sim,
            'zero_similarity': False,
        })
        selected.add(j)
    # 零相似补齐：其他语料按 corpus_id、doc_id 规范序取未入选文档
    if len(entries) < NEIGHBOR_LIMIT:
        for cidx in sorted_corpus_idx:
            if cidx == own_cidx:
                continue
            for j in per_corpus_docs[cidx]:
                if j in selected:
                    continue
                entries.append({
                    'corpus_id': corpus_ids[cidx],
                    'doc_id': doc_ids[j],
                    'similarity': 0.0,
                    'zero_similarity': True,
                    'reason': REASON_ZERO_FILL,
                })
                selected.add(j)
                if len(entries) == NEIGHBOR_LIMIT:
                    break
            if len(entries) == NEIGHBOR_LIMIT:
                break
    # 其他语料有效文档总数不足三篇：空位显式移交，不伪造定位
    while len(entries) < NEIGHBOR_LIMIT:
        entries.append({
            'corpus_id': None, 'doc_id': None,
            'similarity': 0.0, 'zero_similarity': True,
            'reason': REASON_NO_OTHER_CORPUS,
        })
    return {
        'corpus_id': corpus_ids[own_cidx],
        'doc_id': doc_ids[gi],
        'neighbors': entries,
    }


def _write_neighbor_csv(result, path):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(['corpus_id', 'doc_id', 'rank',
                     'neighbor_corpus_id', 'neighbor_doc_id',
                     'cosine_similarity', 'zero_similarity', 'reason'])
    for item in result['neighbors']:
        for rank, neighbor in enumerate(item['neighbors'], 1):
            writer.writerow([
                item['corpus_id'], item['doc_id'], rank,
                neighbor['corpus_id'] or '', neighbor['doc_id'] or '',
                neighbor['similarity'],
                '1' if neighbor['zero_similarity'] else '0',
                neighbor.get('reason', ''),
            ])
    with open(path, 'w', encoding='utf-8-sig', newline='') as handle:
        handle.write(buffer.getvalue())
