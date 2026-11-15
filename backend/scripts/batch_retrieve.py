"""离线批量相似访谈检索命令行入口（纯标准库，可离线运行）。

用法示例：
  python -m scripts.batch_retrieve \
      --corpus c1=interviews_a.csv --corpus c2=interviews_b.csv \
      --lexicon lexicon.json \
      --output result.json --csv-output neighbors.csv \
      --batch-size 64 --spool-dir /tmp/retrieval-spool

语料 CSV 列：doc_id,text（与既有数据集口径一致）。corpus_id 由
“--corpus <corpus_id>=<csv路径>”指定，用 (corpus_id, doc_id) 共同定位文档。
词典 JSON：{"terms": [...], "stopwords": [...]}。

结果 JSON 含统一 df/idf、每篇文档在其他语料中的三篇近邻、空词向量隔离清单，
以及用时与峰值内存（summary.elapsed_seconds / summary.peak_rss_mb）。
不构造稠密词向量或全量两两矩阵，查询分批完成。
"""
import argparse
import json
import sys
from pathlib import Path

from src.services.batch_retrieval import run_batch_retrieval


def _parse_corpus(spec):
    if '=' not in spec:
        raise argparse.ArgumentTypeError(
            f'语料参数须为 <corpus_id>=<csv路径>，收到：{spec}')
    corpus_id, path = spec.split('=', 1)
    corpus_id = corpus_id.strip()
    path = path.strip()
    if not corpus_id or not path:
        raise argparse.ArgumentTypeError(f'语料参数不完整：{spec}')
    if not Path(path).is_file():
        raise argparse.ArgumentTypeError(f'语料文件不存在：{path}')
    return {'corpus_id': corpus_id, 'csv_path': path}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='离线批量跨语料相似访谈检索（TF-IDF + 稀疏倒排余弦，内存有界）')
    parser.add_argument('--corpus', action='append', required=True,
                        type=_parse_corpus, metavar='CORPUS_ID=CSV_PATH',
                        help='语料文件，可重复传入多份；corpus_id 不可重复')
    parser.add_argument('--lexicon', required=True,
                        help='检索专用词典 JSON 路径（terms/stopwords）')
    parser.add_argument('--output', help='完整结果 JSON 输出路径')
    parser.add_argument('--csv-output', help='近邻结果 CSV 输出路径')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='查询分批大小，控制峰值内存；结果与批大小无关（默认64）')
    parser.add_argument('--spool-dir',
                        help='稀疏记录临时目录，默认系统临时目录')
    args = parser.parse_args(argv)

    lexicon_path = Path(args.lexicon)
    if not lexicon_path.is_file():
        parser.error(f'词典文件不存在：{args.lexicon}')
    with lexicon_path.open('r', encoding='utf-8') as handle:
        lexicon = json.load(handle)
    if not isinstance(lexicon, dict) or 'terms' not in lexicon:
        parser.error('词典 JSON 须包含 terms 与 stopwords 字段')
    lexicon.setdefault('stopwords', [])

    result = run_batch_retrieval(
        args.corpus, lexicon,
        batch_size=args.batch_size,
        spool_dir=args.spool_dir,
        output_path=args.output,
        csv_output_path=args.csv_output)

    summary = result['summary']
    print('离线批量检索完成')
    print(f"  语料数            : {summary['corpus_count']}")
    print(f"  输入文档/有效/隔离: {summary['input_document_count']} / "
          f"{summary['valid_document_count']} / {summary['isolated_count']}")
    print(f"  统一词表规模      : {summary['vocabulary_size']}")
    print(f"  分批大小          : {summary['batch_size']}")
    print(f"  总用时(秒)        : {summary['elapsed_seconds']:.4f}")
    for phase, seconds in summary['phase_seconds'].items():
        print(f'    - {phase:<18}: {seconds:.4f}s')
    print(f"  峰值内存(RSS, MiB): {summary['peak_rss_mb']:.2f}")
    print(f"  临时稀疏落盘字节  : {summary['spool_bytes']}")
    if args.output:
        print(f"  结果 JSON         : {args.output}")
    if args.csv_output:
        print(f"  近邻 CSV          : {args.csv_output}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
