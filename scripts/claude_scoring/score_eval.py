"""Score regex-rule eval runs.

Re-executes each entry's `regex` against the eval universe per rule and
computes TPR/FPR. Handles two dataset schemas:

  retrospective (flat) - {true_positives, true_negatives}
    eval universe per rule: (tps - {seed}) for TPR, tns for FPR
  realtime (split)     - {true_positives_train, true_positives_test,
                          true_negatives_train, true_negatives_test, _split_meta}
    eval universe per rule: tps_test for TPR, tns_test for FPR
    (seed lives in train and is naturally excluded from the test halves)

Schema is autodetected per dataset file. dataset_dir and mapping_file may be
overridden when the dataset/mapping live outside the run root (realtime).
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent

# --- TEMPORARY scoring-time label override (remove after Claude runs are regenerated
# against corrected dataset/final labels). Two log strings hardcoded (verified against
# dataset/final/linux_auditd_disable_or_modify_system_firewall.json on 2026-06-03);
# at scoring time moves them TN→TP for one rule only when --apply-label-overrides is
# set. Delete this block + CLI flag + call-site args when no longer needed.
_FW_RULE = 'linux_auditd_disable_or_modify_system_firewall'
_FW_FORCE_POSITIVE = frozenset({
    'type=SERVICE_STOP msg=audit(1722965483.442:15704): pid=1 uid=0 auid=4294967295 ses=4294967295 subj=unconfined msg=\'unit=firewalld comm="systemd" exe="/usr/lib/systemd/systemd" hostname=? addr=? terminal=? res=success\'\x1dUID="root" AUID="unset"',
    'type=SERVICE_STOP msg=audit(1722965483.442:15704): pid=1 uid=0 auid=4294967295 ses=4294967295 subj=unconfined msg=\'unit=firewalld comm="systemd" exe="/usr/lib/systemd/systemd" hostname=? addr=? terminal=? res=success\'',
})


def _looks_like_dataset(d: Path) -> bool:
    for f in list(d.glob('*.json'))[:3]:
        try:
            obj = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if 'true_positives' in obj and 'true_negatives' in obj:
            return True
        if 'true_positives_test' in obj and 'true_negatives_test' in obj:
            return True
    return False


def discover_dirs(root: Path, dataset_dir=None, mapping_file=None):
    root = Path(root)
    results = next(
        (p for p in root.iterdir() if p.is_dir() and list(p.glob('batch_*_result.json'))),
        None,
    )
    if results is None:
        raise SystemExit(f'No results dir (batch_*_result.json) under {root}')

    seed = next(
        (p for p in root.iterdir()
         if p.is_dir() and any(p.glob('batch_*/??.json'))),
        None,
    )

    if dataset_dir is None:
        dataset = next(
            (p for p in root.iterdir()
             if p.is_dir() and any(p.glob('*.json'))
             and _looks_like_dataset(p)),
            None,
        )
    else:
        dataset = Path(dataset_dir)
        if not dataset.is_dir():
            raise SystemExit(f'dataset_dir does not exist: {dataset}')

    if mapping_file is None:
        mapping = root / 'filename_mapping.md'
        if not mapping.exists():
            mapping = next(root.glob('*mapping*.md'), None)
    else:
        mapping = Path(mapping_file)
        if not mapping.exists():
            raise SystemExit(f'mapping_file does not exist: {mapping}')

    if seed is None or dataset is None or mapping is None:
        raise SystemExit(
            f'Missing under {root}: seed={seed} dataset={dataset} mapping={mapping}'
        )
    return results, seed, dataset, mapping


def load_mapping(mapping_file: Path):
    pat = re.compile(r'\|\s*`(\d+)\.json`\s*\|\s*`([\w]+)`\s*\|')
    mapping = {}
    with open(mapping_file) as f:
        for line in f:
            m = pat.search(line)
            if m:
                mapping[m.group(1)] = m.group(2)
    return mapping


def _eval_universe(ds: dict, seed: str, rule: str = '', apply_label_overrides: bool = False):
    if 'true_positives' in ds and 'true_negatives' in ds:
        tps_univ, tns_univ, schema = (
            set(ds['true_positives']) - {seed}, set(ds['true_negatives']), 'retrospective',
        )
    elif 'true_positives_test' in ds and 'true_negatives_test' in ds:
        tps_univ, tns_univ, schema = (
            set(ds['true_positives_test']), set(ds['true_negatives_test']), 'realtime',
        )
    else:
        raise ValueError(f'Unknown dataset schema; keys={sorted(ds.keys())}')

    if apply_label_overrides and rule == _FW_RULE:
        tns_univ -= _FW_FORCE_POSITIVE
        tps_univ |= _FW_FORCE_POSITIVE

    return tps_univ, tns_univ, schema


def _execute(pattern, lines):
    if pattern is None:
        return set()
    return {ln for ln in lines if pattern.search(ln)}


def score_run(root, dataset_dir=None, mapping_file=None, write_files=True,
              apply_label_overrides=False):
    root = Path(root)
    res_dir, seed_dir, ds_dir, mapping_path = discover_dirs(
        root, dataset_dir=dataset_dir, mapping_file=mapping_file,
    )
    mapping = load_mapping(mapping_path)

    ds_cache = {}
    def load_ds(rule):
        if rule not in ds_cache:
            with open(ds_dir / f'{rule}.json') as f:
                ds_cache[rule] = json.load(f)
        return ds_cache[rule]

    per_rule = defaultdict(dict)
    rule_universe = {}
    invalid_regex = []
    schema_seen = set()

    batches = sorted(
        re.match(r'batch_(\d+)_result\.json', f.name).group(1)
        for f in res_dir.iterdir()
        if re.match(r'batch_\d+_result\.json', f.name)
    )

    for bid in batches:
        batch = f'batch_{bid}'
        with open(res_dir / f'{batch}_result.json') as f:
            res = json.load(f)
        for entry in res['results']:
            nn = entry['name']
            rule = mapping.get(nn)
            if rule is None:
                continue
            with open(seed_dir / batch / f'{nn}.json') as f:
                seed = json.load(f)['true_positive']
            tps_univ, tns_univ, schema = _eval_universe(
                load_ds(rule), seed, rule=rule,
                apply_label_overrides=apply_label_overrides,
            )
            schema_seen.add(schema)
            rule_universe[rule] = (len(tps_univ), len(tns_univ))

            regex_str = entry.get('regex')
            try:
                pat = re.compile(regex_str) if regex_str else None
            except re.error as err:
                pat = None
                invalid_regex.append((batch, nn, rule, str(err)[:80]))

            tp = len(_execute(pat, tps_univ))
            fp = len(_execute(pat, tns_univ))
            tpr = (tp / len(tps_univ)) if tps_univ else None
            fpr = (fp / len(tns_univ)) if tns_univ else None
            per_rule[rule][batch] = {
                'tp': tp, 'fp': fp,
                'tpr': round(tpr, 6) if tpr is not None else None,
                'fpr': round(fpr, 6) if fpr is not None else None,
            }

    for rule, by_batch in per_rule.items():
        runs = list(by_batch.values())
        n = len(runs)
        tprs = [r['tpr'] for r in runs if r['tpr'] is not None]
        fprs = [r['fpr'] for r in runs if r['fpr'] is not None]
        n_tp_eval, n_tn_eval = rule_universe[rule]
        by_batch['average'] = {
            'tp': round(sum(r['tp'] for r in runs) / n, 6),
            'fp': round(sum(r['fp'] for r in runs) / n, 6),
            'tpr': round(sum(tprs) / len(tprs), 6) if tprs else None,
            'fpr': round(sum(fprs) / len(fprs), 6) if fprs else None,
            'n_batches': n,
            'n_tp_eval': n_tp_eval,
            'n_tn_eval': n_tn_eval,
            'tpr_evaluable': n_tp_eval > 0,
            'fpr_evaluable': n_tn_eval > 0,
        }

    out = {}
    for rule in sorted(per_rule):
        by_batch = per_rule[rule]
        avg = by_batch.pop('average')
        ordered = {'average': avg}
        for k in sorted(by_batch):
            ordered[k] = by_batch[k]
        out[rule] = ordered

    if write_files:
        with open(res_dir / 'tp_fp_counts.json', 'w') as f:
            json.dump(out, f, indent=2)

    per_rule_eval, tpr_not_eval, fpr_not_eval = {}, {}, {}
    tpr_vals, fpr_vals = [], []
    for rule in sorted(out):
        avg = out[rule]['average']
        entry = {
            'average_tpr': avg['tpr'],
            'average_fpr': avg['fpr'],
            'n_tp_eval': avg['n_tp_eval'],
            'n_tn_eval': avg['n_tn_eval'],
            'tpr_evaluable': avg['tpr_evaluable'],
            'fpr_evaluable': avg['fpr_evaluable'],
        }
        if avg['tpr_evaluable'] and avg['tpr'] is not None:
            per_rule_eval[rule] = entry
            tpr_vals.append(avg['tpr'])
        else:
            tpr_not_eval[rule] = entry
        if avg['fpr_evaluable'] and avg['fpr'] is not None:
            fpr_vals.append(avg['fpr'])
        else:
            fpr_not_eval[rule] = entry

    final = {
        'overall': {
            'average_tpr': round(sum(tpr_vals) / len(tpr_vals), 6) if tpr_vals else None,
            'average_fpr': round(sum(fpr_vals) / len(fpr_vals), 6) if fpr_vals else None,
            'n_rules_total': len(out),
            'n_rules_tpr_evaluated': len(tpr_vals),
            'n_rules_tpr_not_evaluable': len(tpr_not_eval),
            'n_rules_fpr_evaluated': len(fpr_vals),
            'n_rules_fpr_not_evaluable': len(fpr_not_eval),
            'schema': sorted(schema_seen),
        },
        'per_rule': per_rule_eval,
        'tpr_not_evaluable': tpr_not_eval,
        'fpr_not_evaluable': fpr_not_eval,
    }
    if write_files:
        with open(res_dir / 'final_result.json', 'w') as f:
            json.dump(final, f, indent=2)

    return {
        'final': final,
        'schema': sorted(schema_seen),
        'invalid_regex': invalid_regex,
        'res_dir': str(res_dir),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(DEFAULT_ROOT))
    ap.add_argument('--dataset', default=None,
                    help='Override dataset dir (e.g. realtime split_dataset).')
    ap.add_argument('--mapping', default=None,
                    help='Override filename_mapping.md path.')
    ap.add_argument('--no-write', action='store_true')
    ap.add_argument('--apply-label-overrides', action='store_true', default=False,
                    help='Apply temporary scoring-time TN→TP overrides (see _FW_RULE patch).')
    args = ap.parse_args()

    r = score_run(args.root, dataset_dir=args.dataset, mapping_file=args.mapping,
                  write_files=not args.no_write,
                  apply_label_overrides=args.apply_label_overrides)
    final = r['final']
    print(f'root:   {args.root}')
    print(f'schema: {r["schema"]}')
    print(f'overall: {json.dumps(final["overall"], indent=2)}')
    if r['invalid_regex']:
        print(f'WARN: {len(r["invalid_regex"])} entries had invalid regex')
        for x in r['invalid_regex'][:5]:
            print(f'  {x}')


if __name__ == '__main__':
    main()
