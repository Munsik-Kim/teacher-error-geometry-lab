"""Check appendix facts and rounded public displays without fitting or model access."""
import argparse
import json
import math
import re
from pathlib import Path


def compare(expected, actual, path, errors, counters, *, atol, rtol):
    """Compare saved/rebuilt facts under the already frozen numerical tolerance."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            errors.append({'field': path, 'reason': 'mapping keys differ'})
            return
        for key in expected:
            compare(expected[key], actual[key], path + '.' + key, errors, counters,
                    atol=atol, rtol=rtol)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            errors.append({'field': path, 'reason': 'sequence length differs'})
            return
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare(left, right, path + '[' + str(index) + ']', errors, counters,
                    atol=atol, rtol=rtol)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        counters['numeric_values'] += 1
        if (not isinstance(actual, (int, float)) or isinstance(actual, bool)
                or not math.isfinite(expected) or not math.isfinite(actual)
                or not math.isclose(expected, actual, abs_tol=atol, rel_tol=rtol)):
            errors.append({'field': path, 'expected': expected, 'actual': actual,
                           'reason': 'numeric mismatch at frozen tolerance'})
    elif expected != actual:
        errors.append({'field': path, 'expected': expected, 'actual': actual,
                       'reason': 'literal mismatch'})


def has_display(text, value):
    # Public README tables use four decimal places. Both .1234 and 0.1234 work.
    token = format(value, '.4f')
    variants = {token, token[1:] if token.startswith('0.') else token}
    return any(re.search(r'(?<![\d.])' + re.escape(v) + r'(?![\d.])', text)
               for v in variants)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--summary', type=Path,
                        help='optional fresh zero-fit summary.json from RUNNER_TEMP')
    args = parser.parse_args()
    root = args.root.resolve()
    appendix = root / 'followups/linear_feature_closeout'
    facts = json.loads((appendix / 'facts.json').read_text())
    canonical = json.loads((root / 'data/results.json').read_text())
    stored = json.loads((appendix / 'results/closure_posthoc_01/summary.json').read_text())
    errors = []
    counters = {'numeric_values': 0}
    tolerance = {'atol': facts['verification']['atol'], 'rtol': facts['verification']['rtol']}
    compare(stored, facts['closure'], 'facts.closure', errors, counters, **tolerance)
    compare(canonical['p1']['selected_results'], facts['P1']['selected_recorded_results'],
            'facts.P1.selected_recorded_results', errors, counters, **tolerance)
    compare(canonical['p2']['methods'], facts['P2']['methods'],
            'facts.P2.methods', errors, counters, **tolerance)
    if args.summary:
        compare(stored, json.loads(args.summary.read_text()), 'rebuilt_summary',
                errors, counters, **tolerance)

    expected_status = {'registered_H1': 'NOT_RUN', 'registered_H1_effect': None,
                       'registered_H1_interval': None, 'registered_H2': 'NOT_RUN',
                       'registered_H3': 'NOT_RUN', 'independent_train_datasets': 1,
                       'planned': 4, 'attempted': 4, 'valid': 4, 'failed': 0,
                       'missing': 0, 'gold_reused': 1}
    for key, value in expected_status.items():
        if stored[key] != value:
            errors.append({'field': key, 'reason': 'post-hoc scope/denominator changed'})

    arms = {row['arm']: row for row in stored['arms']}
    transfer = {row['error']: row for row in stored['transfer']}
    displays = {}
    for direction in ('high', 'low'):
        mean_risk = (arms[direction + '_plus']['truth_risk']
                     + arms[direction + '_minus']['truth_risk']) / 2
        delta = mean_risk - stored['gold']['truth_risk']
        energy = transfer[direction]['eval_error_mse']
        calculated_ratio = delta / energy
        compare(transfer[direction]['sign_averaged_risk_increase'], delta,
                direction + '.derived_delta', errors, counters, **tolerance)
        compare(transfer[direction]['transfer_ratio_G'], calculated_ratio,
                direction + '.derived_G', errors, counters, **tolerance)
        displays[direction + '_sign_averaged_truth_risk'] = mean_risk
        displays[direction + '_G'] = calculated_ratio
    displayed_values_checked = 0
    for document in ('README.md', 'README.ko.md'):
        text = (root / document).read_text()
        for key, value in displays.items():
            displayed_values_checked += 1
            if not has_display(text, value):
                errors.append({'document': document, 'field': key,
                               'expected_four_decimal_display': format(value, '.4f'),
                               'reason': 'rounded display missing or inconsistent'})
    result = {'status': 'PASS' if not errors else 'FAIL',
              'numeric_values_checked': counters['numeric_values'],
              'root_README_rounded_values_checked': displayed_values_checked,
              'derived_public_displays': {k: format(v, '.4f') for k, v in displays.items()},
              'rebuilt_summary_checked': args.summary is not None,
              'tolerance': tolerance, 'fits_performed': 0, 'model_access': False,
              'errors': errors}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
