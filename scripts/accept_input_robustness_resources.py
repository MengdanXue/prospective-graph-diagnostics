"""Approved resource reliability and full-scope acceptance; no formal trainer.

An outer guardian meters the complete child coordinator, including setup,
hashing and final audit. The original short-probe computations stay unchanged.
Only acceptance_completion.json plus its closed ledger receipt is authoritative.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import preflight_input_robustness_11 as preflight

DEFAULT_CONFIG = ROOT / 'configs/input_robustness_11_v2.json'
DEFAULT_AUTHORIZATION = ROOT / 'results/diagnostic/posthoc_input_robustness_11_v2/budget_authorization.json'
DEFAULT_BINDING = preflight.DEFAULT_BINDING
REQUIRED_JOBS = {'full-protocol-verification', 'lightweight-verification', 'manuscript-build'}
APPROVED_CAPS = {'total_cumulative':1382400, 'resource_acceptance_subbudget':7200,
                 'coordination_validation_analysis_subbudget':7200,
                 'formal_all_attempts_subbudget':1368000,
                 'four_trial_unit_all_attempts':28800, 'paired_batch_all_attempts':57600}
SUPERVISOR_SOURCES = (*preflight.SOURCE_FILES, 'scripts/accept_input_robustness_resources.py',
                      'scripts/input_robustness_budget.py', 'scripts/input_robustness_runtime.py')
PAUSE_EXIT = 75
file_hash, json_hash, write_exclusive = preflight.file_hash, preflight.json_hash, preflight.write_exclusive


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def finite_nonnegative(value, name):
    require(type(value) in (int,float) and math.isfinite(value) and value>=0, f'invalid {name}')
    return value


def no_scores(value):
    if isinstance(value, dict):
        forbidden = {'validation_accuracy','validation_loss','test_accuracy','test_loss',
                     'selected_trial_id','regret','regret_pp','selection_accuracy'}
        require(not forbidden.intersection(value), 'held-out or selected-model score in resource evidence')
        for key in ('validation_evaluations','test_evaluations','formal_records'):
            if key in value:
                require(type(value[key]) is int and value[key] == 0, f'nonzero {key}')
        for key in ('formal_training_enabled','formal_launch_authorized'):
            if key in value:
                require(value[key] is False, 'formal training remains unauthorized')
        for item in value.values():
            no_scores(item)
    elif isinstance(value, list):
        for item in value:
            no_scores(item)


def validate_ci_receipt(receipt, commit):
    require(isinstance(commit,str) and re.fullmatch(r'[0-9a-f]{40}',commit), 'invalid source commit')
    require(receipt.get('commit') == commit and receipt.get('status') == 'completed'
            and receipt.get('conclusion') == 'success', 'same-SHA completed successful CI required')
    run = receipt.get('run_id')
    require(type(run) is int and run > 0, 'positive CI run id required')
    jobs = receipt.get('jobs', [])
    require(len(jobs) == 3 and {j.get('name') for j in jobs} == REQUIRED_JOBS, 'exactly three distinct CI jobs required')
    require(all(j.get('run_id') == run and j.get('status') == 'completed'
                and j.get('conclusion') == 'success' for j in jobs), 'CI jobs must succeed in this run')


def validate_approved_config(config, authorization):
    original = read_json(preflight.DEFAULT_CONFIG)
    preflight.validate_config(config)
    for key in ('datasets','conditions','seeds','models','eligibility','split','training','model_parameters',
                'transform','diagnostics','data_binding','preflight','expected_records','expected_trials',
                'expected_evaluation_units_per_condition','expected_unique_diagnostic_units','evaluator',
                'sensitivity','calibration','pairing','inference'):
        require(config[key] == original[key], f'unchanged scientific contract required: {key}')
    require(config['execution'] == {**original['execution'], 'h2gcn_checkpoint_mode':'original_full_state_copy'},
            'only the original four-thread H2GCN execution is authorized')
    expected_budget = {**original['resource_budget'], 'formal_model_unit_wall_seconds':28800,
                       'formal_total_worker_wall_seconds':1368000}
    require(set(config['resource_budget']) == set(expected_budget)
            and all(config['resource_budget'][k]==v for k,v in expected_budget.items() if k!='on_cap_or_estimate_exceeded'),
            'only approved resource budget fields may change')
    require(config['run_id'] == authorization['run_id'] == 'posthoc_input_robustness_11_v2', 'resource run identity mismatch')
    require(authorization.get('budget_change_authorized') is True
            and authorization.get('resource_acceptance_authorized') is True
            and authorization.get('formal_launch_authorized') is False
            and authorization.get('formal_training_enabled') is False, 'resource-only user authorization required')
    require(config.get('budget_change_authorized') is True and config.get('formal_launch_authorized') is False,
            'resource budget authorized but formal launch remains closed')
    require(authorization['approved_caps_seconds'] == APPROVED_CAPS, 'approved cumulative caps changed')
    require(authorization['retained_h2_execution'] == {'device':'cpu','threads':4,'checkpoint_mode':'original_full_state_copy'},
            'H2 execution authorization mismatch')
    accounting = config['budget_accounting']
    for key, expected in APPROVED_CAPS.items():
        require(accounting.get(key+'_seconds') == expected, f'cumulative accounting cap mismatch: {key}')
    require(accounting['unit_clock_reset_on_retry'] is False and accounting['batch_clock_reset_on_retry'] is False
            and accounting['unused_reserves_auto_transfer'] is False and accounting['cumulative_across_attempts'] is True,
            'attempt budgets cannot reset or borrow reserves')
    controls = config['runtime_controls']
    require((controls['poll_interval_seconds'],controls['max_poll_gap_seconds'],controls['max_clock_divergence_seconds'])
            == (0.25,5,2), 'fixed runtime-monitor boundaries changed')
    require(controls['task_power_request_required'] is True and controls['power_event_watcher_required'] is True
            and controls['outer_guardian_required'] is True, 'complete task-scoped runtime protection required')
    require(controls['normal_pause']['preflight_boundary'].startswith('after trial_003')
            and controls['normal_pause']['stop_new_dispatch'] is True, 'normal pause must use whole-model boundary')
    approved = authorization['approved_proposal']
    require(file_hash(ROOT/approved['path']) == approved['sha256'] == accounting['approved_proposal_sha256'],
            'approved proposal bytes changed')
    require(file_hash(ROOT/config['bound_input_source']['path']) == config['bound_input_source']['sha256'],
            'inherited data binding changed')
    return config


def derive_unit_projections(config, directory, jobs):
    """Allocate the original conservative total to full four-trial units."""
    directory = Path(directory)
    lookup = {j['worker_id']:j for j in jobs}
    require(len(lookup) == len(jobs), 'duplicate worker identities')
    for job in jobs:
        finite_nonnegative(job['wall_seconds'],'raw worker wall duration')
        finite_nonnegative(job['peak_rss_bytes'],'raw process-tree memory')
    result = []
    for dataset in config['datasets']:
        for condition in config['conditions']:
            group = [w for w in preflight.workers(config) if w['dataset']==dataset and w['condition']==condition]
            headers, residuals, cells, h2 = [], [], {}, 0.0
            for worker in group:
                wid = worker['worker_id']
                header = read_json(directory/'workers'/wid/'worker.json')
                finite_nonnegative(header['transform_seconds'],'raw transform duration')
                headers.append(header)
                compute = 0.0
                for model in config['models']:
                    for index in range(4):
                        row = read_json(directory/'workers'/wid/'probes'/model/f'trial_{index:03d}.json')
                        require(row.get('status') == 'success', 'failed probe cannot supply a runtime projection')
                        durations = row['epoch_seconds']
                        require(len(durations)==3 and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in durations),
                                'invalid epoch duration')
                        compute += sum(durations)
                        cells.setdefault((model,index),[]).append(sum(durations)/3)
                        h2 = max(h2, finite_nonnegative(row.get('h2_preparation_seconds',0.0),'raw H2 preparation duration'))
                        for field in ('peak_cuda_reserved_bytes','peak_cuda_allocated_bytes','rss_bytes',
                                      'four_checkpoints_bytes'):
                            finite_nonnegative(row[field],f'raw {field}')
                require(wid in lookup, 'worker job absent from projection')
                residuals.append(max(0.0,lookup[wid]['wall_seconds']-compute))
            transform = max(h['transform_seconds'] for h in headers)
            residual = max(residuals)
            require(all(math.isfinite(v) and v>=0 for v in (transform,residual,h2)), 'invalid setup duration')
            for model in config['models']:
                compute = config['training']['max_epochs']*sum(max(cells[(model,index)]) for index in range(4))
                setup = transform+residual+(h2 if model=='H2GCN' else 0.0)
                result.append({'dataset':dataset,'condition':condition,'model':model,
                    'training_seconds_before_safety_factor':compute,'setup_seconds_before_safety_factor':setup,
                    'transform_seconds':transform,'worker_nontraining_overhead_seconds':residual,
                    'h2_preparation_seconds':h2 if model=='H2GCN' else 0.0,
                    'four_trial_unit_seconds':2*(compute+setup)})
    return result


def summarize_probe_evidence(config, directory, jobs):
    result = preflight.summarize(config, directory, jobs)
    failures = result['failures']
    try:
        for path in (Path(directory)/'workers').rglob('*.json'):
            no_scores(read_json(path))
        projections = derive_unit_projections(config,directory,jobs)
        projected = sum(row['four_trial_unit_seconds'] for row in projections)*len(config['seeds'])
        old_estimate = result['runtime_estimate']['conservative_seconds']
        require(old_estimate is not None and math.isclose(projected,old_estimate,rel_tol=1e-12),
                'full-unit allocation differs from original conservative total')
        for row in projections:
            if row['four_trial_unit_seconds'] > 28800:
                failures.append({**row,'reason':'full four-trial unit including setup exceeds approved cap'})
        if projected > 1368000:
            failures.append({'reason':'full formal projection exceeds the 380-hour subbudget'})
        result['full_unit_projections'] = projections
        result['runtime_estimate'].update(full_unit_allocated_seconds=projected,
                formal_all_attempts_cap_seconds=1368000,four_trial_unit_cap_seconds=28800)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        failures.append({'reason':f'full-unit evidence validation failed: {exc}'})
        result['full_unit_projections'] = []
    result['resource_and_short_repeatability_passed'] = not failures
    no_scores(result)
    return result


def apply_runtime_acceptance(result, monitor_record, ledger_snapshot):
    failures = result['failures']
    if (monitor_record.get('status')!='passed' or monitor_record.get('stop_reasons')
            or monitor_record.get('worker_count')!=44
            or monitor_record.get('idle_sleep_request_acquired') is not True
            or monitor_record.get('idle_sleep_request_released') is not True):
        failures.append({'reason':'complete runtime guardian and power-request lifecycle must pass'})
    if (ledger_snapshot.get('integrity_passed') is not True or ledger_snapshot.get('must_stop') is not False
            or ledger_snapshot.get('stop_reasons')):
        failures.append({'reason':'cumulative ledger integrity or stop state failed'})
    usage = ledger_snapshot.get('usage_seconds',{})
    limits = {'total':1382400,'resource':7200,'formal':1368000,'control':7200}
    if (any(type(usage.get(k)) not in (int,float) or not math.isfinite(usage[k]) or not 0<=usage[k]<=v for k,v in limits.items())
            or usage.get('formal')!=0 or usage.get('control')!=0):
        failures.append({'reason':'resource-only stage exceeds cumulative limits or includes unauthorized formal/control work'})
    result.update(resource_and_short_repeatability_passed=not failures,formal_launch_authorized=False,
                  runtime_guardian=monitor_record,budget_ledger=ledger_snapshot)
    no_scores(result)
    return result


def summarize_acceptance(config, directory, jobs, monitor_record, ledger_snapshot):
    return apply_runtime_acceptance(summarize_probe_evidence(config,directory,jobs),monitor_record,ledger_snapshot)


class ResourceProbeFailure(RuntimeError):
    pass


class GracefulPause(BaseException):
    """A saved-unit control boundary, deliberately outside probe failures."""
    pass


def run_probe_worker(config, worker, data_root, binding, destination):
    """Use the original calculations; change only failure and pause boundaries."""
    original_writer = preflight.write_exclusive
    original_probe = preflight.train_probe
    destination = Path(destination)
    run_root = destination.parent.parent
    def guarded_writer(path,value):
        original_writer(path,value)
        if value.get('status')=='failed' or value.get('failed_probes',0):
            raise ResourceProbeFailure('first failed probe was saved; no following probe is authorized')
        if value.get('status')=='success' and value.get('trial_id')=='trial_003' and value.get('model') in config['models']:
            model = value['model']
            files = [destination/'probes'/model/f'trial_{i:03d}.json' for i in range(4)]
            for probe in files:
                require(read_json(probe).get('status')=='success','whole-model pause marker requires all four probes')
            original_writer(destination/'unit_completions'/f'{model}.json',{
                'phase':'preflight_only','run_id':config['run_id'],'worker_id':worker['worker_id'],
                'model':model,'trials':4,'probe_sha256':[file_hash(p) for p in files],
                'validation_evaluations':0,'test_evaluations':0,'formal_records':0})
            if (run_root/'control'/'pause_request.json').exists():
                raise GracefulPause('ordinary pause: current four-trial model unit saved')
    preflight.write_exclusive = guarded_writer
    def guarded_probe(**kwargs):
        if kwargs.get('trial')==config['training']['trials'][0] and (run_root/'control'/'pause_request.json').exists():
            raise GracefulPause('ordinary pause before the next model starts training')
        return original_probe(**kwargs)
    preflight.train_probe = guarded_probe
    try:
        if (run_root/'control'/'pause_request.json').exists():
            raise GracefulPause('ordinary pause before the next model unit')
        return preflight.run_worker(config,worker,data_root,binding,destination)
    finally:
        preflight.write_exclusive = original_writer
        preflight.train_probe = original_probe


def process_identity(process):
    import psutil
    item = psutil.Process(process.pid)
    return {'pid':item.pid,'create_time':item.create_time()}


def identity_is_running(identity):
    import psutil
    try:
        item = psutil.Process(identity['pid'])
        return item.is_running() and item.create_time()==identity['create_time'] and item.status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def supervise_process(command, *, cwd, environment, log_path, worker_cap_seconds,
                      poll_seconds=0.25, memory_limits=None, guardian_identity=None,
                      emergency_path=None, on_poll=None, injected_gap_seconds=0,
                      injected_postexit_gap_seconds=0):
    """Always inspect the final interval, including after a child exits."""
    import psutil
    from scripts.input_robustness_budget import DualClockGuard
    guard = DualClockGuard(monitor_gap_seconds=5, clock_skew_tolerance_seconds=2)
    started_wall, started_mono = time.time(), time.monotonic()
    Path(log_path).parent.mkdir(parents=True,exist_ok=True)
    observed, owned = set(), {}
    reasons, peak, injected = [], 0, False
    process = None
    def remember_tree():
        nonlocal peak
        try:
            root_process = psutil.Process(process.pid)
            measured = preflight.process_tree_memory(root_process,psutil)
            finite_nonnegative(measured['rss_bytes'],'observed process-tree RSS')
            finite_nonnegative(measured['high_water_bytes'],'observed process-tree peak memory')
            peak = max(peak,measured['rss_bytes'],measured['high_water_bytes'])
            observed.update(measured['process_ids'])
            for pid in measured['process_ids']:
                if pid not in owned:
                    try:
                        item = psutil.Process(pid)
                        owned[pid] = {'pid':pid,'create_time':item.create_time()}
                    except psutil.NoSuchProcess:
                        pass
        except psutil.NoSuchProcess:
            pass
    def stop_all():
        if process is not None and process.poll() is None:
            try:
                preflight.terminate_process_tree(process,psutil)
            except Exception as error:
                reasons.append('process-tree cleanup failed: '+repr(error))
        # Preserve ownership by PID and creation time even if an abnormal child
        # exit orphaned a descendant before the final process-tree query.
        for identity in reversed(list(owned.values())):
            try:
                if identity_is_running(identity):
                    psutil.Process(identity['pid']).kill()
            except psutil.NoSuchProcess:
                pass
            except Exception as error:
                reasons.append('owned-process cleanup failed: '+repr(error))
    try:
        with Path(log_path).open('x',encoding='utf-8',newline='\n') as log:
            process = subprocess.Popen(command,cwd=cwd,env=environment,stdout=log,stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            remember_tree()
            while True:
                elapsed = max(0,time.time()-started_wall,time.monotonic()-started_mono)
                if injected_gap_seconds and not injected and elapsed>=0.5:
                    time.sleep(injected_gap_seconds)
                    injected = True
                remember_tree()
                clock = guard.sample()
                if clock.get('stop_reasons'):
                    reasons.extend(clock['stop_reasons'])
                elapsed = max(0,time.time()-started_wall,time.monotonic()-started_mono)
                if elapsed>worker_cap_seconds:
                    reasons.append('worker wall cap exceeded')
                if guardian_identity and not identity_is_running(guardian_identity):
                    reasons.append('outer guardian is no longer alive')
                if emergency_path and Path(emergency_path).exists():
                    reasons.append('emergency stop requested')
                if memory_limits:
                    if peak>memory_limits['max_worker_rss_bytes']:
                        reasons.append('owned process-tree RSS cap exceeded')
                    if finite_nonnegative(psutil.virtual_memory().available,'available memory')<memory_limits['min_system_available_bytes']:
                        reasons.append('system available-memory reserve crossed')
                    if finite_nonnegative(shutil.disk_usage(Path(log_path).parent).free,'free disk')<memory_limits['min_disk_free_bytes']:
                        reasons.append('free disk reserve crossed')
                if on_poll:
                    extra = on_poll(process,elapsed)
                    if extra:
                        reasons.extend(extra)
                if reasons:
                    stop_all()
                    break
                if process.poll() is not None:
                    if injected_postexit_gap_seconds:
                        time.sleep(injected_postexit_gap_seconds)
                        post = guard.sample()
                        if post.get('stop_reasons'):
                            reasons.extend(post['stop_reasons'])
                    break
                time.sleep(poll_seconds)
            returncode = process.wait(timeout=15)
            final_clock = guard.sample()
            if final_clock.get('stop_reasons'):
                reasons.extend(final_clock['stop_reasons'])
            final_seconds = max(0,time.time()-started_wall,time.monotonic()-started_mono)
            if final_seconds>worker_cap_seconds:
                reasons.append('worker wall cap exceeded')
    finally:
        stop_all()
        if process is not None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=15)
    remaining = [item for item in owned.values() if identity_is_running(item)]
    if remaining:
        reasons.append('owned descendants remain alive after cleanup')
    final_clock = guard.sample()
    reasons.extend(final_clock['stop_reasons'])
    final_seconds = max(0,time.time()-started_wall,time.monotonic()-started_mono)
    if final_seconds>worker_cap_seconds:
        reasons.append('worker wall cap exceeded')
    return {'returncode':returncode,'stop_reason':'; '.join(dict.fromkeys(reasons)) or None,
            'stop_reasons':list(dict.fromkeys(reasons)), 'wall_seconds':final_seconds,
            'peak_rss_bytes':peak,'observed_process_ids':sorted(observed),
            'memory_scope':'owned_worker_process_tree_sum_including_windows_high_water',
            'owned_processes_remaining':remaining,
            'clock_guard':{**final_clock,'statistics':dict(guard.stats)}}


def verify_manifest(manifest, config, authorization, binding_path):
    require(manifest.get('phase') in ('runtime_reliability_only','resource_acceptance_only'), 'wrong guarded phase')
    require(manifest['config_sha256']==json_hash(config)
            and manifest['authorization_sha256']==file_hash(DEFAULT_AUTHORIZATION), 'configuration or authorization drift')
    require(manifest['binding_sha256']==file_hash(binding_path)==config['bound_input_source']['sha256'], 'data-binding drift')
    require(set(manifest['supervisor_sources'])==set(SUPERVISOR_SOURCES), 'incomplete guardian source binding')
    for relative,digest in manifest['supervisor_sources'].items():
        require(file_hash(ROOT/relative)==digest,f'executable-source drift: {relative}')
    validate_approved_config(config,authorization)
    validate_ci_receipt(manifest['ci_receipt'],manifest['source_commit'])


def acceptance_body(args, config, authorization):
    output = args.output_root.resolve()
    manifest = read_json(output/'acceptance_manifest.json')
    verify_manifest(manifest,config,authorization,args.binding)
    require(identity_is_running(manifest['guardian_identity']), 'a live owned guardian is required')
    binding = read_json(args.binding)
    require(binding['run_id']==config['bound_input_source']['original_run_id']
            and binding['status']=='passed' and binding['dataset_count']==11 and binding['bound_split_count']==110
            and set(binding['datasets'])==set(config['datasets'])
            and all(row.get('all_original_diagnostics_exact') is True and len(row.get('full_diagnostic_checks',[]))==10
                    for row in binding['datasets'].values()), 'complete inherited data and split binding required')
    inner = {k:manifest[k] for k in ('analysis_status','run_id','source_commit','config','config_sha256','binding_sha256')}
    inner.update(phase='preflight_only',source_files={k:manifest['supervisor_sources'][k] for k in preflight.SOURCE_FILES},
                 validation_evaluations=0,test_evaluations=0,formal_records=0)
    write_exclusive(output/'preflight_manifest.json',inner)
    jobs, pause = [], False
    budget = preflight.resolved_budget(config)
    for worker in preflight.workers(config):
        if (output/'control'/'pause_request.json').exists():
            pause = True
            break
        verify_manifest(manifest,config,authorization,args.binding)
        command = [sys.executable,str(Path(__file__).resolve()),'--config',str(args.config.resolve()),
                   '--binding',str(args.binding.resolve()),'--data-root',str(args.data_root.resolve()),
                   '--output-root',str(output),'--worker-id',worker['worker_id']]
        job = supervise_process(command,cwd=ROOT,environment=preflight.backend_environment(),
            log_path=output/'logs'/f"{worker['worker_id']}.log",
            worker_cap_seconds=budget['preflight_worker_wall_seconds'],memory_limits=budget,
            guardian_identity=manifest['guardian_identity'],emergency_path=output/'control'/'emergency_stop.json')
        job = {**worker,**job}
        if job['returncode']==PAUSE_EXIT and not job['stop_reason']:
            pause = True
            job['stop_reason']='ordinary pause at a complete model boundary'
        jobs.append(job)
        write_exclusive(output/'jobs'/f"{worker['worker_id']}.json",job)
        print(json.dumps({**worker,'returncode':job['returncode'],'stop_reason':job['stop_reason'],
                          'wall_seconds':job['wall_seconds'],'peak_rss_bytes':job['peak_rss_bytes']}),flush=True)
        if job['returncode']!=0 or job['stop_reason']:
            break
    # A candidate summary is evidence only. The external guardian must still
    # finish resource monitoring, release native handles and close the ledger.
    summary = summarize_probe_evidence(config,output,jobs)
    summary.update(guardian_acceptance_pending=True)
    verify_manifest(manifest,config,authorization,args.binding)
    no_scores(summary)
    write_exclusive(output/'preflight_summary.json',summary)
    write_exclusive(output/'preflight_inventory.json',{'phase':'preflight_only','files':{
        p.relative_to(output).as_posix():{'bytes':p.stat().st_size,'sha256':file_hash(p)}
        for p in sorted(output.rglob('*')) if p.is_file() and not p.relative_to(output).as_posix().startswith('control/')}})
    write_exclusive(output/'body_complete.json',{'phase':'resource_acceptance_only','jobs':len(jobs),
        'candidate_summary_sha256':file_hash(output/'preflight_summary.json'),
        'paused':pause,'status':'candidate_complete' if len(jobs)==44 and not summary['failures'] else 'incomplete',
        'validation_evaluations':0,'test_evaluations':0,'formal_records':0})
    return PAUSE_EXIT if pause else 0 if len(jobs)==44 and not summary['failures'] else 2


def synthetic_worker(args):
    """Tiny durable four-trial work units for reliability checks, without ML."""
    target = args.case_dir
    target.mkdir(parents=True,exist_ok=True)
    if args.synthetic_worker=='stall':
        write_exclusive(target/'started.json',{'pid':os.getpid()})
        time.sleep(45)
        return 0
    if args.synthetic_worker=='exit':
        return 0
    for unit in range(2):
        if (target/'pause_request.json').exists():
            return PAUSE_EXIT
        for trial in range(4):
            time.sleep(args.trial_seconds)
            write_exclusive(target/f'unit_{unit:03d}_trial_{trial:03d}.json',{'unit':unit,'trial':trial,'synthetic':True})
        write_exclusive(target/f'unit_{unit:03d}_complete.json',{'trials':4,'synthetic':True})
        if (target/'pause_request.json').exists():
            return PAUSE_EXIT
    return 0


def reliability_body(args, config, authorization):
    output = args.output_root.resolve()
    manifest = read_json(output/'acceptance_manifest.json')
    verify_manifest(manifest,config,authorization,args.binding)
    cases, paused = [], False
    for name,kind,trial_seconds,cap,gap,postgap in (
        ('normal_pause','unit',0.3,30,0,0),
        ('emergency_stop','unit',1.0,30,0,0),
        ('worker_timeout','stall',0.1,1.0,0,0),
        ('monitor_gap','stall',0.1,30,6.0,0),
        ('postexit_gap','exit',0.1,30,0,6.0)):
        if (output/'control'/'pause_request.json').exists():
            paused = True
            break
        target = output/'reliability_cases'/name
        target.mkdir(parents=True)
        command = [sys.executable,str(Path(__file__).resolve()),'--synthetic-worker',kind,
                   '--case-dir',str(target),'--trial-seconds',str(trial_seconds)]
        def control(process,elapsed):
            first = target/'unit_000_trial_000.json'
            if first.exists() and name=='normal_pause' and not (target/'pause_request.json').exists():
                write_exclusive(target/'pause_request.json',{'synthetic_request':True,'wall':time.time()})
            if first.exists() and name=='emergency_stop':
                return ['synthetic emergency stop requested']
            return []
        measured = supervise_process(command,cwd=ROOT,environment=os.environ.copy(),log_path=target/'worker.log',
            worker_cap_seconds=cap,on_poll=control,guardian_identity=manifest['guardian_identity'],
            injected_gap_seconds=gap,injected_postexit_gap_seconds=postgap)
        trials = sorted(p.name for p in target.glob('unit_*_trial_*.json'))
        units = sorted(p.name for p in target.glob('unit_*_complete.json'))
        passed = not measured['owned_processes_remaining']
        if name=='normal_pause':
            passed = passed and measured['returncode']==PAUSE_EXIT and not measured['stop_reason'] and len(trials)==4 and len(units)==1
        elif name=='emergency_stop':
            passed = passed and measured['returncode']!=0 and len(trials)>=1 and len(trials)<4 and not units
        else:
            passed = passed and bool(measured['stop_reason'])
            if name!='postexit_gap':
                passed = passed and measured['returncode']!=0
        row = {'case':name,'passed':passed,'synthetic_only':True,'trial_files':trials,'unit_files':units,**measured}
        cases.append(row)
        write_exclusive(target/'case_result.json',row)
        if not passed:
            break
    result = {'phase':'runtime_reliability_only','cases':cases,'passed':len(cases)==5 and all(c['passed'] for c in cases),
              'planned_cases':5,'paused':paused,'source_commit':manifest['source_commit'],
              'actual_os_sleep_forced':False,'model_training_steps':0,
              'validation_evaluations':0,'test_evaluations':0,'formal_records':0,
              'scope':'Real owned subprocess control with injected monitor stalls; native power lifecycle is checked by the outer guardian.'}
    verify_manifest(manifest,config,authorization,args.binding)
    write_exclusive(output/'reliability_body_result.json',result)
    return PAUSE_EXIT if paused else 0 if result['passed'] else 2


def ledger_authority(config, authorization):
    return {'run_id':config['run_id'],'authorization_sha256':file_hash(DEFAULT_AUTHORIZATION),
            'proposal_sha256':authorization['approved_proposal']['sha256'],
            'scope_sha256':json_hash(authorization['scope'])}


def ledger_view(snapshot):
    return {**snapshot,'integrity_passed':True,'usage_seconds':snapshot['charged_seconds']}


def validate_closed_completion(receipt_path, ledger_path, authority, *, expected_phase=None):
    """Completion alone is insufficient: bind its digest to the trusted ledger head."""
    from scripts.input_robustness_budget import inspect_ledger
    receipt_path, ledger_path = Path(receipt_path), Path(ledger_path)
    receipt = read_json(receipt_path)
    completion_path = receipt_path.parent/'acceptance_completion.json'
    require(receipt['completion_sha256']==file_hash(completion_path), 'completion receipt digest mismatch')
    completion = read_json(completion_path)
    require(completion['authoritative_only_with_closed_ledger'] is True, 'conditional completion contract missing')
    require(completion['status']=='passed' and receipt['status']=='passed', 'acceptance did not pass')
    require(completion['manifest_sha256']==file_hash(receipt_path.parent/'acceptance_manifest.json')
            and completion['runtime_guardian_sha256']==file_hash(receipt_path.parent/'runtime_guardian.json'),
            'completed manifest or runtime evidence changed')
    runtime = read_json(receipt_path.parent/'runtime_guardian.json')
    require(runtime['status']=='passed' and not runtime['stop_reasons']
            and runtime['idle_sleep_request_acquired'] is True and runtime['idle_sleep_request_released'] is True,
            'completed runtime lifecycle did not pass')
    if expected_phase:
        require(completion['phase']==expected_phase, 'wrong accepted predecessor phase')
    inspection = inspect_ledger(ledger_path,authority=authority,expected_head=receipt['head'])
    require(inspection['must_stop'] is False and not inspection['open_attempts'], 'ledger is stopped or incomplete')
    attempt = inspection['attempts'][receipt['attempt_id']]
    require(attempt['outcome']=='completed' and attempt['record_sha256']==receipt['completion_sha256'],
            'closed ledger does not bind successful completion')
    provenance = inspection['phases'][attempt['phase_id']]['provenance']
    require(completion['attempt_id']==receipt['attempt_id'] and completion['source_commit']==provenance['source_commit']
            and completion['run_id']==authority['run_id'], 'completion identity differs from ledger authority/source')
    no_scores(completion)
    return {'receipt':receipt,'completion':completion,'ledger':inspection}


def environment_identity():
    import psutil
    versions = {}
    for name in ('torch','torch-geometric','numpy','scipy','psutil'):
        versions[name] = importlib.metadata.version(name)
    return {'python_executable':sys.executable,'python':platform.python_version(),
            'platform':platform.platform(),'machine':platform.machine(),'processor':platform.processor(),
            'logical_cpu_count':psutil.cpu_count(),'physical_memory_bytes':psutil.virtual_memory().total,
            'library_versions':versions,'execution_backend':preflight.backend_environment().get('CUBLAS_WORKSPACE_CONFIG'),
            'h2gcn_device':'cpu','h2gcn_threads':4,'h2gcn_checkpoint_mode':'original_full_state_copy'}


def run_guardian(args, config, authorization, *, power_request=None, power_watcher=None):
    """One synchronous owner meters a separately supervised complete coordinator."""
    import psutil
    from scripts.input_robustness_budget import BudgetLedger
    from scripts.input_robustness_runtime import TaskPowerRequest, PowerEventWatcher
    # Include interpreter/bootstrap time through process creation, with both
    # clock readings preserved; a long startup is an abnormal gap, never free.
    own = psutil.Process(os.getpid())
    now_wall, now_mono = time.time(), time.monotonic()
    started_at = {'wall':own.create_time(), 'monotonic':now_mono-max(0,now_wall-own.create_time())}
    require(args.ci_receipt is not None and args.budget_ledger is not None, 'CI receipt and shared ledger are required')
    output = args.output_root.resolve()
    require(not output.exists(), 'acceptance output must be new; preserve all previous attempts')
    git = ['git','-c',f'safe.directory={ROOT.as_posix()}']
    commit = subprocess.check_output([*git,'rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    require(not subprocess.check_output([*git,'status','--porcelain'],cwd=ROOT,text=True).strip(),
            'tested clean committed source required before acceptance')
    upstream = subprocess.check_output([*git,'rev-parse','@{upstream}'],cwd=ROOT,text=True).strip()
    require(upstream==commit, 'remote-tracking branch must match reviewed source')
    ci_receipt = read_json(args.ci_receipt)
    validate_ci_receipt(ci_receipt,commit)
    require(file_hash(args.binding)==config['bound_input_source']['sha256'], 'selected binding differs from authorization')
    authority = ledger_authority(config,authorization)
    phase = 'runtime_reliability_only' if args.reliability_only else 'resource_acceptance_only'
    predecessor = None
    if args.reliability_only:
        require(args.ledger_head_receipt is None, 'fixed reliability acceptance is first and cannot reset/retry a ledger')
        ledger = BudgetLedger.create(args.budget_ledger,authority=authority)
    else:
        require(args.ledger_head_receipt is not None, 'passed reliability receipt and existing shared ledger required')
        predecessor = validate_closed_completion(args.ledger_head_receipt,args.budget_ledger,authority,
                                                 expected_phase='runtime_reliability_only')
        ledger = BudgetLedger.open(args.budget_ledger,authority=authority,expected_head=predecessor['receipt']['head'])
    phase_id = phase+'_'+commit
    source_files = {relative:file_hash(ROOT/relative) for relative in SUPERVISOR_SOURCES}
    environment = environment_identity()
    provenance = {'source_commit':commit,'config_sha256':json_hash(config),'source_files':source_files,
                  'environment':environment,'data_binding_sha256':file_hash(args.binding)}
    gate = {'ci_receipt':ci_receipt,'ci_receipt_file_sha256':file_hash(args.ci_receipt),
            'review_record_sha256':file_hash(args.ledger_head_receipt) if predecessor else file_hash(DEFAULT_AUTHORIZATION)}
    ledger.register_phase(phase_id,provenance=provenance,gate_receipt=gate)
    activity_id, attempt_id = phase+'_v2', phase+'_v2_attempt_001'
    snapshot = ledger.begin_attempt(attempt_id,activity_id=activity_id,phase_id=phase_id,budget_group='resource',
                                    estimated_seconds=30 if args.reliability_only else 1800,started_at=started_at)
    output.mkdir(parents=True)
    power_request = power_request if power_request is not None else TaskPowerRequest()
    power_watcher = power_watcher if power_watcher is not None else PowerEventWatcher()
    reasons, job, candidate = list(snapshot['stop_reasons']), None, None
    ordinary_pause = False
    manifest = {'phase':phase,'analysis_status':config['analysis_status'],'run_id':config['run_id'],
                'source_commit':commit,'supervisor_sources':source_files,'config':config,
                'config_sha256':json_hash(config),'authorization_sha256':file_hash(DEFAULT_AUTHORIZATION),
                'binding_sha256':file_hash(args.binding),'environment':environment,'ci_receipt':ci_receipt,
                'ci_receipt_file_sha256':file_hash(args.ci_receipt),'guardian_identity':process_identity(own),
                'ledger_authority':authority,'attempt_id':attempt_id,'phase_id':phase_id,
                'predecessor_receipt_sha256':file_hash(args.ledger_head_receipt) if predecessor else None,
                'validation_evaluations':0,'test_evaluations':0,'formal_records':0}
    def poll_runtime(process=None, elapsed=None):
        current = ledger.poll()
        found = list(current['stop_reasons'])
        if power_watcher.events:
            found.append('suspend/resume notification or callback failure')
        if (output/'control'/'emergency_stop.json').exists():
            found.append('emergency stop requested')
        return found
    try:
        write_exclusive(output/'acceptance_manifest.json',manifest)
        require(not reasons,'ledger denied runtime dispatch: '+repr(reasons))
        power_request.acquire()
        power_watcher.start()
        reasons.extend(poll_runtime())
        require(not reasons,'runtime protection denied dispatch: '+repr(reasons))
        budget = preflight.resolved_budget(config)
        command = [sys.executable,str(Path(__file__).resolve()),'--body','--config',str(args.config.resolve()),
                   '--binding',str(args.binding.resolve()),'--data-root',str(args.data_root.resolve()),
                   '--output-root',str(output)]
        if args.reliability_only:
            command.append('--reliability-only')
        job = supervise_process(command,cwd=ROOT,environment=preflight.backend_environment(),
            log_path=output/'guardian_body.log',worker_cap_seconds=snapshot['remaining_seconds']['resource'],
            memory_limits=budget,emergency_path=output/'control'/'emergency_stop.json',on_poll=poll_runtime)
        reasons.extend(job['stop_reasons'])
        ordinary_pause = job['returncode']==PAUSE_EXIT and not reasons
        if job['returncode']!=0 and not ordinary_pause:
            reasons.append('guarded body did not complete successfully')
        if args.reliability_only:
            candidate = read_json(output/'reliability_body_result.json')
            if not ordinary_pause and (candidate.get('passed') is not True or len(candidate.get('cases',[]))!=5):
                reasons.append('fixed reliability cases did not all pass')
        elif not ordinary_pause:
            marker = read_json(output/'body_complete.json')
            require(marker['candidate_summary_sha256']==file_hash(output/'preflight_summary.json'), 'body candidate digest mismatch')
            candidate = read_json(output/'preflight_summary.json')
            if marker['status']!='candidate_complete' or marker['jobs']!=44 or candidate['failures']:
                reasons.append('fixed 44-worker resource evidence did not pass')
        verify_manifest(manifest,config,authorization,args.binding)
        reasons.extend(poll_runtime())
    except BaseException as error:
        reasons.append('guardian exception: '+repr(error))
        write_exclusive(output/'guardian_failure.json',{'error':repr(error),'traceback':traceback.format_exc(),
                                                      'validation_evaluations':0,'test_evaluations':0,'formal_records':0})
    finally:
        # Keep the event watcher through request release, so this last interval
        # is observed too. Cleanup errors remain failures even after handle close.
        for cleanup in (power_request.release,power_watcher.stop):
            try:
                cleanup()
            except BaseException as error:
                reasons.append('runtime cleanup failed: '+repr(error))
        reasons.extend(poll_runtime())
    power, watcher = power_request.snapshot(), power_watcher.snapshot()
    if (not power['acquired'] or not power['released'] or power['cleanup_errors']
            or not watcher['registered'] or not watcher['stopped'] or watcher['subscription_active']
            or watcher['cleanup_errors'] or watcher['events']):
        reasons.append('complete native power-request/watcher lifecycle failed')
    reasons = list(dict.fromkeys(reasons))
    runtime = {'status':'passed' if not reasons and not ordinary_pause else 'paused' if ordinary_pause and not reasons else 'failed',
               'stop_reasons':reasons,'worker_count':len(candidate['jobs']) if candidate and not args.reliability_only else 0,
               'idle_sleep_request_acquired':power['acquired'],'idle_sleep_request_released':power['released'],
               'power_request':power,'power_event_watcher':watcher,'body_supervision':job}
    write_exclusive(output/'runtime_guardian.json',runtime)
    current = ledger_view(ledger.poll(force=True))
    reasons.extend(current['stop_reasons'])
    if candidate and not args.reliability_only:
        candidate = apply_runtime_acceptance(candidate,runtime,current)
        if candidate['failures']:
            reasons.append('resource acceptance candidate failed final runtime/ledger checks')
    status = 'paused' if ordinary_pause and not reasons else 'failed' if reasons else 'passed'
    completion = {'phase':phase,'run_id':config['run_id'],'status':status,'source_commit':commit,
                  'authoritative_only_with_closed_ledger':True,'attempt_id':attempt_id,
                  'manifest_sha256':file_hash(output/'acceptance_manifest.json'),
                  'runtime_guardian_sha256':file_hash(output/'runtime_guardian.json'),
                  'candidate':candidate,'stop_reasons':list(dict.fromkeys(reasons)),
                  'formal_launch_authorized':False,'formal_training_enabled':False,
                  'validation_evaluations':0,'test_evaluations':0,'formal_records':0}
    no_scores(completion)
    write_exclusive(output/'acceptance_completion.json',completion)
    # This closed event binds the completion bytes. A last-interval limit or
    # failed durable close invalidates the candidate, irrespective of its status.
    try:
        closed = ledger.close_attempt(attempt_id,outcome='completed' if status=='passed' else 'external_interruption' if status=='paused' else 'failed',
            record_sha256=file_hash(output/'acceptance_completion.json'),reason='; '.join(reasons) or ('whole-model ordinary pause' if status=='paused' else None))
    except BaseException as error:
        status = 'failed'
        reasons.append('final ledger close failed: '+repr(error))
        write_exclusive(output/'ledger_close_failure.json',{'error':repr(error),'stop_reasons':reasons,
            'candidate_is_not_authoritative':True,'validation_evaluations':0,'test_evaluations':0,'formal_records':0})
        # A guard may reject a successful close before its event is written.
        # Record that still-owned open attempt as failed, never retry the work.
        if ledger.snapshot()['attempts'][attempt_id]['outcome']=='open':
            closed = ledger.close_attempt(attempt_id,outcome='failed',
                record_sha256=file_hash(output/'acceptance_completion.json'),reason='; '.join(reasons))
        else:
            # A torn durable close requires independent inspection; no receipt
            # is emitted that could be mistaken for a finalized journal head.
            return 2
    receipt = {'status':'passed' if status=='passed' and not closed['must_stop'] else status if status!='passed' else 'failed',
               'phase':phase,'attempt_id':attempt_id,'completion_sha256':file_hash(output/'acceptance_completion.json'),
               'head':closed['head'],'ledger':ledger_view(closed),
               'validation_evaluations':0,'test_evaluations':0,'formal_records':0}
    write_exclusive(output/'closed_ledger_receipt.json',receipt)
    print(json.dumps({'status':receipt['status'],'phase':phase,'charged_seconds':closed['charged_seconds'],
                      'stop_reasons':list(dict.fromkeys(reasons+closed['stop_reasons'])),
                      'receipt':str(output/'closed_ledger_receipt.json')}),flush=True)
    return 0 if receipt['status']=='passed' else PAUSE_EXIT if receipt['status']=='paused' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--binding',type=Path,default=DEFAULT_BINDING)
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--ci-receipt',type=Path)
    parser.add_argument('--budget-ledger',type=Path)
    parser.add_argument('--ledger-head-receipt',type=Path)
    parser.add_argument('--reliability-only',action='store_true')
    parser.add_argument('--request-pause',action='store_true')
    parser.add_argument('--emergency-stop',action='store_true')
    parser.add_argument('--body',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--worker-id',help=argparse.SUPPRESS)
    parser.add_argument('--synthetic-worker',choices=['stall','exit','unit'],help=argparse.SUPPRESS)
    parser.add_argument('--case-dir',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--trial-seconds',type=float,default=0.1,help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.synthetic_worker:
        return synthetic_worker(args)
    require(args.output_root is not None,'--output-root is required')
    if args.request_pause or args.emergency_stop:
        require(args.request_pause != args.emergency_stop,'choose one control action')
        require((args.output_root/'acceptance_manifest.json').exists(),'target must be an existing acceptance run')
        name = 'pause_request.json' if args.request_pause else 'emergency_stop.json'
        write_exclusive(args.output_root/'control'/name,{'requested_at':time.time(),
            'action':'finish current whole model unit then exit' if args.request_pause else 'emergency owned-process stop'})
        return 0
    require(args.data_root is not None,'--data-root is required')
    config, authorization = read_json(args.config), read_json(DEFAULT_AUTHORIZATION)
    validate_approved_config(config,authorization)
    if args.worker_id:
        manifest = read_json(args.output_root/'acceptance_manifest.json')
        verify_manifest(manifest,config,authorization,args.binding)
        require(identity_is_running(manifest['guardian_identity']), 'live outer guardian required')
        worker = next(w for w in preflight.workers(config) if w['worker_id']==args.worker_id)
        destination = args.output_root/'workers'/worker['worker_id']
        try:
            run_probe_worker(config,worker,args.data_root,read_json(args.binding),destination)
            return 0
        except GracefulPause:
            return PAUSE_EXIT
        except BaseException as error:
            write_exclusive(destination/'worker_failure.json',{'error':repr(error),'traceback':traceback.format_exc(),
                'validation_evaluations':0,'test_evaluations':0,'formal_records':0})
            return 2
    if args.body:
        return reliability_body(args,config,authorization) if args.reliability_only else acceptance_body(args,config,authorization)
    return run_guardian(args,config,authorization)


if __name__=='__main__':
    raise SystemExit(main())
