"""V22 controlled source-sampling intervention on the audited v21 route.

Only the train dataloader sampler changes. The v21 partial-feature schedule,
freeze mask, feature-BN policy, and checkpoint initialization stay identical.
"""
import argparse
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

from data_integrity import sha256
from build_kitchen_source_extension_v20 import read, write, verify
from train_kitchen_partial_features_v21 import (
    DATA_SHA,
    DEPENDENCIES as V21_DEPENDENCIES,
    FEATURES,
    PREFIX,
    PRIMARY,
    START_SHA,
    PartialFeatureAudit,
    frozen_schedule,
)
from train_scratch_semantic_v16 import verify_trainer_reconstruction


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_META = ROOT / 'model_work/runs/fire-kitchen-partial-features-v21-20261028/experiment_meta.json'
DEPENDENCIES = ['train_kitchen_partial_features_v21.py'] + V21_DEPENDENCIES

DRAWS_PER_EPOCH = 2409
BATCH_SIZE = 8
EPOCHS = 12
NOMINAL_BATCHES_PER_EPOCH = 302
NOMINAL_BATCHES = EPOCHS * NOMINAL_BATCHES_PER_EPOCH
if (DRAWS_PER_EPOCH + BATCH_SIZE - 1) // BATCH_SIZE != NOMINAL_BATCHES_PER_EPOCH:
    raise ValueError('Frozen draw count must produce 302 batches per epoch')
if NOMINAL_BATCHES != 3624:
    raise ValueError('Frozen v22 nominal batch total must be 3624')
SAMPLER_SEED = 20261028
LOADER_SEED_BASE = 6148914691236517205
STRATA_MASS = {
    'new4_positives': 0.20,
    'new10_negatives': 0.20,
    'remaining': 0.60,
}
EXPECTED_IMAGES = {
    'new4_positives': 4,
    'new10_negatives': 10,
}
EXPECTED_ENTRIES = {
    'new4_positives': 32,
    'new10_negatives': 36,
    'remaining': 2341,
}


def _value_sha256(value):
    payload = json.dumps(value, ensure_ascii=True, separators=(',', ':'), sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _sequence_sha256(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()


def _resolved(value):
    return Path(str(value)).resolve()


def _counter_difference(actual, expected):
    return {
        'missing': sorted(str(path) for path in expected.keys() - actual.keys())[:10],
        'extra': sorted(str(path) for path in actual.keys() - expected.keys())[:10],
        'count_mismatch': [
            [str(path), actual[path], expected[path]]
            for path in sorted(actual.keys() & expected.keys())
            if actual[path] != expected[path]
        ][:10],
    }


class SourceSamplingPlan:
    """Membership and weights derived only from the verified v20 manifest."""

    def __init__(self, folder, manifest_sha256, rows):
        self.folder = Path(folder).resolve()
        self.manifest_sha256 = manifest_sha256
        self.rows = rows
        self.train_dir = self.folder / 'images/train'
        self._build()

    @classmethod
    def from_verified_manifest(cls, folder, manifest_sha256):
        manifest = read(Path(folder) / 'build_manifest.json')
        if sha256(Path(folder) / 'build_manifest.json') != manifest_sha256:
            raise ValueError('Verified v20 manifest changed before sampling-plan construction')
        return cls(folder, manifest_sha256, manifest['lineage'])

    def _build(self):
        names = []
        self.path_to_stratum = {}
        self.path_weights = {}
        image_counts = Counter()
        entry_counts = Counter()

        for row in self.rows:
            name = row.get('image')
            weight = row.get('weight')
            if not isinstance(name, str) or not name or Path(name).name != name:
                raise ValueError('Manifest contains an invalid image name')
            if isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0:
                raise ValueError('Manifest contains an invalid entry weight')
            if name in self.path_weights:
                raise ValueError('Manifest contains a duplicate image name')

            new_source = row.get('v20_new_source', False)
            if not isinstance(new_source, bool):
                raise ValueError('Unknown v20_new_source classification')
            target = row.get('visible_target')
            if target not in {None, 'flame', 'visible_no_flame'}:
                raise ValueError('Unknown visible_target classification')

            if new_source:
                if target == 'flame':
                    stratum = 'new4_positives'
                elif target == 'visible_no_flame':
                    stratum = 'new10_negatives'
                else:
                    raise ValueError('New source has unknown or missing visible_target')
            else:
                stratum = 'remaining'

            path = (self.train_dir / name).resolve()
            self.path_to_stratum[path] = stratum
            self.path_weights[path] = weight
            names.append(name)
            image_counts[stratum] += 1
            entry_counts[stratum] += weight

        if len(names) != 1445 or sum(entry_counts.values()) != DRAWS_PER_EPOCH:
            raise ValueError('Verified manifest unique/entry totals changed')
        if image_counts != Counter(EXPECTED_IMAGES) + Counter({'remaining': 1431}):
            raise ValueError('Expected new4 positives and new10 negatives are not present')
        if entry_counts != Counter(EXPECTED_ENTRIES):
            raise ValueError('New/remaining manifest entry weights differ from the frozen baseline')

        self.unique_images = image_counts
        self.entries = entry_counts
        self.per_entry_weight = {
            stratum: STRATA_MASS[stratum] / entry_counts[stratum]
            for stratum in STRATA_MASS
        }
        if not 0.0 < sum(STRATA_MASS.values()) == 1.0:
            raise ValueError('Sampling masses must sum to one')
        self.member_names = {
            stratum: sorted(
                path.name for path, value in self.path_to_stratum.items() if value == stratum
            )
            for stratum in STRATA_MASS
        }
        self.image_weight_pairs = [
            [stratum, path.name, self.per_entry_weight[stratum]]
            for stratum in STRATA_MASS
            for path in sorted(self.path_weights)
            if self.path_to_stratum[path] == stratum
        ]

    def verify_im_files(self, im_files):
        if len(im_files) != DRAWS_PER_EPOCH:
            raise ValueError('Train dataset entry count differs from 2409')
        actual = Counter(_resolved(path) for path in im_files)
        expected = Counter(self.path_weights)
        if actual != expected:
            raise ValueError(f'Train dataset paths differ from manifest: {_counter_difference(actual, expected)}')
        unknown = set(actual) - set(self.path_to_stratum)
        if unknown:
            raise ValueError(f'Train dataset contains unknown manifest paths: {sorted(map(str, unknown))[:10]}')
        return actual

    def build_sampler(self, im_files):
        import torch
        from torch.utils.data import WeightedRandomSampler

        self.verify_im_files(im_files)
        resolved = [_resolved(path) for path in im_files]
        weights = [self.per_entry_weight[self.path_to_stratum[path]] for path in resolved]
        generator = torch.Generator()
        generator.manual_seed(SAMPLER_SEED)

        class RecordingWeightedRandomSampler(WeightedRandomSampler):
            def __init__(self):
                super().__init__(
                    torch.tensor(weights, dtype=torch.double),
                    num_samples=DRAWS_PER_EPOCH,
                    replacement=True,
                    generator=generator,
                )
                self.draw_records = []

            def __iter__(self):
                record = []
                self.draw_records.append(record)
                for index in super().__iter__():
                    value = int(index)
                    record.append(value)
                    yield value

        sampler = RecordingWeightedRandomSampler()
        if len(sampler) != DRAWS_PER_EPOCH:
            raise ValueError('WeightedRandomSampler length differs from 2409')
        return sampler, weights

    def public_plan(self, entry_weights):
        return {
            'scope': 'Controlled v22 train-sampler-only intervention; no new data or labels',
            'manifest_sha256': self.manifest_sha256,
            'membership_rule': {
                'new4_positives': "v20_new_source=true and visible_target='flame'",
                'new10_negatives': "v20_new_source=true and visible_target='visible_no_flame'",
                'remaining': 'all other verified manifest rows',
                'model_outputs_or_gt_filenames_used': False,
            },
            'strata': {
                stratum: {
                    'mass': STRATA_MASS[stratum],
                    'unique_images': self.unique_images[stratum],
                    'entries': self.entries[stratum],
                    'entry_weight': self.per_entry_weight[stratum],
                    'expected_draws_per_epoch': STRATA_MASS[stratum] * DRAWS_PER_EPOCH,
                    'member_names_sha256': _sequence_sha256(self.member_names[stratum]),
                }
                for stratum in STRATA_MASS
            },
            'sampler': {
                'class': 'WeightedRandomSampler',
                'replacement': True,
                'num_samples': DRAWS_PER_EPOCH,
                'generator_seed': SAMPLER_SEED,
                'loader_generator_seed_base': LOADER_SEED_BASE,
                'loader_generator_seed_rank_expression': f'{LOADER_SEED_BASE} + RANK',
                'batch_size': BATCH_SIZE,
                'nominal_batches_per_epoch': NOMINAL_BATCHES_PER_EPOCH,
                'drop_last': False,
            },
            'entry_weights_sha256': _sequence_sha256(
                [_resolved(path).name + '=' + repr(weight) for path, weight in zip(
                    sorted(self.path_weights), [self.per_entry_weight[self.path_to_stratum[path]]
                                                for path in sorted(self.path_weights)]
                )]
            ),
            'image_weight_pairs_sha256': _value_sha256(self.image_weight_pairs),
            'manifest_entry_weights_sha256': _sequence_sha256(repr(value) for value in entry_weights),
            'validated_unique_images': sum(self.unique_images.values()),
            'validated_entries': sum(self.entries.values()),
        }


class SourceSamplingAudit:
    """Audit sampled indices and the im_file sequence actually consumed."""

    def __init__(self, out, plan):
        self.out = Path(out)
        self.plan = plan
        self.sampler = None
        self.im_files = None
        self.dataset_entry_weights_sha256 = None
        self.loader_rank = None
        self.loader_generator_seed = None
        self.consumed = {}
        self.previous_batches = 0
        self.previous_steps = 0
        self.epochs = []

    def bind_train_loader(self, dataset, sampler, weights, rank, loader_generator_seed):
        if self.sampler is not None:
            raise RuntimeError('Train sampling audit already bound')
        im_files = [_resolved(path) for path in dataset.im_files]
        self.plan.verify_im_files(im_files)
        if len(dataset) != DRAWS_PER_EPOCH:
            raise ValueError('Train dataset length differs from 2409')
        if len(weights) != DRAWS_PER_EPOCH:
            raise ValueError('Train dataset weight count differs from 2409')
        self.sampler = sampler
        self.im_files = im_files
        self.dataset_entry_weights_sha256 = _sequence_sha256(repr(value) for value in weights)
        self.loader_rank = rank
        self.loader_generator_seed = loader_generator_seed

    def record_batch(self, trainer, batch):
        files = batch.get('im_file')
        if not files:
            raise RuntimeError('Train preprocess_batch received no im_file')
        epoch = int(trainer.epoch)
        if epoch in self.consumed and any(record['epoch'] == epoch + 1 for record in self.epochs):
            raise RuntimeError('Train epoch was consumed after its audit closed')
        self.consumed.setdefault(epoch, []).extend(_resolved(path) for path in files)

    def epoch_end(self, trainer, cadence):
        epoch = int(trainer.epoch)
        records = self.sampler.draw_records
        if len(records) != epoch + 1:
            raise RuntimeError('WeightedRandomSampler draw-record count differs from epoch count')
        indices = records[epoch]
        if len(indices) != DRAWS_PER_EPOCH:
            raise RuntimeError('WeightedRandomSampler did not create 2409 draws')
        if any(index < 0 or index >= len(self.im_files) for index in indices):
            raise RuntimeError('WeightedRandomSampler produced an out-of-range index')

        expected_files = [self.im_files[index] for index in indices]
        consumed = self.consumed.get(epoch, [])
        if len(consumed) != DRAWS_PER_EPOCH:
            raise RuntimeError('Actual consumed im_file count differs from 2409')
        if consumed != expected_files:
            raise RuntimeError('Actual consumed im_file sequence differs from sampler draws')

        draw_counts = Counter(path.name for path in expected_files)
        consumed_counts = Counter(path.name for path in consumed)
        if draw_counts != consumed_counts:
            raise RuntimeError('Sampled and consumed per-image counts differ')
        stratum_counts = Counter(self.plan.path_to_stratum[path] for path in consumed)
        if set(stratum_counts) != set(STRATA_MASS) or sum(stratum_counts.values()) != DRAWS_PER_EPOCH:
            raise RuntimeError('Actual stratum draw counts are incomplete')

        batches = cadence.batches - self.previous_batches
        steps = len(cadence.steps) - self.previous_steps
        if batches != NOMINAL_BATCHES_PER_EPOCH:
            raise RuntimeError('Actual epoch batch count differs from 302')
        if steps < 0 or steps > batches:
            raise RuntimeError('Actual optimizer-step cadence is invalid')

        record = {
            'epoch': epoch + 1,
            'draws': len(indices),
            'actual_consumed_im_files': len(consumed),
            'actual_batches': batches,
            'actual_optimizer_steps': steps,
            'draw_index_sequence_sha256': _sequence_sha256(indices),
            'draw_im_file_sequence_sha256': _sequence_sha256(str(path) for path in expected_files),
            'consumed_im_file_sequence_sha256': _sequence_sha256(str(path) for path in consumed),
            'per_stratum_counts': {name: stratum_counts[name] for name in STRATA_MASS},
            'actual_consumed_im_file_counts': dict(sorted(consumed_counts.items())),
        }
        self.epochs.append(record)
        self.previous_batches = cadence.batches
        self.previous_steps = len(cadence.steps)
        self.save()

    def save(self):
        actual_batches = sum(record['actual_batches'] for record in self.epochs)
        actual_optimizer_steps = sum(
            record['actual_optimizer_steps'] for record in self.epochs
        )
        write(self.out / 'source_sampling_actual.json', {
            'scope': 'Actual sampler draws and consumed train im_file names; expanded sequences are represented by hashes',
            'manifest_sha256': self.plan.manifest_sha256,
            'sampler_seed': SAMPLER_SEED,
            'num_samples': DRAWS_PER_EPOCH,
            'dataset_entry_weights_sha256': self.dataset_entry_weights_sha256,
            'rank': self.loader_rank,
            'loader_generator_seed': self.loader_generator_seed,
            'epochs': self.epochs,
            'actual_batches': actual_batches,
            'actual_optimizer_steps': actual_optimizer_steps,
        })

    def compact_summary(self):
        return {
            'sampler_seed': SAMPLER_SEED,
            'num_samples_per_epoch': DRAWS_PER_EPOCH,
            'rank': self.loader_rank,
            'loader_generator_seed': self.loader_generator_seed,
            'epochs': [
                {key: value for key, value in record.items()
                 if key != 'actual_consumed_im_file_counts'}
                for record in self.epochs
            ],
            'actual_batches': self.previous_batches,
            'actual_optimizer_steps': self.previous_steps,
            'audit_sha256': sha256(self.out / 'source_sampling_actual.json'),
        }


def trainer_class(plan, sampling_audit):
    import torch
    from ultralytics.data.build import InfiniteDataLoader
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.utils import RANK

    class SourceSamplingTrainer(DetectionTrainer):
        source_sampling_plan = plan
        source_sampling_audit = sampling_audit

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
            if mode != 'train':
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            if rank not in {-1, 0}:
                raise ValueError(f'Only single-process rank -1/0 is allowed, got {rank}')
            if getattr(self, 'world_size', 1) != 1:
                raise ValueError('DDP is not allowed for v22')
            if self.args.workers != 0:
                raise ValueError('V22 requires workers=0')
            if batch_size != BATCH_SIZE:
                raise ValueError('V22 requires batch size 8')

            dataset = self.build_dataset(dataset_path, mode, batch_size)
            sampler, weights = self.source_sampling_plan.build_sampler(dataset.im_files)
            loader_generator = torch.Generator()
            loader_seed = LOADER_SEED_BASE + RANK
            loader_generator.manual_seed(loader_seed)
            self.source_sampling_audit.bind_train_loader(
                dataset,
                sampler,
                weights,
                rank,
                loader_seed,
            )
            loader = InfiniteDataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                sampler=sampler,
                pin_memory=True,
                collate_fn=dataset.collate_fn,
                drop_last=False,
                generator=loader_generator,
            )
            if len(loader) != NOMINAL_BATCHES_PER_EPOCH:
                raise RuntimeError('V22 train loader must expose 302 nominal batches')
            return loader

        def preprocess_batch(self, batch):
            self.source_sampling_audit.record_batch(self, batch)
            return super().preprocess_batch(batch)

    return SourceSamplingTrainer


def _schedule_without_output(config):
    return {key: value for key, value in config.items() if key not in {'name', 'project'}}


def validate_reference(reference, data):
    required = {
        'source_sha256': START_SHA,
        'candidate_training': True,
        'frozen_prefix_blocks': PREFIX,
        'feature_blocks': FEATURES,
        'feature_bn_statistics_fixed': True,
        'checkpoint_screen': PRIMARY,
    }
    for key, value in required.items():
        if reference.get(key) != value:
            raise ValueError(f'V21 reference identity differs: {key}')
    if reference['dataset']['manifest_sha256'] != DATA_SHA:
        raise ValueError('V21 reference dataset differs')
    if reference['config']['data'] != str(data.resolve()):
        raise ValueError('V21 reference data path differs')
    if reference['config']['epochs'] != EPOCHS or reference['config']['batch'] != BATCH_SIZE:
        raise ValueError('V21 reference schedule differs')


def validate_completion(meta, partial_epochs, cadence, sampling):
    if meta['source_sha256'] != START_SHA or meta['dataset']['manifest_sha256'] != DATA_SHA:
        raise ValueError('Wrong completed v22 source/data')
    if meta['config']['epochs'] != EPOCHS or meta['config']['batch'] != BATCH_SIZE:
        raise ValueError('Wrong completed v22 schedule')
    if meta['config']['freeze'] != PREFIX or meta['frozen_prefix_blocks'] != PREFIX:
        raise ValueError('Wrong completed freeze route')
    if meta['feature_blocks'] != FEATURES or not meta['feature_bn_statistics_fixed']:
        raise ValueError('Wrong completed feature route')
    if set(meta['checkpoints']) != set(PRIMARY) or not meta['serialized_frozen_state_equal']:
        raise ValueError('Both audited best/last checkpoints are required')
    if set(meta['serialized_adaptation']) != set(PRIMARY):
        raise ValueError('Saved checkpoint adaptation identity is incomplete')

    if len(partial_epochs) != EPOCHS or [row['epoch'] for row in partial_epochs] != list(range(1, 13)):
        raise ValueError('Partial-feature audit is incomplete')
    if not all(
        row['frozen_prefix_unchanged']
        and row['feature_bn_unchanged']
        and row['changed_feature_weight_tensors'] > 0
        and row['changed_head_weight_tensors'] > 0
        and row['ema_fixed_state_restored']
        for row in partial_epochs
    ):
        raise ValueError('Partial-feature update audit failed')

    sampling_epochs = sampling.get('epochs', [])
    if len(sampling_epochs) != EPOCHS or [row['epoch'] for row in sampling_epochs] != list(range(1, 13)):
        raise ValueError('Source-sampling audit is incomplete')
    for row in sampling_epochs:
        if row['draws'] != DRAWS_PER_EPOCH or row['actual_consumed_im_files'] != DRAWS_PER_EPOCH:
            raise ValueError('A source-sampling epoch did not consume 2409 images')
        if row['actual_batches'] != NOMINAL_BATCHES_PER_EPOCH:
            raise ValueError('A source-sampling epoch did not contain 302 batches')
        if sum(row['per_stratum_counts'].values()) != DRAWS_PER_EPOCH:
            raise ValueError('A source-sampling epoch has incomplete stratum counts')

    if meta['actual_batches'] != NOMINAL_BATCHES:
        raise ValueError('Actual batch total differs from 3624')
    if cadence['batches'] != meta['actual_batches']:
        raise ValueError('Optimizer cadence batch total differs')
    if cadence['actual_optimizer_steps'] != meta['actual_optimizer_steps']:
        raise ValueError('Optimizer cadence step total differs')
    if not 0 < meta['actual_optimizer_steps'] <= meta['actual_batches']:
        raise ValueError('Actual optimizer-step total is invalid')
    sampling_batches = sum(row['actual_batches'] for row in sampling_epochs)
    sampling_steps = sum(row['actual_optimizer_steps'] for row in sampling_epochs)
    if sampling_batches != cadence['batches'] or sampling_steps != len(cadence['steps']):
        raise ValueError('Sampling and optimizer cadence totals differ')


def checkpoint_adaptation(audit, fixed, torch):
    from backbone_freeze_audit import is_backbone_key
    from ultralytics import YOLO

    checkpoints = sorted((audit.out / 'weights').glob('*.pt'))
    if {path.name for path in checkpoints} != set(PRIMARY):
        raise ValueError('Expected exactly best.pt and last.pt')
    identities = {}
    adaptations = {}
    for checkpoint in checkpoints:
        live = YOLO(str(checkpoint)).model.state_dict()
        if any(not torch.equal(live[key].detach().cpu(), value) for key, value in fixed.items()):
            raise RuntimeError('Saved fixed prefix/feature-BN/DFL state changed')
        changed = [
            key for key, value in live.items()
            if key.endswith('.weight') and not torch.equal(value.detach().cpu(), audit.initial[key])
        ]
        counts = {
            'feature_weight_tensors': sum(
                is_backbone_key(key, FEATURES) and not is_backbone_key(key, PREFIX)
                for key in changed
            ),
            'head_weight_tensors': sum(key.startswith('model.23.') for key in changed),
        }
        if min(counts.values()) <= 0:
            raise RuntimeError('Saved checkpoint lacks feature/head adaptation')
        identities[checkpoint.name] = sha256(checkpoint)
        adaptations[checkpoint.name] = counts
    return identities, adaptations


def finalize_existing(out):
    """Audit a completed v22 run without invoking the trainer."""
    from types import SimpleNamespace

    import torch
    from backbone_freeze_audit import is_backbone_key, state_digest
    from ultralytics import YOLO

    out = Path(out).resolve()
    marker = out.parent / (out.name + '_preflight.json')
    if not out.is_dir() or not marker.is_file():
        raise ValueError('Completed run and preflight marker are required')
    if (out / 'failed.json').exists():
        raise ValueError('A failed marker exists for this run')
    if (out / 'experiment_meta.json').exists():
        raise ValueError('experiment_meta.json already exists')

    identity = read(marker)
    if identity.get('role') != (
        'V22 controlled train-sampler-only intervention on the audited v21 partial-feature route'
    ):
        raise ValueError('Wrong preflight experiment identity')
    snapshot = out / 'source_snapshot'
    if sha256(snapshot / Path(__file__).name) != identity['script_sha256']:
        raise ValueError('Completed run script snapshot differs from preflight')
    for name, expected in identity['dependencies'].items():
        if sha256(snapshot / name) != expected:
            raise ValueError(f'Completed run dependency snapshot differs: {name}')
    if _value_sha256(read(out / 'source_sampling_plan.json')) != (
        identity['sampling']['source_sampling_plan_value_sha256']
    ):
        raise ValueError('Completed source-sampling plan changed')

    start = Path(identity['config']['pretrained']).resolve()
    if sha256(start) != identity['source_sha256']:
        raise ValueError('Completed run source checkpoint changed')
    core = YOLO(str(start)).model
    initial = {
        key: value.detach().cpu().clone()
        for key, value in core.state_dict().items()
    }
    initialization = read(out / 'actual_initialization_audit.json')
    if (
        not initialization['exact_v16_best_state_equal']
        or initialization['state_tensors'] != len(initial)
        or initialization['state_sha256'] != state_digest(initial)
    ):
        raise ValueError('Completed run initialization audit differs from source')

    fixed = {
        key: value
        for key, value in initial.items()
        if is_backbone_key(key, PREFIX)
        or (
            is_backbone_key(key, FEATURES)
            and key.endswith(('running_mean', 'running_var', 'num_batches_tracked'))
        )
        or key == 'model.23.dfl.conv.weight'
    }
    audit = SimpleNamespace(out=out, initial=initial)
    checkpoints, adaptations = checkpoint_adaptation(audit, fixed, torch)

    sampling = read(out / 'source_sampling_actual.json')
    cadence = read(out / 'optimizer_cadence_audit.json')
    partial_epochs = read(out / 'partial_feature_update_audit.json')
    sampling_epochs = sampling.get('epochs', [])
    sampling_summary = {
        'sampler_seed': sampling['sampler_seed'],
        'num_samples_per_epoch': sampling['num_samples'],
        'rank': sampling['rank'],
        'loader_generator_seed': sampling['loader_generator_seed'],
        'epochs': [
            {
                key: value
                for key, value in record.items()
                if key != 'actual_consumed_im_file_counts'
            }
            for record in sampling_epochs
        ],
        'actual_batches': sum(row['actual_batches'] for row in sampling_epochs),
        'actual_optimizer_steps': sum(
            row['actual_optimizer_steps'] for row in sampling_epochs
        ),
        'audit_sha256': sha256(out / 'source_sampling_actual.json'),
    }

    identity.update(
        {
            'checkpoints': checkpoints,
            'actual_batches': cadence['batches'],
            'actual_optimizer_steps': len(cadence['steps']),
            'sampling': {**identity['sampling'], **sampling_summary},
            'serialized_frozen_state_equal': True,
            'serialized_adaptation': adaptations,
            'completed_audits': {
                name: sha256(out / name)
                for name in [
                    'source_sampling_actual.json',
                    'actual_initialization_audit.json',
                    'frozen_prefix_audit.json',
                    'partial_feature_update_audit.json',
                    'optimizer_cadence_audit.json',
                ]
            },
        }
    )
    validate_completion(identity, partial_epochs, cadence, sampling)
    write(out / 'experiment_meta.json', identity)
    print(
        f'V22 audited without retraining: draws={DRAWS_PER_EPOCH}, '
        f'batches={identity["actual_batches"]}, '
        f'optimizer_steps={identity["actual_optimizer_steps"]}; '
        'best/last still require fixed held-out comparisons',
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference-meta', type=Path, default=DEFAULT_REFERENCE_META)
    parser.add_argument(
        '--finalize-existing',
        action='store_true',
        help='Audit and finalize an already completed run without training',
    )
    args = parser.parse_args()

    if args.finalize_existing:
        finalize_existing(args.out)
        return
    if args.start is None or args.data is None:
        parser.error('--start and --data are required unless --finalize-existing is used')

    import torch
    import ultralytics
    from ultralytics import YOLO

    start = args.start.resolve()
    data = args.data.resolve()
    out = args.out.resolve()
    reference_path = args.reference_meta.resolve()
    marker = out.parent / (out.name + '_preflight.json')
    if out.exists() or marker.exists():
        raise ValueError('Existing run or preflight marker')
    if ultralytics.__version__ != '8.4.70':
        raise RuntimeError(f'V22 requires Ultralytics 8.4.70, got {ultralytics.__version__}')
    if sha256(start) != START_SHA:
        raise ValueError('Source checkpoint changed')

    dataset = verify(data.parent, DATA_SHA)
    plan = SourceSamplingPlan.from_verified_manifest(data.parent, DATA_SHA)
    reference = read(reference_path)
    validate_reference(reference, data)

    model = YOLO(str(start))
    if model.names != {0: 'fire'}:
        raise ValueError('Wrong class identity')
    rebuilt = verify_trainer_reconstruction(model, start)

    config = frozen_schedule(data, out, start, PREFIX)
    if config['workers'] != 0 or config['device'] != 0:
        raise ValueError('Frozen v21 schedule has an unexpected runtime configuration')
    reference_config = _schedule_without_output(reference['config'])
    candidate_config = _schedule_without_output(config)
    if reference_config != candidate_config:
        raise ValueError('V22 schedule differs from v21 beyond output paths')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')

    audit = PartialFeatureAudit(model.model, out)
    sampling_audit = SourceSamplingAudit(out, plan)
    SourceSamplingTrainer = trainer_class(plan, sampling_audit)

    # Constructing the sampler here validates membership and fixes the exact
    # entry-weight bytes before any training-side dataset is built.
    manifest_entries = []
    for stratum in STRATA_MASS:
        for name in plan.member_names[stratum]:
            manifest_entries.extend(
                [plan.per_entry_weight[stratum]]
                * plan.path_weights[(plan.train_dir / name).resolve()]
            )
    plan_public = plan.public_plan(manifest_entries)
    source_sampling_plan_value_sha256 = _value_sha256(plan_public)

    identity = {
        'role': 'V22 controlled train-sampler-only intervention on the audited v21 partial-feature route',
        'single_intervention': 'verified-manifest stratified replacement sampling',
        'source_sha256': START_SHA,
        'dataset': dataset,
        'config': config,
        'frozen_prefix_blocks': PREFIX,
        'feature_blocks': FEATURES,
        'feature_bn_statistics_fixed': True,
        'candidate_training': True,
        'checkpoint_screen': PRIMARY,
        'teacher_loss': False,
        'actual_trainer_reconstruction': rebuilt,
        'script_sha256': sha256(Path(__file__)),
        'dependencies': {
            name: sha256(Path(__file__).with_name(name))
            for name in DEPENDENCIES
        },
        'prerequisites': {
            str(reference_path.relative_to(ROOT) if reference_path.is_relative_to(ROOT) else reference_path):
                sha256(reference_path)
        },
        'v21_schedule_equal_except_output_paths': True,
        'sampling': {
            'masses': STRATA_MASS,
            'expected_unique_images': plan.unique_images,
            'expected_entries': plan.entries,
            'source_sampling_plan_value_sha256': source_sampling_plan_value_sha256,
            'entry_weights_sha256': plan_public['entry_weights_sha256'],
            'image_weight_pairs_sha256': plan_public['image_weight_pairs_sha256'],
            'manifest_entry_weights_sha256': plan_public['manifest_entry_weights_sha256'],
        },
        'nominal_batches_per_epoch': NOMINAL_BATCHES_PER_EPOCH,
        'nominal_batches': NOMINAL_BATCHES,
        'actual_batches': None,
        'actual_optimizer_steps': None,
        'checkpoints': None,
        'environment': {
            'python': platform.python_version(),
            'torch': torch.__version__,
            'ultralytics': ultralytics.__version__,
        },
    }
    write(marker, identity)

    source_bytes = {
        name: Path(__file__).with_name(name).read_bytes()
        for name in [Path(__file__).name] + DEPENDENCIES
    }

    def capture_sources(trainer):
        if Path(trainer.save_dir).resolve() != out:
            raise RuntimeError('Trainer changed the intended output directory')
        write(out / 'source_sampling_plan.json', plan_public)
        if _value_sha256(read(out / 'source_sampling_plan.json')) != source_sampling_plan_value_sha256:
            raise RuntimeError('Source-sampling plan changed at training launch')
        identity['sampling']['source_sampling_plan_sha256'] = sha256(
            out / 'source_sampling_plan.json'
        )
        snapshot = out / 'source_snapshot'
        snapshot.mkdir()
        for name, payload in source_bytes.items():
            if Path(__file__).with_name(name).read_bytes() != payload:
                raise RuntimeError('A source dependency changed at training launch')
            (snapshot / name).write_bytes(payload)

    def on_epoch_end(trainer):
        sampling_audit.epoch_end(trainer, audit.cadence)
        audit.epoch_end(trainer)

    model.add_callback('on_train_start', capture_sources)
    model.add_callback('on_train_start', audit.on_start)
    model.add_callback('on_train_batch_start', audit.batch_start)
    model.add_callback('on_train_batch_end', audit.cadence.on_batch_end)
    model.add_callback('on_train_epoch_end', on_epoch_end)

    try:
        model.train(trainer=SourceSamplingTrainer, **config)
    except Exception as error:
        write(out / 'failed.json', {
            'error_type': type(error).__name__,
            'message': str(error),
        })
        raise
    finally:
        audit.cadence.close()

    fixed = {
        **audit.freeze.reference,
        **audit.feature_bn,
        'model.23.dfl.conv.weight': audit.initial['model.23.dfl.conv.weight'],
    }
    checkpoints, adaptations = checkpoint_adaptation(audit, fixed, torch)
    sampling = read(out / 'source_sampling_actual.json')
    cadence = read(out / 'optimizer_cadence_audit.json')

    identity.update({
        'checkpoints': checkpoints,
        'actual_batches': audit.cadence.batches,
        'actual_optimizer_steps': len(audit.cadence.steps),
        'sampling': {
            **identity['sampling'],
            **sampling_audit.compact_summary(),
        },
        'serialized_frozen_state_equal': True,
        'serialized_adaptation': adaptations,
        'completed_audits': {
            name: sha256(out / name)
            for name in [
                'source_sampling_actual.json',
                'actual_initialization_audit.json',
                'frozen_prefix_audit.json',
                'partial_feature_update_audit.json',
                'optimizer_cadence_audit.json',
            ]
        },
    })
    validate_completion(
        identity,
        audit.epochs,
        cadence,
        sampling,
    )
    write(out / 'experiment_meta.json', identity)
    print(
        f'V22 completed: draws={DRAWS_PER_EPOCH}, batches={identity["actual_batches"]}, '
        f'optimizer_steps={identity["actual_optimizer_steps"]}; '
        'best/last still require fixed held-out comparisons',
        flush=True,
    )


if __name__ == '__main__':
    main()
