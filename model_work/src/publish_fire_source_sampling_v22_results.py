"""Publish v22 training, comparison, video and visual-review evidence."""
import argparse
import json
from pathlib import Path

from data_integrity import sha256


ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )


def source_counts(summary):
    return {
        key: sum(row[key] for row in summary['frames'])
        for key in ('tp', 'fp', 'fn')
    }


def video_counts(comparison, label):
    clips = [clip['model_counts'][label] for clip in comparison['clips']]
    return dict(
        frames_with_any_box=sum(clip['frames_with_any_box'] for clip in clips),
        boxes=sum(clip['boxes'] for clip in clips),
        longest_empty_sample_run=max(clip['longest_empty_sample_run'] for clip in clips),
    )


def publish(training, comparison, videos, review, differences, json_out, markdown_out):
    if json_out.exists() or markdown_out.exists():
        raise ValueError('Refusing to overwrite published results')
    training = Path(training).resolve()
    comparison = Path(comparison).resolve()
    videos = Path(videos).resolve()
    review = Path(review).resolve()
    differences = Path(differences).resolve()

    meta = read(training / 'experiment_meta.json')
    image_result = read(comparison / 'comparison_complete.json')
    video_result = read(videos / 'comparison_complete.json')
    review_inputs = read(review / 'review_inputs.json')
    difference_inputs = read(differences / 'review_inputs.json')
    if not meta['serialized_frozen_state_equal']:
        raise ValueError('Saved checkpoint freeze audit failed')
    if image_result['plan']['source_experiment_meta_sha256'] != sha256(
        training / 'experiment_meta.json'
    ):
        raise ValueError('Image comparison is not bound to training metadata')
    if review_inputs['image_comparison_sha256'] != sha256(
        comparison / 'comparison_complete.json'
    ):
        raise ValueError('Visual review is not bound to image comparison')
    if review_inputs['video_comparison_sha256'] != sha256(
        videos / 'comparison_complete.json'
    ):
        raise ValueError('Visual review is not bound to video comparison')
    if difference_inputs['comparison_sha256'] != sha256(
        videos / 'comparison_complete.json'
    ):
        raise ValueError('Difference review is not bound to video comparison')
    if video_result['total_samples'] != 307:
        raise ValueError('Wrong shared video scope')

    models = {}
    for label, row in image_result['models'].items():
        new_source = source_counts(row['new_training_sources'])
        tight = row['visible_flame']['supplemental']['0.5']
        models[label] = dict(
            weights_sha256=row['weights_sha256'],
            blue_localized_iou50=row['blue']['primary_localized_iou50'],
            tight_flame=dict(
                tp=tight['tp'],
                fp=tight['fp'],
                fn=tight['fn'],
                micro_f1=tight['micro_f1'],
            ),
            indoor_nofire=dict(
                images=row['sources']['nofire_real_indoor']['images'],
                images_with_fp=row['sources']['nofire_real_indoor']['images_with_fp'],
                fp=row['sources']['nofire_real_indoor']['fp'],
            ),
            new_training_sources=dict(**new_source, images=14),
            legacy_map50=row['legacy']['map50'],
            video_307=video_counts(video_result, label),
        )

    fixed_pages = [
        dict(
            kind='fixed',
            file=str(review / page['file']),
            sha256=page['sha256'],
        )
        for page in review_inputs['pages']
    ]
    difference_pages = [
        dict(
            kind='count_difference',
            file=str(differences / page['file']),
            sha256=page['sha256'],
        )
        for page in difference_inputs['sheets']
    ]
    for page in fixed_pages + difference_pages:
        if sha256(Path(page['file'])) != page['sha256']:
            raise ValueError('Reviewed page changed')

    result = dict(
        date='2026-10-28',
        role='V22 controlled train-sampler-only intervention on the v21 partial-feature route',
        training=dict(
            run=str(training),
            source_sha256=meta['source_sha256'],
            epochs=meta['config']['epochs'],
            draws_per_epoch=meta['sampling']['num_samples_per_epoch'],
            actual_batches=meta['actual_batches'],
            actual_optimizer_steps=meta['actual_optimizer_steps'],
            checkpoints=meta['checkpoints'],
            serialized_frozen_state_equal=meta['serialized_frozen_state_equal'],
            serialized_adaptation=meta['serialized_adaptation'],
            completed_audits=meta['completed_audits'],
            environment=meta['environment'],
        ),
        source_sampling=dict(
            masses=meta['sampling']['masses'],
            plan_sha256=sha256(training / 'source_sampling_plan.json'),
            actual_sha256=sha256(training / 'source_sampling_actual.json'),
            epochs=meta['sampling']['epochs'],
        ),
        models=models,
        acceptance=image_result['acceptance'],
        videos=dict(
            path=str(videos),
            shared_samples=video_result['total_samples'],
            model_inferences=video_result['model_inferences'],
            reference_predictions_exact=2149,
            comparison_sha256=sha256(videos / 'comparison_complete.json'),
            per_clip=video_result['clips'],
        ),
        visual_review=dict(
            reviewed_by='Codex main',
            fixed_pages=len(fixed_pages),
            count_difference_pages=len(difference_pages),
            visible_flame_sheets=len(review_inputs['visible_flame_inputs']),
            total_pages_reviewed=len(fixed_pages)
            + len(difference_pages)
            + len(review_inputs['visible_flame_inputs']),
            count_different_samples=difference_inputs['selected_samples'],
            pages=fixed_pages + difference_pages,
            visible_flame_inputs=review_inputs['visible_flame_inputs'],
            observations=[
                'context4_11 now produces a broad blue-burner box that covers the labelled blue ring, explaining a new-source true positive while remaining looser than the tight target.',
                'The 14-source set removes most earlier red-tray, sticker and utensil false positives: v22_best=3/1/0 and v22_last=4/0/0 TP/FP/FN, but these are training-exposed images.',
                'Visible-flame sheets still contain large whole-pot boxes and metal/blue-reflection fragments; v22_best=5/11/10 and v22_last=11/7/4 tight TP/FP/FN.',
                'Video 7 sample 1 has a visible blue flame but v22_last returns no box while both v21 roles return one.',
                'Several v22_last additions in videos 6, 7 and 9 are duplicate, burner-edge or towel/sticker boxes rather than new flame components; v22_best removes some towel boxes at video 9 samples 34/35/40 but also adds duplicate flame boxes.',
            ],
        ),
        decision=dict(
            demo='v16 best retained',
            localization_development='v18 last retained',
            v22_best='not promoted; misses tight-F1 and legacy-mAP gates',
            v22_last='not promoted; misses tight-F1 and legacy-mAP gates',
            reason='Source recall and indoor false positives improve, but tight boxes and legacy mAP regress; video review finds a blue-flame miss and duplicate/false boxes',
            automatic_default_replacement=False,
        ),
        reserved_sources_predicted=False,
        independent_test=False,
        on_board=False,
        onnx_export_this_round=False,
        limits=[
            'All 14 new sources are training-exposed development images.',
            'All videos and old regression images have been exposed repeatedly.',
            'Sampled video box counts are not accuracy or event truth.',
            'No ONNX export, quantization, independent kitchen test or board acceptance was performed.',
        ],
    )

    rows = []
    rows.append('# 火焰 v22 来源分层采样结果｜2026-10-28')
    rows.append('')
    rows.append('## 实验定位')
    rows.append(
        '- v22 只改变训练采样：新4个正例来源、新增10个负例来源、其余来源的期望质量比例为 20%/20%/60%；12轮、每轮2409次抽样、batch=8。'
    )
    rows.append(
        f'- 训练实际完成：{meta["actual_batches"]} batches、{meta["actual_optimizer_steps"]} 次优化更新；冻结前缀、特征BN、DFL、采样和两个检查点审计均通过。'
    )
    rows.append(
        '- 所有图片和视频均为已曝光开发反馈；没有独立测试、ONNX新导出、量化或上板验证。'
    )
    rows.append('')
    rows.append('## 9模型关键结果')
    rows.append('')
    rows.append(
        '| 模型 | 蓝焰/5 | 紧框 TP/FP/FN | 紧框 microF1 | 室内无火FP（图/框） | 新来源14图 TP/FP/FN | 旧mAP50 | 视频有框点/框数 |'
    )
    rows.append('|---|---:|---:|---:|---:|---:|---:|---:|')
    for label, row in models.items():
        tight = row['tight_flame']
        indoor = row['indoor_nofire']
        source = row['new_training_sources']
        video = row['video_307']
        rows.append(
            f'| {label} | {row["blue_localized_iou50"]}/5 | '
            f'{tight["tp"]}/{tight["fp"]}/{tight["fn"]} | '
            f'{tight["micro_f1"]:.4f} | '
            f'{indoor["images_with_fp"]}图/{indoor["fp"]}框 | '
            f'{source["tp"]}/{source["fp"]}/{source["fn"]} | '
            f'{row["legacy_map50"]:.4f} | '
            f'{video["frames_with_any_box"]}/{video["boxes"]} |'
        )
    rows.append('')
    rows.append('## v22验收逐项')
    rows.append('')
    rows.append('| 候选/角色 | 新来源TP≥3,FN≤1,FP≤5 | 紧框F1不低于v21 | 蓝焰≥4/5 | 室内图/框不高于v21 | 旧mAP50不低于v21 | 全部通过 |')
    rows.append('|---|---|---|---|---|---|')
    for candidate, result_row in image_result['acceptance'].items():
        tests = result_row['tests']
        rows.append(
            f'| {candidate} / {result_row["reference"]} | '
            f'{tests["new_source_tp_at_least_3"] and tests["new_source_fn_at_most_1"] and tests["new_source_fp_at_most_5"]} | '
            f'{tests["tight_f1_not_below_v21"]} | '
            f'{tests["blue_localized_at_least_4"]} | '
            f'{tests["indoor_images_not_above_v21"] and tests["indoor_boxes_not_above_v21"]} | '
            f'{tests["legacy_map50_not_below_v21"]} | {result_row["passed"]} |'
        )
    rows.append('')
    rows.append('## 实际看图与视频')
    rows.append(
        '- 实际查看 26 页：4页新来源、9页固定视频、6页紧火焰、7页全部框数差异帧；另有307帧身份和2149条v21参考预测精确复现审计。'
    )
    rows.append(
        '- 收益：新来源和室内无火指标显著改善；context4_11 的蓝焰环首次得到覆盖蓝环的大框。'
    )
    rows.append(
        '- 代价：可见火焰仍保留整锅大框和金属/蓝色反光碎框；v22_last 在视频7约1秒漏掉一个可见蓝焰，并在多帧增加重复框、灶沿框或毛巾/贴纸框。'
    )
    rows.append('')
    rows.append('## 决策')
    rows.append(
        '- 演示继续保留 v16 best，定位开发继续保留 v18 last；v22 best/last 均不晋升。'
    )
    rows.append(
        f'- v22_best：新来源3/1/0、蓝焰4/5、室内6图/6框，但紧框F1 {models["v22_best"]["tight_flame"]["micro_f1"]:.4f}<{models["v21_best"]["tight_flame"]["micro_f1"]:.4f}，旧mAP50 {models["v22_best"]["legacy_map50"]:.4f}<{models["v21_best"]["legacy_map50"]:.4f}。'
    )
    rows.append(
        f'- v22_last：新来源4/0/0、蓝焰4/5、室内13图/16框，但紧框F1 {models["v22_last"]["tight_flame"]["micro_f1"]:.4f}<{models["v21_last"]["tight_flame"]["micro_f1"]:.4f}，旧mAP50 {models["v22_last"]["legacy_map50"]:.4f}<{models["v21_last"]["legacy_map50"]:.4f}。'
    )
    rows.append('')
    rows.append('## 证据入口')
    rows.append(f'- 训练：`{training}`')
    rows.append(f'- 图片对照：`{comparison}`')
    rows.append(f'- 九视频307同帧：`{videos}`')
    rows.append(f'- 固定看图页：`{review}`')
    rows.append(f'- 框数差异页：`{differences}`')
    rows.append('- 结果JSON：`docs/fire_v22_results_20261028.json`')
    rows.append('')

    write(json_out, result)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text('\n'.join(rows), encoding='utf-8')
    print(f'Published {json_out} and {markdown_out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--videos', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--differences', type=Path, required=True)
    parser.add_argument('--json-out', type=Path, required=True)
    parser.add_argument('--markdown-out', type=Path, required=True)
    args = parser.parse_args()
    publish(
        args.training,
        args.comparison,
        args.videos,
        args.review,
        args.differences,
        args.json_out.resolve(),
        args.markdown_out.resolve(),
    )
