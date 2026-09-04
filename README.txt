RAV
===

Dataset
-------

通过网盘分享的文件：RAV-data
链接：https://pan.baidu.com/s/1YrFSpqPy6iCD_ILKzpQKFQ?pwd=nprm
提取码：nprm

Directory pipeline
------------------

The directory pipeline discovers every ``.mat`` file, searches the full
lambda/beta grid on the search GPUs, merges the best parameters, and then runs
the selected parameters once per seed on the repeat GPUs:

    python run_directory_pipeline.py \
        --data_dir /path/to/mat/files \
        --output_dir results_experiment

Defaults:

- Search GPUs: ``0,1,2,3,4,5``
- Repeat GPUs: ``0,1,2,3,4``
- Repeat seeds: ``42,3407,4079,2024,0``
- Lambda grid: ``1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000``
- Beta grid: ``1e-5,1e-4,1e-3,1e-2,1e-1,1``

The pipeline is resumable. Completed parameter groups are reused when the same
command and output directory are used again.

Outputs
-------

- ``best_params.csv``: selected parameters for every dataset.
- ``repeats/repeat_metrics.csv``: metrics and artifact paths for every seed.
- ``repeats/summary_mean_std.csv``: per-dataset mean and standard deviation.
- ``repeats/labels/<dataset>/seed_<seed>.csv``: sample labels for every run.

Each labels CSV contains ``index``, ``label``, ``pred``, and ``aligned_pred``.
``aligned_pred`` is the predicted cluster ID mapped into the ground-truth label
space with the Hungarian algorithm.

Tests
-----

    python -m unittest discover -s tests -v
