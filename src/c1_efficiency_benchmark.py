
import os
import time
from pathlib import Path

import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import (
    MultimodalDermDataset,
    test_transforms,
)
from src.models.models import (
    MultimodalDermModel,
    C1CrossAttentionModel,
)


WARMUP_ITERS = 30
BENCHMARK_ITERS = 100
BENCHMARK_SEED = 42


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )


def benchmark_model(
    model,
    clinic,
    derm,
    metadata,
    device,
):
    model.eval()

    # -----------------------------
    # Warm-up
    # -----------------------------
    with torch.inference_mode():
        for _ in range(WARMUP_ITERS):
            model(
                clinic,
                derm,
                meta_features=metadata,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # -----------------------------
    # Timed inference
    # -----------------------------
    start = time.perf_counter()

    with torch.inference_mode():
        for _ in range(BENCHMARK_ITERS):
            model(
                clinic,
                derm,
                meta_features=metadata,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    latency_ms = (
        elapsed
        / BENCHMARK_ITERS
        * 1000.0
    )

    throughput = (
        1000.0 / latency_ms
    )

    if device.type == "cuda":
        peak_memory_mb = (
            torch.cuda.max_memory_allocated()
            / 1024**2
        )
    else:
        peak_memory_mb = float("nan")

    return {
        "latency_ms_per_sample":
            latency_ms,
        "throughput_samples_per_sec":
            throughput,
        "peak_gpu_memory_mb":
            peak_memory_mb,
    }


def main():

    paths = Config.ensure_runtime_dirs()

    if os.path.isdir(
        "/content/local_images"
    ):
        paths["img_dir"] = (
            "/content/local_images"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # -----------------------------
    # Metadata dimension
    # -----------------------------
    encoder = joblib.load(
        paths["meta_encoder"]
    )

    meta_input_dim = len(
        encoder.get_feature_names_out()
    )

    # -----------------------------
    # One fixed Test sample
    # -----------------------------
    dataset = MultimodalDermDataset(
        paths["test_csv"],
        paths["img_dir"],
        paths["label_mapping"],
        meta_encoder_path=
            paths["meta_encoder"],
        transform=test_transforms,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    clinic = batch[
        "clinic_img"
    ].to(device)

    derm = batch[
        "derm_img"
    ].to(device)

    metadata = batch[
        "metadata"
    ].to(device)

    experiments = [
        {
            "model":
                "B5_Dual_Metadata",
            "fusion":
                "Concatenation",
        },
        {
            "model":
                "C1_CrossAttention",
            "fusion":
                "Cross-Attention",
        },
    ]

    rows = []

    for exp in experiments:

        print(
            "\nBenchmarking:",
            exp["model"]
        )

        if exp["model"] == \
                "B5_Dual_Metadata":

            model = MultimodalDermModel(
                num_classes=
                    Config.NUM_CLASSES,
                num_concepts=
                    Config.NUM_CONCEPTS,
                modality="dual",
                bottleneck_type="none",
                use_metadata=True,
                meta_input_dim=
                    meta_input_dim,
            )

        else:

            model = C1CrossAttentionModel(
                num_classes=
                    Config.NUM_CLASSES,
                meta_input_dim=
                    meta_input_dim,
                d_model=256,
                num_heads=4,
            )

        checkpoint = os.path.join(
            paths["output_dir"],
            f"{exp['model']}_seed_"
            f"{BENCHMARK_SEED}.pth"
        )

        if not os.path.exists(
            checkpoint
        ):
            raise FileNotFoundError(
                checkpoint
            )

        state = torch.load(
            checkpoint,
            map_location=device
        )

        model.load_state_dict(
            state,
            strict=True
        )

        model = model.to(device)

        if device.type == "cuda":
            torch.cuda.empty_cache()

        result = benchmark_model(
            model,
            clinic,
            derm,
            metadata,
            device,
        )

        row = {
            "Model":
                exp["model"],
            "Fusion":
                exp["fusion"],
            "Parameters":
                count_parameters(model),
            "Latency_ms_per_sample":
                result[
                    "latency_ms_per_sample"
                ],
            "Throughput_samples_per_sec":
                result[
                    "throughput_samples_per_sec"
                ],
            "Peak_GPU_Memory_MB":
                result[
                    "peak_gpu_memory_mb"
                ],
            "Batch_Size":
                1,
            "Warmup_Iterations":
                WARMUP_ITERS,
            "Benchmark_Iterations":
                BENCHMARK_ITERS,
            "Checkpoint_Seed":
                BENCHMARK_SEED,
        }

        rows.append(row)

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)

    result_dir = (
        Path(paths["results_dir"])
        / "contribution_1"
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output = (
        result_dir
        / "c1_efficiency.csv"
    )

    df.to_csv(
        output,
        index=False
    )

    print(
        "\n=== C1 EFFICIENCY ==="
    )

    print(
        df.to_markdown(
            index=False,
            floatfmt=".4f"
        )
    )

    print(
        "\n✅ Saved:",
        output
    )


if __name__ == "__main__":
    main()
