"""
SimpleTuner Qwen-Image LoRA training A/B benchmark: MFA vs baseline.

Usage:
    # From the SimpleTuner venv (Python 3.13):
    python tests/run_simpletuner_training.py             # MFA enabled (benchmark)
    python tests/run_simpletuner_training.py --baseline  # MFA disabled (baseline)

Prerequisites:
    - SimpleTuner installed: pip install -e ~/Documents/code/SimpleTuner
    - metal-flash-sdpa installed: pip install -e ~/Documents/code/metal-flash-sdpa
    - ~60GB disk space for model download (cached in ~/.cache/huggingface)

This script:
    1. Optionally enables metal-flash-sdpa monkey-patch (default) or runs baseline
    2. Runs SimpleTuner's Trainer with Qwen-Image LoRA config at full 1024px resolution
    3. Reports timing and MFA usage statistics for comparison
"""
import os
import sys
import time

# Ensure venv bin is on PATH (needed for ninja JIT compilation by optimum-quanto)
venv_bin = os.path.dirname(sys.executable)
os.environ["PATH"] = venv_bin + ":" + os.environ.get("PATH", "")

# Import metal-flash-sdpa BEFORE any other imports that might use SDPA
import metal_flash_sdpa

# We'll decide whether to enable after parsing args, but import must happen early
# to ensure monkey-patch is available before SimpleTuner loads

# Now import SimpleTuner
os.environ["ACCELERATE_LOG_LEVEL"] = "WARNING"

import torch
print(f"[info] PyTorch {torch.__version__}, MPS available: {torch.backends.mps.is_available()}")

# Config for Qwen-Image LoRA on M3 Pro 36GB
# int2-quanto quantization brings 20B transformer to ~5GB
CONFIG = {
    "base_model_precision": "int2-quanto",
    "caption_dropout_probability": 0.0,
    "checkpoint_step_interval": 0,  # no checkpoints for test
    "checkpoints_total_limit": 0,
    "compress_disk_cache": False,
    "data_backend_config": os.path.expanduser(
        "~/Documents/code/SimpleTuner/simpletuner/examples/multidatabackend-small-dreambooth-1024px.json"
    ),
    "disable_benchmark": True,
    "disable_bucket_pruning": True,
    "flow_schedule_shift": 1.73,
    "gradient_checkpointing": True,
    "hub_model_id": "metal-flash-sdpa-test",
    "ignore_final_epochs": True,
    "learning_rate": 1e-4,
    "lora_alpha": 8,
    "lora_rank": 8,
    "lora_type": "standard",
    "lr_scheduler": "constant_with_warmup",
    "lr_warmup_steps": 5,
    "max_grad_norm": 0.01,
    "max_train_steps": 20,  # Full benchmark run
    "minimum_image_size": 0,
    "mixed_precision": "bf16",
    "model_family": "qwen_image",
    "model_flavour": "v1.0",
    "model_type": "lora",
    "num_eval_images": 1,  # generate 1 validation image
    "num_train_epochs": 0,
    "optimizer": "optimi-lion",
    "output_dir": None,  # Will be set based on run type in main()
    "push_checkpoints_to_hub": False,
    "push_to_hub": False,
    "quantize_via": "cpu",
    "quantize_activations": False,
    "report_to": "none",
    "resolution": 1024,  # Full training resolution
    "resolution_type": "pixel_area",
    "resume_from_checkpoint": "latest",
    "seed": 42,
    "skip_file_discovery": False,
    "tracker_project_name": "metal-flash-sdpa-test",
    "tracker_run_name": "integration-test",
    "train_batch_size": 1,
    "use_ema": False,
    "vae_batch_size": 1,
    "validation_steps": 99999,  # only validate at end
    # Skip config fallback to avoid loading stale configs
    "__skip_config_fallback__": True,
}


def main():
    from simpletuner.helpers.training.trainer import Trainer
    from simpletuner.helpers.training.attention_backend import AttentionBackendController, AttentionPhase

    # Parse command line args
    baseline_mode = "--baseline" in sys.argv
    run_type = "baseline" if baseline_mode else "mfa"

    # Set output directory based on run type
    CONFIG["output_dir"] = f"/tmp/metal-flash-sdpa-benchmark/{run_type}"

    # Enable or disable MFA
    if baseline_mode:
        print("[metal-flash-sdpa] DISABLED for baseline run")
    else:
        metal_flash_sdpa.enable()
        print("[metal-flash-sdpa] ENABLED for benchmark run")
        print(f"[info] metal-flash-sdpa MIN_SEQ_LEN={metal_flash_sdpa.MIN_SEQ_LEN}")

    print("\n[1/4] Initializing Trainer...")
    start = time.time()
    trainer = Trainer(config=CONFIG)
    print(f"  Trainer init: {time.time() - start:.1f}s")

    print("\n[2/4] Loading model components...")
    start = time.time()
    # Follow exact init sequence from simpletuner/train.py
    trainer.configure_webhook()
    trainer.init_noise_schedule()
    trainer.init_seed()
    trainer.init_huggingface_hub()
    trainer.init_preprocessing_models()
    trainer.init_precision(preprocessing_models_only=True)
    trainer.init_data_backend()
    trainer.init_unload_text_encoder()
    trainer.init_unload_vae()
    trainer.init_load_base_model()
    trainer.init_delete_model_caches()
    trainer.init_controlnet_model()
    trainer.init_tread_model()
    trainer.init_precision()
    trainer.init_freeze_models()
    trainer.init_trainable_peft_adapter()
    trainer.init_ema_model()
    trainer.init_precision(ema_only=True)
    trainer.move_models(destination="accelerator")
    trainer.init_distillation()
    trainer.init_validations()
    print(f"  Model loading: {time.time() - start:.1f}s")

    # Apply attention backend (this uses diffusers' native backend by default)
    AttentionBackendController.apply(trainer.config, AttentionPhase.EVAL)
    trainer.init_benchmark_base_model()
    AttentionBackendController.apply(trainer.config, AttentionPhase.TRAIN)

    print(f"\n[3/4] Starting training (max_train_steps={CONFIG['max_train_steps']})...")
    metal_flash_sdpa.reset_dispatch_count()
    start = time.time()

    trainer.resume_and_prepare()
    trainer.init_trackers()
    trainer.train()

    elapsed = time.time() - start
    steps = CONFIG['max_train_steps']
    avg_time_per_step = elapsed / steps

    print(f"\n[4/4] Results:")
    print(f"  Run type: {run_type.upper()}")
    print(f"  Training time: {elapsed:.1f}s ({avg_time_per_step:.2f}s/step)")

    if not baseline_mode:
        dispatch_count = metal_flash_sdpa.get_dispatch_count()
        fallback_count = metal_flash_sdpa._fallback_count
        print(f"  MFA dispatch count: {dispatch_count}")
        print(f"  MFA fallback count: {fallback_count}")
        print(f"  MFA dispatches per step: {dispatch_count / steps:.1f}")

        if dispatch_count > 0:
            print("\n  SUCCESS: metal-flash-sdpa is being used for attention!")
        else:
            print("\n  WARNING: MFA was not dispatched. Check if sequence lengths are below MIN_SEQ_LEN.")

    print(f"\n  Output saved to: {CONFIG['output_dir']}")
    print(f"\n{'='*60}")
    print(f"SUMMARY - {run_type.upper()} RUN")
    print(f"{'='*60}")
    print(f"  Total time:     {elapsed:.1f}s")
    print(f"  Avg per step:   {avg_time_per_step:.2f}s")
    print(f"  Resolution:     {CONFIG['resolution']}px")
    print(f"  Steps:          {steps}")
    print(f"{'='*60}")

    # Cleanup
    trainer.cleanup()


if __name__ == "__main__":
    main()
