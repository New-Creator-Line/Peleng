# YOLO26 Segmentation Comparison Report

**Generated:** Auto-benchmark on `data/datasets/My First Project.v1i.yolo26/data.yaml`

| Metric | 40ep_Original | Plus15_FineTuned | Delta (B vs A) |
|:---|:---:|:---:|:---:|
| **Weights Path** | `best.pt` | `best.pt` | - |
| **Mask mAP@50** | **99.50%** | **99.19%** | **-0.31%** |
| **Mask mAP@50-95** | 84.66% | 87.12% | +2.46% |
| **Mask Precision** | 99.59% | 95.09% | -4.50% |
| **Mask Recall** | 100.00% | 100.00% | +0.00% |
| **Box mAP@50** | 99.50% | 99.19% | -0.31% |
| **Box mAP@50-95** | 89.48% | 85.17% | -4.31% |
| **Inference Latency** | 55.8 ms | 62.2 ms | - |

## Training Dynamics

| Parameter | 40ep_Original | Plus15_FineTuned |
|:---|:---:|:---:|
| **Epochs Trained** | 40 | N/A |
| **Best Epoch** | 28 | N/A |
| **Best Mask mAP50 (Train log)** | 94.14% | N/A |

## Visual Side-by-Side Images
The side-by-side comparison images have been saved to `results/comparisons`.
