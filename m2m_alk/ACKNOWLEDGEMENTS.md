# Acknowledgements

M2M-ALK relies on patient-level CT and WSI representations. The code in this repository is released as a research implementation and does not redistribute third-party pathology foundation model weights or CLAM code. Users should download third-party resources from their official pages and comply with each project's license, access conditions, and citation requirements.

We gratefully acknowledge the following open resources:

| Resource | Official source | Notes |
| --- | --- | --- |
| CLAM | https://github.com/mahmoodlab/CLAM | Clustering-constrained Attention Multiple Instance Learning for weakly supervised WSI analysis and attention-based heatmap generation. |
| CTransPath | https://github.com/Xiyue-Wang/TransPath | Transformer-based histopathology feature extractor used as one of the pathology foundation model baselines. |
| CONCH v1.5 | https://huggingface.co/MahmoodLab/conchv1_5 | Vision-language pathology foundation model used as one of the WSI encoder baselines. |
| UNI2 | https://huggingface.co/MahmoodLab/UNI2-h | Pathology foundation model used as one of the WSI encoder baselines. |
| Prov-GigaPath | https://github.com/prov-gigapath/prov-gigapath | Whole-slide pathology foundation model used as one of the WSI encoder baselines. |
| Virchow2 | https://huggingface.co/paige-ai/Virchow2 | Pathology foundation model selected as the fixed WSI feature extractor in the final M2M-ALK pipeline. |

If you use any of these resources, please cite the corresponding original papers and model cards/repositories in addition to this repository.
