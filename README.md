# SMRL — Structured Multidimensional Representation Learning

**A research codebase extending the SMRL framework for efficient, transform-domain sequence modeling.**

This repository constitutes a **continuation of the research** introduced in El Ichi *et al.* (2026), *Structured Multidimensional Representation Learning for Large Language Models* ([arXiv:2603.05727](https://arxiv.org/abs/2603.05727)). Building upon the theoretical formulation of tensorized token representations, L-transforms, and slice-wise multi-head attention, we investigate algorithmic and empirical extensions aimed at closing the gap between asymptotic complexity reductions and practical wall-clock performance, while broadening the evaluative scope of the original framework.

---

## Relation to Prior Work

The reference article establishes a multidimensional encoding of transformer activations via mode-\(3\) orthogonal transforms (e.g., DCT) and frontal-slice operators, yielding a structured alternative to dense self-attention. The present codebase takes that formulation as its **methodological foundation** and focuses on subsequent research questions that remain open after the initial theoretical and prototype demonstration:

| Direction | Scientific objective |
|-----------|----------------------|
| **Efficient attention** | Integrate linear-complexity attention surrogates (e.g., Linformer) with the tensor-product encoder to further alleviate the \(O(n^2)\) bottleneck while retaining slice-level structure. |
| **Batched slice execution** | Realize fully batched, GPU-parallel slice kernels so that reported FLOP reductions translate into measurable latency and throughput gains. |
| **Extended evaluation** | Scale experimental validation to broader benchmarks and task regimes, including fine-tuning of pretrained models. |
| **Learned spectral bias** | Study alternative orthogonal bases and *trainable* transform operators to adapt the inductive spectral prior across domains. |

---

## Motivation

Self-attention remains a principal computational bottleneck in large-scale sequence models. Multidimensional (tensor / spectral) encodings provide an inductive pathway to structure activations and localize attention within lower-dimensional slices. The outstanding scientific challenge is twofold: (i) to **operationalize** these constructions under efficient attention approximations and batched numerical realizations; and (ii) to **empirically substantiate** their utility under standardized pretraining and fine-tuning protocols.

---

## Research Agenda

1. **Hybridization of tensor-product encoding and efficient attention.**  
   Combine the SMRL encoder with low-rank / projected attention mechanisms to mitigate quadratic cost without discarding the frontal-slice factorization.

2. **Fully batched slice-parallel computation.**  
   Treat the slice index as a concurrent batch axis, enabling high-throughput execution that materializes theoretical FLOP savings as wall-clock speedups.

3. **Broader empirical assessment and transfer.**  
   Expand evaluation beyond limited prototypes to comprehensive benchmarks and downstream fine-tuning of pretrained checkpoints.

4. **Alternative orthogonal and learned operators.**  
   Explore transforms beyond fixed DCT—including domain-adapted orthogonal maps and end-to-end learned spectral operators—to modulate the transform-domain bias.

---

## Repository Layout

```
SMRL_LLM/
├── experiments/SMRL/     # Training entry points, configs, model heads
│   ├── models/           # Language modeling & sequence classification
│   └── configs/          # Experiment defaults
├── shared/
│   ├── modules/          # Attention, embeddings, FFN, norms, blocks
│   ├── tools/            # DCT, tensorization, losses, optimizers
│   └── baseline/         # Reference Transformer / GPT-style modules
```

Core implementations reside under `shared/modules/attentions/` (slice-aware multi-head attention, tensor transforms) and `shared/tools/functions/` (DCT and embedding tensorization).

---

## Status

This repository is an **active research artifact**. Interfaces and experimental configurations may evolve as attention backends, batched kernels, and evaluation suites mature.

---

## Citation

If you use this codebase, please cite the foundational work:

```bibtex
@misc{ichi2026structuredmultidimensionalrepresentationlearning,
      title={Structured Multidimensional Representation Learning for Large Language Models},
      author={Alaa El Ichi and Khalide Jbilou and Mohamed El Guide and Franck Dufrenois},
      year={2026},
      eprint={2603.05727},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.05727},
}
```

---

## License

License to be added.
