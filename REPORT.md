# Multimodal Organ Diffusion Report for NSCLC

## Intro

This project was originally conceived as my capstone project, but I ended up building it as a final project for this course instead. In this project, I built a multimodal prediction pipeline for non-small cell lung cancer (NSCLC) with three explicit goals: predicting overall survival (OS), predicting recurrence and its location, and producing structured organ-level diffusion paths as an explanation of how the model distributes latent metastatic spread tendency. Traditional clinical models often focus on one modality only, such as imaging or tabular clinical variables, and they usually provide limited insight into how risk may be distributed across organs. I wanted to design a system that could integrate CT imaging, clinical information, RNA expression, and immune-related signals into a single framework while still producing outputs that are clinically meaningful across all three goals.

For OS, I used a Cox-based survival head trained on time-to-event labels. For recurrence, I trained a classification head that jointly predicts whether recurrence occurs and, when location labels are available, distinguishes local, regional, and distant patterns. For diffusion paths, I introduced an organ diffusion explanation layer built on top of the graph reasoning module: it produces per-organ susceptibility scores, edge-level diffusion probabilities, and ranked top-k organ-to-organ paths extracted by beam search. This distinction is important: the diffusion outputs are not direct organ-level ground truth labels, because the public cohort does not provide reliable metastatic path supervision. Instead, they act as model-guided explanations constrained by the primary survival and recurrence tasks.

## Method

I implemented the pipeline as a staged multimodal architecture. The imaging branch starts from CT preprocessing, tumor mask extraction, and organ segmentation. This branch provides tumor-level and organ-level imaging features. The RNA branch aligns expression profiles to patient IDs, learns dense molecular embeddings, and derives immune tokens from signature-based features. The clinical branch converts the tabular records into structured EHR features and then encodes them into a fixed-dimensional representation.

After extracting modality-specific features, I aligned them into a fixed set of organ nodes, including the primary tumor, lung, bone, liver, lymph node or mediastinum, and brain. I then fused these signals with cross-attention so that each organ query could attend to the available evidence tokens. The fused organ tokens Z were then refined by a graph reasoning module and passed through a joint prediction and explanation architecture.

**Graph reasoning.** The graph module operated on Z using a graph transformer, where learned adjacency logits were injected as biases into the attention scores and non-candidate edges were hard-masked. This allowed the model to propagate information across organ nodes in a structured way constrained by biological priors. The output Z′ was a refined organ-level token matrix encoding both within-node features and cross-organ interactions.

**Joint prediction and explanation.** Rather than treating the explanation layer as a post-hoc step, I designed it to be in-loop: the explanation outputs were computed from Z′ and then fed back into the prediction heads as additional context. Concretely, Z′ was first pooled into a global patient representation and passed through a base trunk MLP. In parallel, the LatentDiffusionExplainer produced organ susceptibility scores (sigmoid MLP per node) and edge diffusion probabilities (sigmoid MLP on paired node features plus edge metadata). These were then used to construct two context vectors: a susceptibility-weighted sum of Z′ across non-primary organs, and an edge-weighted sum of Z′ using the primary node's outgoing diffusion probabilities. Four scalar summary statistics were also derived from the diffusion outputs. All of these were concatenated with the base trunk and passed through a fusion layer to produce a joint trunk, which then fed the final OS and recurrence prediction heads.

The reason for this design is that without any feedback, the explanation heads would have no incentive to produce clinically meaningful patterns. By routing the diffusion outputs back into the primary prediction pathway and including explanation-based auxiliary losses in the joint training objective, the model is encouraged to allocate organ susceptibility and edge probabilities in ways that are consistent with the observed survival and recurrence outcomes. The diffusion outputs therefore reflect task-constrained latent structure rather than arbitrary activations.

## Workflow

1. I constructed a patient manifest and time-zero labels for 211 patients.
2. I preprocessed CT data, matched tumor segmentation, and trained or applied the organ segmentation model.
3. I generated RNA, immune, and EHR embeddings for the patients with available data.
4. I assembled all modality outputs into organ-level tokens, fused them, and performed graph-based reasoning.
5. I trained the model in phases, beginning with a larger non-RNA baseline and then fine-tuning on the smaller RNA subset.
6. I exported both quantitative outputs and visual diffusion reports, and I also verified that the pipeline can run an external-case inference smoke test.

## Results

### Quantitative Performance

The full cohort contains 211 patients. CT was available for all 211 cases, PET for 201, tumor segmentation for 144, AIM semantic annotations for 190, and RNA for 130. This distribution already explains one of the core challenges of the project: the richest multimodal setting is also the smallest one.

| Experiment | Cohort | C-index | Recurrence AUC | Location Accuracy |
| --- | --- | ---: | ---: | ---: |
| Stage 12 cross-validation baseline | 211 | 0.531 | 0.567 | 0.377 |
| Phase 3 baseline without RNA | 211 | 0.642 | 0.675 | 0.462 |
| Initial Phase 4 RNA fine-tuning | 130 | 0.534 | 0.526 | 0.444 |
| Best tuned Phase 4 run | 130 | 0.511 | 0.696 | 0.333 |

These numbers show a clear pattern. The strongest survival-oriented validation result came from the Phase 3 baseline trained on the larger 211-patient cohort without RNA. However, after tuning, the Phase 4 RNA-based model achieved the best recurrence discrimination, reaching a recurrence AUC of 0.696. This suggests that RNA information can improve recurrence modeling, but only after careful tuning, and that the small size of the RNA subset still limits stability for survival and location prediction.

The explanation outputs were also structured and consistent. In the best tuned RNA-subset model, the dominant top path was `Primary -> Bone`, appearing in 97.7% of cases. The highest mean organ susceptibility scores were Lung (0.751), Brain (0.728), and Primary (0.714). I interpret these as model-internal risk allocation patterns rather than biological truth claims.

### Visual Results

The cohort-level visualization below summarizes the dominant diffusion structure learned by the best tuned model.

![Figure 1. Cohort-level primary diffusion summary.](output/stage13/13.4_visualize_diffusion/cohort_primary_diffusion.svg)

*Figure 1. Cohort-level diffusion summary for the best tuned model. It highlights the dominant organ-to-organ explanation pattern across the RNA subset, with bone emerging as the main destination in the latent diffusion layer.*

To make the results more concrete, I also include representative patient-level figures from the generated visualization set.

![Figure 2. Patient R01-151 diffusion map.](output/stage13/13.4_visualize_diffusion/patients/R01-151_primary_diffusion.svg)

*Figure 2. Patient `R01-151` is a high-risk distant-recurrence example with strong bone and liver diffusion tendencies, making it a useful illustration of the model's high-confidence explanation behavior.*

![Figure 3. Patient R01-106 diffusion map.](output/stage13/13.4_visualize_diffusion/patients/R01-106_primary_diffusion.svg)

*Figure 3. Patient `R01-106` was predicted as a regional case, showing that the model can produce a different recurrence-location prediction while still maintaining a structured organ diffusion pattern.*

![Figure 4. Patient R01-049 diffusion map.](output/stage13/13.4_visualize_diffusion/patients/R01-049_primary_diffusion.svg)

*Figure 4. Patient `R01-049` was predicted as a local case. I include it because it shows that the latent path ranking and the supervised recurrence-location label are related but not identical.*

![Figure 5. Patient R01-026 diffusion map.](output/stage13/13.4_visualize_diffusion/patients/R01-026_primary_diffusion.svg)

*Figure 5. Patient `R01-026` is a strong bone-dominant example with very high top-path probabilities, making it a representative high-separation case for qualitative inspection.*

Beyond the training results, I also confirmed that the system can generate an external-case inference report in HTML. This means the project is not only a modeling experiment, but also a deployable reporting prototype.

## Conclusion

Overall, this project demonstrates that a multimodal NSCLC pipeline can jointly support prediction and structured explanation. I successfully integrated imaging, clinical, RNA, and immune features into an organ-level architecture with graph reasoning, and I produced both quantitative outputs and visual reports. The results indicate that the larger non-RNA baseline remains more stable for survival modeling, while the tuned RNA-based model offers the strongest recurrence discrimination. This means multimodal learning is promising, but it requires careful tuning and is still constrained by sample size.

The most important conclusion is that the framework is feasible and extensible. It already supports end-to-end processing, cohort-level visualization, patient-level diffusion figures, and external-case reporting. At the same time, the explanation layer should be interpreted carefully, because it provides latent task-guided structure rather than organ-level truth. In future work, stronger validation on larger multimodal cohorts would be necessary to confirm whether the learned diffusion patterns reflect clinically reliable biology.
