# TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs

Jun Zhang<sup>1,2,\*</sup> Teng Wang<sup>2,!</sup> Yuying Ge<sup>2</sup> Yixiao Ge<sup>2</sup> Xinhao Li<sup>1</sup> Limin Wang<sup>1,3,!</sup> <sup>1</sup>Nanjing University <sup>2</sup>ARC Lab, Tencent PCG <sup>3</sup>Shanghai AI Lab

https://timelens-arc-lab.github.io/

## Abstract

This paper does not introduce a novel method but instead establishes a straightforward, incremental, yet essential baseline for video temporal grounding (VTG), a core capability in video understanding. While multimodal large language models (MLLMs) excel at various video understanding tasks, the recipes for optimizing them for VTG remain under-explored. In this paper, we present TimeLens, a systematic investigation into building MLLMs with strong VTG ability, along two primary dimensions: data quality and algorithmic design. Wefirst expose critical quality issues in existing VTG benchmarks and introduce TimeLens-Bench, comprising meticulously re-annotated versions of three popular benchmarks with strict quality criteria. Our analysis reveals dramatic model re-rankings compared to legacy benchmarks, confirming the unreliability ofprior evaluation standards. We also address noisy training data through an automated re-annotation pipeline, yielding TimeLens-100K, a large-scale, high-quality training dataset. Building on our data foundation, we conduct in-depth explorations of algorithmic design principles, yielding a series of meaningful insights and effective yet efficient practices. These include interleaved textual encoding for time representation, a thinking-free reinforcement learning with verifiable rewards (RLVR) approach as the training paradigm, and carefully designed recipesfor RLVR training. These efforts culminate in TimeLens models, a family of MLLMs with state-of-the-art VTG performance among open-source models and even surpass proprietary models such as GPT-5 and Gemini-2.5-Flash. All codes, data, and models will be released tofacilitatefuture research.

## 1. Introduction

Recent multimodal large language models (MLLMs) have excelled at understanding “what” happens in a video, yet they largely fail when asked “when.” This limitation is central to the task of video temporal grounding (VTG). The challenge is twofold: 1) VTG necessitates a fundamental shift from coarse semantic aggregation to fine-grained time-aware perception; 2) Distinguishing queried events requires modeling long-term visual dynamics over appearance-centric features, which are notoriously difficult to annotate and learn. As MLLMs become integral to perception [43, 44, 55, 58] and reasoning systems [6, 13, 37, 39, 40, 66], equipping them with robust temporal awareness is no longer optional, but essential [26, 35, 46, 49, 54].

![](assets/images/792dd038922b2df54b151737d12d0bf5b7b9f559a5c887a3ee68bfafb279a411.jpg)  
Figure 1. Overview of the proposed TimeLens framework. We systematically explore the key factors for building performant video temporal grounding models, dissecting our efforts along two primary dimensions: data quality and algorithmic design. For data quality, we focus on benchmark diagnosis, benchmark refinement, and creating a reliable evaluation suite. For algorithmic design, we study various aspects including time encoding, training recipes, and optimization strategies to establish best practices and develop the TimeLens models.

This work focuses on post-training MLLMs with leading temporal grounding ability. This investigation is a straightforward extension given the recent progress in pretrained foundation MLLMs [2, 3, 53]. Different from heavily studied general understanding tasks, recipes for fine-grained grounding tasks are not yet to be established. This paper aims to systematically investigate core components of building timeaware MLLMs (Fig. 1) along two primary dimensions: data quality and algorithmic design.

Our investigation starts by exposing critical flaws in evaluation benchmarks. We find that existing VTG benchmarks [11, 25, 27] not only lack a clear comparison between leading proprietary and open-source models but are also rife with low-quality queries and erroneous timestamps. This noisy data may render current leaderboards misleading and misguide research efforts. To rectify this, we undertook a meticulous data overhaul. We first defined strict criteria for query and timestamp quality, in terms of uniqueness, existence, clarity, and accuracy. We then manually re-annotated three popular datasets (Charades-STA [11], ActivityNet Captions [25], QVHighlights [27]) to create TimeLens-Bench, a rigorously cross-validated benchmark. As shown in Fig. 2a, the necessity of this correction is confirmed by a dramatic re-ranking of models on TimeLens-Bench compared to their performance on legacy benchmarks, proving the unreliability of prior evaluation standards. Beyond evaluation, we also fix the noisy training data by automated re-annotation, yielding TimeLens-100K, a large-scale, high-quality training dataset.

![](assets/images/0483840d2400328a612b367d4df29e40b25c17517d3f6b1d0b5add675d76ccf7.jpg)  
(a)

![](assets/images/388381c5e43a2573c83108bddecf6ffde4452134903e4aa339fdf0cc35db5ecb.jpg)  
(b)  
Figure 2. (a) Impact of data quality on model evaluation. A comparison of Mean IoU on original versus our refined Charades-STA benchmarks. The deviation from the diagonal line shows that legacy benchmarks are misleading, as they inflate the results of some open-source models while underestimating proprietary ones. (b) Cumulative performance gains of TimeLens explorations. This analysis shows how each component boosts the model’s average performance on TimeLens-Bench. From data curation to thinking-free RLVR with early stopping and difficulty-based data sampling, each step demonstrates a clear positive impact towards our final TimeLens model. TimeLens-7B and TimeLens-8B are based on Qwen2.5-VL-7B and Qwen3-VL-8B, respectively.

With our curated data suite as a solid foundation, we conduct in-depth explorations on the algorithmic design principles from three key aspects. First, for timestamp representation, we discover that a simple yet effective interleaved textual encoding strategy outperforms more complex alternatives. Second, we determine that VTG is fundamentally a perception-driven task, and thus employ a pure thinkingfree reinforcement learning with verifiable rewards (RLVR) approach that outperforms other training paradigms in both efficiency and performance. Finally, our detailed analysis of RLVR training reveals two key recipes for both performance and training efficiency: (1) early stopping when reward metrics plateau, and (2) difficulty-based data sampling. By integrating these insights and design principles, we ultimately develop TimeLens models, a family of MLLMs with superior VTG capability. As shown in Fig. 2b, our model achieves state-of-the-art performance among opensource models and even surpasses proprietary models such as GPT-5 and Gemini-2.5-Flash.

Through these efforts, we identified and addressed longoverlooked quality issues in existing datasets, and derived a series of insights and best practices in algorithmic design. We hope TimeLens can serve as a solid foundation in both data curation and algorithmic design principles, to facilitate future research on building MLLMs with strong VTG capabilities. Our code, data, and models will be open-sourced.

## 2. Related Work

Temporal Grounding Datasets. Numerous VTG datasets have been proposed, spanning diverse domains [14, 22, 25, 27, 41, 45, 50]. Early works [11, 38, 65] trained and evaluated models on the training and test splits of a single benchmark [25, 45] to assess their ability to fit single-domain data distribution. In recent works [17, 37, 46], large diverse corpuses composed of multiple different source datasets [1, 22, 36, 41, 50, 60] are aggregated for training, and a suite of distinct benchmarks [11, 25, 27] are used to probe the models’ real-world cross-domain generalizability.

However, the critical issue of data quality has been overlooked. There lacks a systematic examination on whether existing datasets are reliable enough for training and evaluation. In this paper, we manually inspect existing datasets, identify and correct errors, and produce quality-improved training and evaluation suites for developing more practical

![](assets/images/9c2ba092c7826bb08cb4cb7057bcd8e641aadfb78c56f21388b9a55fc5986dfd.jpg)  
✘ Original Query: A man is running down a track by a field. Re-annotated: A man kneels on the ground with both knees.

![](assets/images/0e6f52c119f13b92842560dd4c9b43031cd585c82a8ba7bb70877e9128dd0848.jpg)

![](assets/images/35c9fffc7ea5428a72678c25c90067a2b010f165da8da3d2f8da6822b60af216.jpg)  
✘ Original Query: Man in gray top walks from outside to inside. Re-annotated: Three people are crossing the crosswalk.  
Query 1: A person sits in a chair. ✘ Query 2 (Duplicate): The person sits in the chair momentarily. Re-annotated: A man moves the position of a chair.

![](assets/images/8512e4844e71e239bd9bc7e1bec9fcb1d5192a06dd9aef2313fbab7f9673093c.jpg)  
Figure 3. Qualitative examples of errors and fixes. We present representative errors identified in existing datasets, spanning different error types, including multiple event occurrences, no event occurrence, duplicate queries for the same video, unclear query, and inaccurate annotation. Through our rigorous manual refinement, these errors have been properly corrected, significantly improving data quality.

![](assets/images/833f9636592a4d17fb7591744676e434917e1f40406e910bf32a464781c4f3ef.jpg)  
✘ Original Query: When all of the lather is gone, the process continues until the boot is finally dry. Re-annotated: A man is tying shoelaces on a shoe.

VTG models.

MLLMs for Temporal Grounding. Substantial works focus on algorithmic designs to improve MLLMs’ VTG capability. One line of research explores model architectures, including token compression methods to reduce computation on long videos [46, 61], timestamp encoding strategies to align the timestamps of each frame with its corresponding features [5, 12, 28, 34, 51, 55, 62]. Another line of works investigate training strategies: introducing various supervised fine-tuning tasks to improve VTG performance [7, 61], or designing verifiable rewards to improve performance via reinforcement learning [4, 33, 54, 59].

Despite the abundance of proposed designs, their inconsistent experimental settings make it difficult to fairly compare their relative merits and establish best practices. In this paper, we systematically analyze these design choices using our quality-assured training and evaluation suites, offering key insights for improving MLLMs’ VTG capability.

## 3. Towards Reliable, High-Quality VTG Data 3.1. Annotation Criteria

Task Formulation. For temporal grounding, a model takes as input a video v and a text query q, localizes the event E described by q, and outputs the corresponding temporal segment $S = ( t _ { \mathrm { s t a r t } } , t _ { \mathrm { e n d } } )$ . In practice, a video is typically annotated with one or more query-segment pairs $\{ ( q _ { i } , S _ { i } ) \} _ { i = 1 } ^ { n }$

Input Criteria. The input video and query should satisfy:

• Query clarity and specificity. The query must be clear, precise, and unambiguous for accurate and definitive grounding (A counterexample like “the game continues”).

• Event existence. The event described in the text query must genuinely exist within the video content.

• Query uniqueness. All queries must be unique in a single video. The presence of multiple nearly identical queries describing the same event is equivalent to duplicating or weighting certain samples, leading to biased metrics. Indeed, this issue is severe in Charades-STA dataset.

• Avoid information leakage in queries. Queries like “ending credits” leak their temporal position, allowing the model to answer via shortcut, without truly “grounding” the query over the entire video. However, annotators tend to label such queries since they are easy to identify.

Output Criteria. The temporal segment should satisfy:

• Annotation precision. The annotated event boundaries should be precise, excluding any subsegments that do not conform to the query’s description.

• Annotation exhaustiveness. There should be no other time segments outside the annotated one that also satisfy the query’s description.

## 3.2. Manual Auditing and Refinement

We introduce a rigorous and efficient pipeline for auditing and refining existing temporal grounding datasets.

Diagnose-then-Refine. Our pipeline follows a diagnosethen-refine workflow. Given a video-query pair from existing datasets, annotators first carefully review the video to identify potential errors against the criteria in Sec. 3.1. If an error is detected, they select the error category, then either revise the query or choose a new valid event to describe. Subsequently, the precise temporal segment is annotated. The core principle is that the same annotator performs both error detection and subsequent correction, which not only improves efficiency but also strengthens annotators’ awareness of potential errors, thereby reducing the risk of introducing similar ones.

![](assets/images/ea5d0ed03de76857bcbc50677b02ef1f50809272d6e78d7f4a43d4a6ac2c0356.jpg)  
Figure 4. Statistics of errors indicating alarmingly high proportion of errors in existing datasets.

Error Identification. Directly applying the abstract criteria from Sec. 3.1 for error detection proves overly challenging for annotators. Therefore, as shown in Fig. 3, we derive from these criteria a set of concrete, easily identifiable error types with clear illustrations. Annotators check whether each error type is present and fill in the corresponding information. Additionally, we group all queries from the same video together to detect violations of “query uniqueness” and improve annotation efficiency. During the process, we do not provide original temporal segments to annotators.

Quality Control. Upon completion of each small data batch, every sample is assigned to a different annotator for cross-validation and error correction. If the error rate in a batch exceeds a threshold, the entire batch is rejected for re-annotation and then validated again. For annotator selection and training, we sampled a small subset of data for trial annotation with over a dozen vendors, then selected the vendor with the highest quality and consistency. Before formal annotation, we provided a detailed handbook and conducted several training sessions. The annotation interface and detailed manual are provided in Sec. G of the appendix.

## 3.3. Empirical Analysis on TimeLens-Bench

In this section, we present our efforts and findings by applying the above annotation pipeline to existing datasets. We focus on three most widely-used temporal grounding benchmarks: Charades-STA [11], ActivityNet Captions [25], and QVHighlights [27]. These datasets exhibit diversity across video domains, video durations, and query semantics. They are all manually annotated and generally considered the highest-quality VTG datasets available. Therefore, analyzing them offers a representative view of the quality and issues prevalent in existing data. Through diagnosis and refinement, we release TimeLens-Bench, comprising refined versions of the three aforementioned benchmarks: Charades-TimeLens, ActivityNet-TimeLens, and QVHighlights-TimeLens. Together, they form a comprehensive evaluation suite that combines diversity with high quality. Detailed statistics for these benchmarks are provided in Sec. B.

Finding 1: Widely-used benchmarks have an alarmingly high proportion of errors.

Error Statistics and Analysis. As shown in Fig. 4, we observe an alarmingly high proportion of errors across different categories in these benchmarks. The distribution of error composition varies across different datasets, yet all datasets exhibit consistently high overall error rates. For example, in Charades-STA, we find that 20.6% of samples violate query uniqueness, while 34.9% exhibit annotation accuracy issues. Such severe errors will lead to unreliable evaluation results and misguide research efforts.

Qualitative Examples of Errors and Fixes. As shown in Fig. 3, various error examples are identified in existing datasets, including multiple event occurrences, no event occurrence, duplicate queries for the same video, unclear query, and inaccurate annotation. Through our rigorous manual refinement, these detected errors have been properly corrected, significantly improving data quality. Our refined datasets provide more reliable evaluation results.

Finding 2: Low-quality evaluation data inflates performance of open-source models, while underestimating proprietary models.

Counter-intuitive Evaluation Results. We evaluate various frontier models on both the original and refined benchmarks, observing drastically contrasting performance trends. As illustrated in Fig. 2a, on the original benchmarks, we observe a surprising phenomenon: frontier proprietary models like Gemini-2.5-Pro [8] receive poor scores, whereas opensource models [3, 54] attain significantly higher ones. Conversely, on our refined benchmarks, this trend reverses. The proprietary models exhibit much better results, though with room for improvement, while the open-source models suffer a substantial performance degradation, lagging far behind their proprietary counterparts. This reversal indicates that the original benchmarks produce misleading results due to inherent quality flaws, while our refined benchmarks yield results that align more closely with real-world user experience, providing reliable evaluation for developing better VTG models.

<table><tr><td rowspan="2">Model</td><td colspan="4">Charades-TimeLens</td><td colspan="4">ActivityNet-TimeLens</td><td colspan="4">QVHighlights-TimeLens</td></tr><tr><td>R1@0.3 R1@0.5 R1@0.7</td><td></td><td></td><td>mIoU</td><td>|R1@0.3 R1@0.5 R1@0.7 mIoU|</td><td></td><td></td><td></td><td>|R1@0.3 R1@0.5 R1@0.7 mIoU</td><td></td><td></td><td></td></tr><tr><td>Proprietary Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-40 [23]</td><td>60.6</td><td>44.5</td><td>23.5</td><td>41.8</td><td>55.2</td><td>41.4</td><td>25.8</td><td>40.4</td><td>69.0</td><td>54.8</td><td>38.5</td><td>52.1</td></tr><tr><td>GPT-5 [42]</td><td>59.3</td><td>42.0</td><td>22.0</td><td>40.5</td><td>57.4</td><td>44.9</td><td>30.4</td><td>42.9</td><td>72.4</td><td>60.4</td><td>46.4</td><td>56.8</td></tr><tr><td>Gemini-2.0-Flash [8]</td><td>66.4</td><td>53.5</td><td>27.1</td><td>46.7</td><td>62.9</td><td>54.0</td><td>37.7</td><td>49.3</td><td>76.2</td><td>66.4</td><td>48.3</td><td>60.8</td></tr><tr><td>Gemini-2.5-Flash [8]</td><td>68.7</td><td>56.1</td><td>30.6</td><td>48.6</td><td>66.8</td><td>57.5</td><td>41.3</td><td>52.5</td><td>78.2</td><td>69.4</td><td>55.0</td><td>64.3</td></tr><tr><td>Gemini-2.5-Pro [8]</td><td>74.1</td><td>61.1</td><td>34.0</td><td>52.8</td><td>72.3</td><td>64.2</td><td>47.1</td><td>58.1</td><td>84.1</td><td>75.9</td><td>61.1</td><td>70.4</td></tr><tr><td>Open-Source Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VideoChat-Flash-7B [31]</td><td>60.2</td><td>37.9</td><td>17.8</td><td>39.7</td><td>35.5</td><td>21.8</td><td>10.5</td><td>24.8</td><td>45.2</td><td>30.6</td><td>16.7</td><td>32.7</td></tr><tr><td>VideoChat-R1-7B [32]</td><td>51.9</td><td>30.8</td><td>11.7</td><td>33.7</td><td>35.0</td><td>23.9</td><td>11.3</td><td>25.0</td><td>29.3</td><td>19.1</td><td>9.4</td><td>21.5</td></tr><tr><td>Time-R1-7B [54]</td><td>57.9</td><td>32.0</td><td>16.9</td><td>36.6</td><td>44.8</td><td>31.0</td><td>19.0</td><td>33.1</td><td>65.8</td><td>51.5</td><td>36.1</td><td>49.2</td></tr><tr><td>TRACE [17]</td><td>37.2</td><td>21.8</td><td>9.6</td><td>27.1</td><td>43.4</td><td>33.9</td><td>22.0</td><td>32.7</td><td>49.7</td><td>39.1</td><td>28.1</td><td>39.0</td></tr><tr><td>TRACE-uni [17]</td><td>38.2</td><td>22.9</td><td>10.4</td><td>28.1</td><td>44.3</td><td>35.1</td><td>22.6</td><td>33.6</td><td>49.9</td><td>40.0</td><td>29.2</td><td>39.8</td></tr><tr><td>TimeSuite [61]</td><td>56.3</td><td>35.5</td><td>18.0</td><td>38.1</td><td>27.1</td><td>17.5</td><td>8.6</td><td>19.8</td><td>27.1</td><td>16.9</td><td>9.9</td><td>21.7</td></tr><tr><td>Grounded-VideoLLM [51]</td><td>43.3</td><td>28.7</td><td>13.5</td><td>30.0</td><td>39.2</td><td>29.6</td><td>19.5</td><td>30.0</td><td>43.7</td><td>33.8</td><td>22.5</td><td>33.4</td></tr><tr><td>MiMo-VL-7B [9]</td><td>57.9</td><td>42.6</td><td>20.5</td><td>39.6</td><td>49.3</td><td>38.7</td><td>22.4</td><td>35.5</td><td>57.1</td><td>42.6</td><td>28.4</td><td>41.5</td></tr><tr><td>Qwen2.5-VL-7B [3]</td><td>59.7</td><td>37.8</td><td>16.6</td><td>39.3</td><td>44.1</td><td>31.0</td><td>16.1</td><td>31.4</td><td>41.5</td><td>27.8</td><td>15.2</td><td>31.6</td></tr><tr><td>TimeLens-7B</td><td>70.5</td><td>55.6</td><td>28.4</td><td>48.8</td><td>62.8</td><td>51.0</td><td>32.6</td><td>46.2</td><td>74.1</td><td>62.7</td><td>43.1</td><td>56.0</td></tr><tr><td>Qwen3-VL-235B-A22B [2]</td><td>71.7</td><td>50.8</td><td>24.5</td><td>47.8</td><td>69.0</td><td>57.5</td><td>39.3</td><td>52.2</td><td>79.6</td><td>70.2</td><td>54.5</td><td>64.6</td></tr><tr><td>Qwen3-VL-8B [2]</td><td>69.2</td><td>53.4</td><td>27.5</td><td>48.3</td><td>62.1</td><td>51.2</td><td>34.4</td><td>46.8</td><td>74.2</td><td>64.6</td><td>49.3</td><td>59.4</td></tr><tr><td>TimeLens-8B</td><td>76.6</td><td>63.0</td><td>35.2</td><td>55.2</td><td>68.9</td><td>58.4</td><td>40.6</td><td>53.2</td><td>80.2</td><td>71.6</td><td>55.5</td><td>65.5</td></tr></table>

Table 1. Main Results. We benchmark the performance of various state-of-the-art proprietary and open-source models on TimeLens-Bench. Our TimeLens models are built upon their respective baseline models (preceding rows in the table). Our TimeLens-7B not only delivers substantial improvements over the Qwen2.5-VL baseline but also closes the gap with the more powerful Qwen3-VL-8B model. Building upon the stronger Qwen3-VL baseline, our TimeLens-8B pushes performance even further, setting a new state-of-the-art among open-source models and surpassing prominent proprietary models like GPT-5 and Gemini-2.5-Flash.

## 3.4. Training Data Re-annotation

By applying our manual pipeline from Sec. 3.2 to a sampled subset of existing VTG training corpus [1, 22, 41, 50, 60], we found that the training data exhibits an even higher error rate compared to the evaluation benchmarks. This motivated us to refine training data based on scalable re-annotation. Given the vast scale of the training sets, we employ an automated pipeline to improve their quality based on advanced multimodal models. Owing to the poor quality of these training datasets, especially the high proportion of queries that fail to meet our criteria in Sec. 3.1, we re-annotate the videos rather than refining existing labels. Through this process, we curate TimeLens-100K, a large-scale, high-quality, and diverse VTG training set. Additional details are provided in Sec. H.

Finding 3: Improved annotation quality in training data yields stronger grounding ability.

As presented in Fig. 2b, models trained on TimeLens-100K demonstrate substantially improved performance on our refined evaluation benchmarks. This performance gain serves as a direct validation of the data’s enhanced quality. Notably, our automated re-annotation for training data is developed entirely independently of the manual benchmark refinement process, ensuring an unbiased evaluation.

## 4. Benchmarking Grounding MLLMs

In this section, we benchmark the performance of various state-of-the-art proprietary and open-source models on TimeLens-Bench, including our TimeLens models derived from the exploration in Sec. 5.

Evaluation Metrics. We evaluate VTG performance using the “R1@m” metric, which measures the proportion of test instances where the highest-ranked predicted segment achieves an IoU exceeding threshold m (where m takes values from 0.3, 0.5, 0.7). Additionally, we employ mIoU as a primary measure, computing the mean IoU across the entire test set for conciseness.

Evaluation Results. As shown in Tab. 1, we observe a significant performance gap between existing open-source and proprietary models, and our TimeLens models substantially narrow this gap. TimeLens-7B delivers substantial improvements over its baseline, demonstrating the effectiveness of the insights and best practices obtained from our experiments in Sec. 5. It surpasses strong open-source competitors such as Time-R1-7B [54] and MiMo-VL-7B [9], as well as proprietary models like GPT-4o [23] and GPT-5 [42]. More remarkably, on the already stronger baseline Qwen3- VL-8B, our TimeLens-8B model still achieves substantial performance gains, establishing a new state-of-the-art among open-source models and even surpassing frontier proprietary models like Gemini-2.5-Flash [8].

![](assets/images/5124ca20f68b35675cf635378dc134dab9bd185010f3972c0f6639d92aca0082.jpg)  
(a) Interleaved Textual Timestamp Encoding.

![](assets/images/a285ae08e431ee8a6c51d4a17c14e1ba5ead860818ba00485b79108c73375ac4.jpg)  
(b) Visual Timestamp Overlay.

![](assets/images/5417a8802e0c16a6a715528eefd6558073854d5111e6180d2621bd6a848422e7.jpg)  
(c) Position-embedding-based Time Encoding.

Figure 5. Illustration of different timestamp encoding schemes. (a) Textual Encoding uses the text tokenizer of LLMs to tokenize textual timestamps into textual tokens. (b) Visual Overlay directly overlays timestamps as visual text onto the corresponding frames. (c) Position-embedding-based Methods aligns the positional encodings of visual tokens in the LLM with the sampling time of each frame.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Timestamp Format</td><td colspan="4">Charades-TimeLens</td><td colspan="4">ActivityNet-TimeLens</td><td colspan="4">QVHighlights-TimeLens</td></tr><tr><td>|R1@0.3 R1 @0.5 R1@0.7 mIoU|R1@0.3 R1@0.5 R1@0.7 mIoU|R1@0.3 R1@0.5 R1 @0.7 mIoU</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Position Embed. [3]</td><td></td><td>57.9 65.5</td><td>32.0</td><td>16.9</td><td>36.6</td><td>44.8</td><td>31.0</td><td>19.0</td><td>33.1</td><td>65.8</td><td>51.5 43.6</td><td>36.1</td><td>49.2</td></tr><tr><td>Visual Overlay</td><td>Frame Index Raw Timestamp</td><td>67.6</td><td>48.0 50.7</td><td>22.2 25.7</td><td>44.0 46.3</td><td>47.5 54.0</td><td>34.0 42.2</td><td>17.8 26.3</td><td>33.3 39.8</td><td>61.4 70.0</td><td>58.3</td><td>24.1 42.1</td><td>42.3 53.6</td></tr><tr><td>Not-Interleaved Textual Prefix</td><td>Raw Timestamp</td><td>64.9</td><td>49.4</td><td>27.3</td><td>45.8</td><td>48.2</td><td>35.5</td><td>21.4</td><td>35.2</td><td>59.7</td><td>45.8</td><td>26.4</td><td>42.8</td></tr><tr><td>Interleaved</td><td>Frame Index</td><td>66.0</td><td>51.6</td><td>25.6</td><td>45.6</td><td>51.0</td><td>39.1</td><td>23.1</td><td>36.9</td><td>64.4</td><td>52.1</td><td>32.3</td><td>47.2</td></tr><tr><td>Textual Prefix</td><td>Raw Timestamp</td><td>70.0</td><td>53.9</td><td>28.1</td><td>48.3</td><td>57.9</td><td>46.3</td><td>30.5</td><td>43.1</td><td>73.0</td><td>62.2</td><td>46.1</td><td>56.7</td></tr></table>

Table 2. Ablation on timestamp encoding methods. For each method, we experiment with two timestamp formats: raw timestamps (e.g., “10.2s”) or frame indices (e.g., “1, 2, 3”). “Position Embed.” means “Position Embedding”. Results show that interleaved textual prefix with raw timestamps is the most effective approach, while maintaining simplicity.

## 5. Exploring Algorithmic Designs

In this section, we conduct a systematic study on the algorithmic designs for improving MLLMs’ VTG performance, covering various aspects from model architectures to training strategies. Leveraging our high-quality training and evaluation suites as a reliable testbed, we derive several novel and valuable insights. As shown in Fig. 2b, each of our findings contributes a non-trivial performance gain, ultimately culminating in our TimeLens model.

Experimental Setup. Our experiments use Qwen2.5-VL-7B [3] as the baseline. For RLVR experiments, we employ GRPO [47] as optimization method. We use TimeLens-Bench for evaluation and TimeLens-100K for training. To ensure rigor, all ablation studies are based on the final, bestperforming model configuration, isolating the impact of a single design choice at a time. Due to limited computational resources, we adopt a lower per-frame resolution for our ablation experiments. More implementation details are provided in Sec. C of the appendix.

## 5.1. Timestamp Encoding

Finding 4: Encoding timestamps as interleaved textual prefix is the most effective while maintaining simplicity.

To enable MLLMs to perform temporal grounding, a critical design decision is timestamp encoding (i.e., aligning the timestamp of each frame with its corresponding features). Effective timestamp encoding allows the model to accurately perceive the absolute temporal position of each frame and the relative order between frames, thereby producing precise localization results. As illustrated in Fig. 5, various timestamp encoding strategies have been proposed:

• Position-embedding based. These methods adapt position embeddings in LLMs to represent the temporal position of each frame. For example, MRoPE [3, 9] and 3D RoPE [48] extend pure-text RoPE to multimodal scenarios, encoding the spatial and temporal dimensions of video frame tokens.

<table><tr><td>Training Paradigm</td><td>Training Time</td><td colspan="3">Charades-TimeLens R1@0.3 R1@0.5 R1@0.7 mIoU R1@0.3 R1@0.5 R1@0.7 mIoUR1@0.3 R1@0.5 R1@0.7 mIoU</td><td colspan="4">ActivityNet-TimeLens</td><td colspan="4">QVHighlights-TimeLens</td></tr><tr><td></td><td>1.0×</td><td>68.8 53.0</td><td>26.2</td><td>47.4</td><td>53.3</td><td>42.6</td><td></td><td></td><td></td><td>40.6</td><td></td></tr><tr><td>SFT (32K Data)</td><td></td><td>70.6</td><td>54.9</td><td></td><td>53.2</td><td></td><td>27.5</td><td>39.9 39.7</td><td>65.8 63.1</td><td>54.8 51.1</td><td>52.0</td></tr><tr><td>SFT (100K Data)</td><td>2.4× 1.9×</td><td>60.3 46.4</td><td>27.1 24.7</td><td>48.6 42.7</td><td>54.3</td><td>43.1 44.2</td><td>27.2</td><td></td><td></td><td>36.9</td><td>49.0</td></tr><tr><td>Thinking-based RLVR SFT +</td><td></td><td></td><td></td><td></td><td></td><td></td><td>29.1</td><td>41.2</td><td>72.1 62.7</td><td>48.2</td><td>57.8</td></tr><tr><td>Thinking-free RLVR</td><td>2.9×</td><td>71.7 56.7</td><td>29.8</td><td>50.1</td><td>56.9</td><td>46.1</td><td>30.1</td><td>42.7</td><td>72.2</td><td>60.6 43.8</td><td>55.9</td></tr><tr><td>Thinking-free RLVR</td><td>1.0×</td><td>70.0</td><td>53.9 28.1</td><td>48.3</td><td>57.9</td><td>46.3</td><td>30.5</td><td>43.1</td><td>73.0</td><td>62.2 46.1</td><td>56.7</td></tr></table>

Table 3. Ablation on different training paradigms. We compare the performance and efficiency of different training paradigms, showing that thinking-free RLVR achieves the best performance while maintaining high efficiency. All training is conducted on our quality-improved TimeLens-100K training data. Training time is measured on 8 H20 GPUs, where 1.0 corresponds to approximately 4h10m. As described in Sec. 5.3, before RLVR training, offline inference on the training data is required to select samples with appropriate difficulty; this time is also included in the reported RLVR training time.

• Visual overlay. These methods [7, 12, 55] directly overlay timestamps or frame index onto each frame, enabling MLLMs to “read” the temporal position through their OCR capabilities.

• Textual encoding. These methods convert timestamps into text tokens using the MLLM’s text tokenizer. There are two main variants: the Interleaved approach [5, 15, 20, 34, 56] in Fig. 5a inserts timestamp tokens before the visual tokens of each frame. In contrast, the Non-interleaved approach [29, 31, 52] adds an instruction like “This video samples N frames of a T-second video at t<sub>1</sub>, t<sub>2</sub>, . . . seconds.” into the prompt.

We conduct a comprehensive comparison of different timestamp encoding methods. For each method, we experiment with two timestamp formats: raw timestamps (e.g., “10.2s”) or frame indices (e.g., “1, 2, 3”), which are simpler but neglects the temporal interval between frames. As shown in Tab. 2, our results reveal: Position-embedding based methods yield unsatisfactory results. Given that they require fundamental modifications to the RoPE mechanism in LLMs, their practicality is limited without large-scale retraining. Instead, interleaved textual prefix with raw timestamps achieves the best performance among all approaches, while remaining simple and intuitive.

## 5.2. Optimization Paradigms

Finding 5: For the optimization paradigm, a pure thinking-free RLVR approach achieves superior performance and efficiency. Both SFT and thinking-based RLVR are not necessary.

In this section, we review different training paradigms and conduct systematic experiments to compare their effectiveness and efficiency for VTG, seeking insights into the optimal training paradigm.

Earlier works [17, 18, 22, 46, 61] employ supervised finetuning (SFT) to improve MLLMs’ VTG capability. Recently, some works [4, 54] utilize reinforcement learning with verifiable rewards (RLVR), following a “think-then-answer” approach [16] (details in Sec. C): during sampling, the model first generates an explicit thinking process and then produces the final answer. The task-specific VTG accuracy reward is computed only on the final answer. Despite these efforts, there lacks a systematic comparison of the respective merits of these methods, leaving some key questions unanswered:

• Is RLVR superior to SFT? While the pioneering work Time-R1 [54] demonstrates that RLVR outperforms SFT, they compare the two methods using the same amount of training data, despite RLVR requiring significantly more training time. A fair comparison under equal training budgets remains absent.

• Is explicit “thinking” necessary for RLVR? Recent works suggest that the thinking process is not essential when applying RLVR to visual perception such as counting [30, 43]. Whether this holds for VTG, a predominantly perception-oriented task, remains unanswered.

• Does a preceding SFT phase benefit RLVR? An SFT phase prior to RLVR is typically employed to enhance the model’s capability and facilitate subsequent RLVR training [9, 48]. However, whether this preceding SFT phase actually improves final performance in the VTG scenario remains unexplored.

In Tab. 3, we compare the performance and efficiency of different training paradigms. Our results reveal that thinkingfree RLVR surpasses both SFT and thinking-based RLVR in performance while being more efficient. Adding a preceding SFT phase before RLVR yields no significant performance gain compared to pure RLVR. Overall, a pure thinking-free RLVR approach maintains simplicity, superior performance, and high efficiency.

![](assets/images/3e6be480abf92876df1e95bc64be450e93978da83fddbe734e507455d8aa8ea1.jpg)  
Figure 6. The effectiveness of early stopping for RLVR. We show the trends of training reward and evaluation metrics during RLVR training. When the temporal IoU reward and the withingroup reward standard deviation plateau, performance reaches its peak. Continued training beyond this point leads to performance degradation. Therefore, performing early stopping when the reward plateaus ensures optimal training efficiency and performance. Training is conducted on 12K samples selected from TimeLens-100K via difficulty-aware sampling.

## 5.3. Recipes for RLVR Training

Building on the finding in Sec. 5.2 that thinking-free RLVR is the optimal training paradigm, in this section, we further explore effective recipes for RLVR training, focusing on two key questions: (i) How long should we train? (ii) How to effectively sample training data?

Finding 6: For RLVR training, performing early stopping when reward metrics plateau saves computational cost, while preventing performance degradation.

How long should we train? In SFT, the prevailing wisdom is “train longer, generalize better” [19]. Given training data with sufficient scale and quality, we typically train MLLMs for at least one full epoch over the entire dataset, ensuring the model sees as much data as possible to enhance generalization. However, whether this strategy is optimal for RLVR remains to be explored.

In Fig. 6, we conduct RLVR training on 12K data from TimeLens-100K, tracking the reward and evaluating model checkpoints at different training steps on our evaluation benchmarks. When the temporal IoU reward and the withingroup reward standard deviation plateau, model performance has reached its peak. Continued training beyond this point leads to performance degradation. Therefore, in RL training, even with sufficiently high data quality, training for a full epoch over all available data is suboptimal. A good practice is performing early stopping when reward metrics plateau, which not only saves computational cost but also prevents performance degradation.

![](assets/images/8d37424a4095cd8bfd374e64ea9c6625ef743d3b23d1347ee8b1923a0d6cfa64.jpg)  
Figure 7. The importance of difficulty-based training data sampling for RLVR. We investigate the impact of training data with different difficulty levels on performance by adjusting the mean of the Gaussian distribution for difficulty-based data sampling. Model performance improves as the average sample difficulty increases, and eventually plateaus when difficulty becomes high. This demonstrates that selecting samples with sufficiently high difficulty is crucial for achieving optimal performance.

Finding 7: For RLVR training, sampling training data with sufficiently high difficulty relative to the model is crucial for performance.

How to sample training data? For RLVR training, it is crucial to select samples with appropriate difficulty relative to the model, and many works propose assessing training sample difficulty and employing difficulty-aware sampling [21, 54, 57]. To evaluate the impact of sample difficulty on video temporal grounding, we conduct experiments on our TimeLens-100K high-quality training corpus. Following prior works [54, 57], we use the model to be trained to perform offline inference on the training data, compute IoU metrics to estimate sample difficulty, and then perform Gaussian sampling based on sample difficulty (details in Sec. C). By varying the mean of the Gaussian distribution, we obtain training sets with different difficulty levels relative to the model, and conduct RLVR training on each set independently.

As shown in Fig. 7, model performance improves as the average sample difficulty increases, and eventually plateaus when difficulty becomes sufficiently high (over 0.75). This trend demonstrates that selecting training samples with sufficiently high difficulty relative to the model is crucial for achieving optimal performance.

Acknowledgements. This work is supported by the National Key R&D Program of China (No. 2022ZD0160900), the Basic Research Program of Jiangsu (No. BK20250009), the Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of China (No. JYB2025XDXM118), and the Collaborative Innovation Center of Novel Software Technology and Industrialization.

## References

[1] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. In Proceedings of the IEEE international conference on computer vision, pages 5803– 5812, 2017. 2, 5, 17

[2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 1, 5, 15, 18

[3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 1, 4, 5, 6, 13, 14, 15, 16, 18

[4] Ruizhe Chen, Zhiting Fan, Tianze Luo, Heqing Zou, Zhaopeng Feng, Guiyang Xie, Hansheng Zhang, Zhuochen Wang, Zuozhu Liu, and Huaijian Zhang. Datasets and recipes for video temporal grounding via reinforcement learning. arXiv preprint arXiv:2507.18100, 2025. 3, 7

[5] Shimin Chen, Xiaohan Lan, Yitian Yuan, Zequn Jie, and Lin Ma. Timemarker: A versatile video-llm for long and short video understanding with superior temporal localization ability. arXiv preprint arXiv:2411.18211, 2024. 3, 7

[6] Yukang Chen, Wei Huang, Baifeng Shi, Qinghao Hu, Hanrong Ye, Ligeng Zhu, Zhijian Liu, Pavlo Molchanov, Jan Kautz, Xiaojuan Qi, et al. Scaling rl to long videos. arXiv preprint arXiv:2507.07966, 2025. 1

[7] Jen-Hao Cheng, Vivian Wang, Huayu Wang, Huapeng Zhou, Yi-Hao Peng, Hou-I Liu, Hsiang-Wei Huang, Kuang-Ming Chen, Cheng-Yen Yang, Wenhao Chai, et al. Tempura: Temporal event masked prediction and understanding for reasoning in action. arXiv preprint arXiv:2505.01583, 2025. 3, 7

[8] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing

the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 4, 5, 6, 14, 15, 16, 17

[9] Team Core, Zihao Yue, Zhenru Lin, Yifan Song, Weikun Wang, Shuhuai Ren, Shuhao Gu, Shicheng Li, Peidian Li, Liang Zhao, et al. Mimo-vl technical report. arXiv preprint arXiv:2506.03569, 2025. 5, 6, 7, 14, 15, 18

[10] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24108–24118, 2025. 16

[11] Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia. Tall: Temporal activity localization via language query. In Proceedings of the IEEE international conference on computer vision, pages 5267–5275, 2017. 1, 2, 4, 12

[12] Yuying Ge, Yixiao Ge, Chen Li, Teng Wang, Junfu Pu, Yizhuo Li, Lu Qiu, Jin Ma, Lisheng Duan, Xinyu Zuo, et al. Archunyuan-video-7b: Structured video comprehension of realworld shorts. arXiv preprint arXiv:2507.20939, 2025. 3, 7

[13] Sara Ghazanfari, Francesco Croce, Nicolas Flammarion, Prashanth Krishnamurthy, Farshad Khorrami, and Siddharth Garg. Chain-of-frames: Advancing video understanding in multimodal llms via frame-aware reasoning. arXiv preprint arXiv:2506.00318, 2025. 1

[14] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18995–19012, 2022. 2

[15] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025. 7

[16] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 7, 13

[17] Yongxin Guo, Jingyu Liu, Mingda Li, Qingbin Liu, Xi Chen, and Xiaoying Tang. Trace: Temporal grounding video llm via causal event modeling. arXiv preprint arXiv:2410.05643, 2024. 2, 5, 7

[18] Yongxin Guo, Jingyu Liu, Mingda Li, Dingxin Cheng, Xiaoying Tang, Dianbo Sui, Qingbin Liu, Xi Chen, and Kevin Zhao. Vtg-llm: Integrating timestamp knowledge into video llms for enhanced video temporal grounding. In Proceedings ofthe AAAI Conference on Artificial Intelligence, pages 3302–3310, 2025. 7, 12

[19] Elad Hoffer, Itay Hubara, and Daniel Soudry. Train longer, generalize better: closing the generalization gap in large batch training of neural networks. Advances in neural information processing systems, 30, 2017. 8

[20] Wenyi Hong, Weihan Wang, Ming Ding, Wenmeng Yu, Qingsong Lv, Yan Wang, Yean Cheng, Shiyu Huang, Junhui Ji, Zhao Xue, et al. Cogvlm2: Visual language models for image and video understanding. arXiv preprint arXiv:2408.16500, 2024. 7

[21] Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. Glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv e-prints, pages arXiv–2507, 2025. 8, 13

[22] Bin Huang, Xin Wang, Hong Chen, Zihan Song, and Wenwu Zhu. Vtimellm: Empower llm to grasp video moments. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14271–14280, 2024. 2, 5, 7, 17

[23] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 5, 6, 14, 16, 17

[24] Xuan Ju, Yiming Gao, Zhaoyang Zhang, Ziyang Yuan, Xintao Wang, Ailing Zeng, Yu Xiong, Qiang Xu, and Ying Shan. Miradata: A large-scale video dataset with long durations and structured captions. Advances in Neural Information Processing Systems, 37:48955–48970, 2024. 17

[25] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. In Proceedings of the IEEE international conference on computer vision, pages 706–715, 2017. 1, 2, 4, 12

[26] Xiaohan Lan, Yitian Yuan, Xin Wang, Zhi Wang, and Wenwu Zhu. A survey on temporal sentence grounding in videos. ACM Transactions on Multimedia Computing, Communications and Applications, 19(2):1–33, 2023. 1

[27] Jie Lei, Tamara L Berg, and Mohit Bansal. Detecting moments and highlights in videos via natural language queries. Advances in Neural Information Processing Systems, 34: 11846–11858, 2021. 1, 2, 4, 12

[28] Hongyu Li, Jinyu Chen, Ziyu Wei, Shaofei Huang, Tianrui Hui, Jialin Gao, Xiaoming Wei, and Si Liu. Llava-st: A multimodal large language model for fine-grained spatialtemporal understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8592–8603, 2025. 3

[29] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023. 7

[30] Ming Li, Jike Zhong, Shitian Zhao, Yuxiang Lai, Haoquan Zhang, Wang Bill Zhu, and Kaipeng Zhang. Think or not think: A study of explicit thinking in rule-based visual reinforcement fine-tuning. arXiv preprint arXiv:2503.16188, 2025. 7

[31] Xinhao Li, Yi Wang, Jiashuo Yu, Xiangyu Zeng, Yuhan Zhu, Haian Huang, Jianfei Gao, Kunchang Li, Yinan He, Chenting Wang, et al. Videochat-flash: Hierarchical compression for long-context video modeling. arXiv preprint arXiv:2501.00574, 2024. 5, 7, 14, 18

[32] Xinhao Li, Ziang Yan, Desen Meng, Lu Dong, Xiangyu Zeng, Yinan He, Yali Wang, Yu Qiao, Yi Wang, and Limin Wang.

Videochat-r1: Enhancing spatio-temporal perception via reinforcement fine-tuning. arXiv preprint arXiv:2504.06958, 2025. 5, 14, 16, 18

[33] Yunheng Li, Jing Cheng, Shaoyong Jia, Hangyi Kuang, Shaohui Jiao, Qibin Hou, and Ming-Ming Cheng. Tempsamp-r1: Effective temporal sampling with reinforcement fine-tuning for video llms. arXiv preprint arXiv:2509.18056, 2025. 3

[34] Zeqian Li, Shangzhe Di, Zhonghua Zhai, Weilin Huang, Yanfeng Wang, and Weidi Xie. Universal video temporal grounding with generative multi-modal large language models. arXiv preprint arXiv:2506.18883, 2025. 3, 7

[35] Kevin Qinghong Lin, Pengchuan Zhang, Joya Chen, Shraman Pramanick, Difei Gao, Alex Jinpeng Wang, Rui Yan, and Mike Zheng Shou. Univtg: Towards unified video-language temporal grounding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2794–2804, 2023. 1

[36] Ye Liu, Zongyang Ma, Zhongang Qi, Yang Wu, Ying Shan, and Chang W Chen. Et bench: Towards open-ended eventlevel video-language understanding. Advances in Neural Information Processing Systems, 37:32076–32110, 2024. 2

[37] Ye Liu, Kevin Qinghong Lin, Chang Wen Chen, and Mike Zheng Shou. Videomind: A chain-of-lora agent for long video reasoning. arXiv preprint arXiv:2503.13444, 2025. 1, 2, 12

[38] Chujie Lu, Long Chen, Chilie Tan, Xiaolin Li, and Jun Xiao. Debug: A dense bottom-up grounding approach for natural language video localization. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 5144–5153, 2019. 2

[39] Arsha Nagrani, Mingda Zhang, Ramin Mehran, Rachel Hornung, Nitesh Bharadwaj Gundavarapu, Nilpa Jha, Austin Myers, Xingyi Zhou, Boqing Gong, Cordelia Schmid, et al. Neptune: The long orbit to benchmarking long video understanding. arXiv preprint arXiv:2412.09582, 2024. 1

[40] Arsha Nagrani, Sachit Menon, Ahmet Iscen, Shyamal Buch, Ramin Mehran, Nilpa Jha, Anja Hauth, Yukun Zhu, Carl Vondrick, Mikhail Sirotenko, et al. Minerva: Evaluating complex video reasoning. arXiv preprint arXiv:2505.00681, 2025. 1

[41] Andreea-Maria Oncescu, Joao F Henriques, Yang Liu, Andrew Zisserman, and Samuel Albanie. Queryd: A video dataset with high-quality text and audio narrations. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 2265– 2269. IEEE, 2021. 2, 5, 17

[42] OpenAI. Introducing gpt-5, 2025. Available from OpenAI announcement, August 7, 2025. 5, 6, 14, 17

[43] Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Mateusz Malinowski, Yi Yang, Carl Doersch, et al. Perception test: A diagnostic benchmark for multimodal video models. Advances in Neural Information Processing Systems, 36:42748–42761, 2023. 1, 7

[44] Yukun Qi, Yiming Zhao, Yu Zeng, Xikun Bao, Wenxuan Huang, Lin Chen, Zehui Chen, Jie Zhao, Zhongang Qi, and

Feng Zhao. Vcr-bench: A comprehensive evaluation framework for video chain-of-thought reasoning. arXiv preprint arXiv:2504.07956, 2025. 1

[45] Michaela Regneri, Marcus Rohrbach, Dominikus Wetzel, Stefan Thater, Bernt Schiele, and Manfred Pinkal. Grounding action descriptions in videos. Transactions ofthe Association for Computational Linguistics, 1:25–36, 2013. 2

[46] Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. Timechat: A time-sensitive multimodal large language model for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14313–14323, 2024. 1, 2, 3, 7

[47] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 6, 12

[48] Kwai Keye Team, Biao Yang, Bin Wen, Changyi Liu, Chenglong Chu, Chengru Song, Chongling Rao, Chuan Yi, Da Li, Dunju Zang, et al. Kwai keye-vl technical report. arXiv preprint arXiv:2507.01949, 2025. 6, 7

[49] Vidi Team, Celong Liu, Chia-Wen Kuo, Dawei Du, Fan Chen, Guang Chen, Jiamin Yuan, Lingxi Zhang, Lu Guo, Lusha Li, et al. Vidi: Large multimodal models for video understanding and editing. arXiv preprint arXiv:2504.15681, 2025. 1, 16

[50] Alex Jinpeng Wang, Linjie Li, Kevin Qinghong Lin, Jianfeng Wang, Kevin Lin, Zhengyuan Yang, Lijuan Wang, and Mike Zheng Shou. Cosmo: Contrastive streamlined multimodal model with interleaved pre-training. arXiv preprint arXiv:2401.00849, 2024. 2, 5, 17

[51] Haibo Wang, Zhiyang Xu, Yu Cheng, Shizhe Diao, Yufan Zhou, Yixin Cao, Qifan Wang, Weifeng Ge, and Lifu Huang. Grounded-videollm: Sharpening fine-grained temporal grounding in video large language models. arXiv preprint arXiv:2410.03290, 2024. 3, 5

[52] Jiankang Wang, Zhihan Zhang, Zhihang Liu, Yang Li, Jiannan Ge, Hongtao Xie, and Yongdong Zhang. Spacevllm: Endowing multimodal large language model with spatio-temporal video grounding capability. arXiv preprint arXiv:2503.13983, 2025. 7

[53] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Zun Wang, Yansong Shi, et al. Internvideo2: Scaling foundation models for multimodal video understanding. In European Conference on Computer Vision, pages 396–416. Springer, 2024. 1

[54] Ye Wang, Ziheng Wang, Boshen Xu, Yang Du, Kejun Lin, Zihan Xiao, Zihao Yue, Jianzhong Ju, Liang Zhang, Dingyi Yang, et al. Time-r1: Post-training large vision language model for temporal video grounding. arXiv preprint arXiv:2503.13377, 2025. 1, 3, 4, 5, 6, 7, 8, 13, 14, 17, 18

[55] Yongliang Wu, Xinting Hu, Yuyang Sun, Yizhou Zhou, Wenbo Zhu, Fengyun Rao, Bernt Schiele, and Xu Yang. Number it: Temporal grounding videos like flipping manga. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 13754–13765, 2025. 1, 3, 7

[56] Linli Yao, Haoning Wu, Kun Ouyang, Yuanxing Zhang, Caiming Xiong, Bei Chen, Xu Sun, and Junnan Li. Generative

frame sampler for long video understanding. arXiv preprint arXiv:2503.09146, 2025. 7

[57] Ruifeng Yuan, Chenghao Xiao, Sicong Leng, Jianyu Wang, Long Li, Weiwen Xu, Hou Pong Chan, Deli Zhao, Tingyang Xu, Zhongyu Wei, et al. Vl-cogito: Progressive curriculum reinforcement learning for advanced multimodal reasoning. arXiv preprint arXiv:2507.22607, 2025. 8, 13

[58] Yitian Yuan, Xiaohan Lan, Xin Wang, Long Chen, Zhi Wang, and Wenwu Zhu. A closer look at temporal sentence grounding in videos: Dataset and metric. In Proceedings of the 2nd international workshop on human-centric multimedia analysis, pages 13–21, 2021. 1

[59] Feng Yue, Zhaoxing Zhang, Junming Jiao, Zhengyu Liang, Shiwen Cao, Feifei Zhang, and Rong Shen. Tempo-r0: A video-mllm for temporal video grounding through efficient temporal sensing reinforcement learning. arXiv preprint arXiv:2507.04702, 2025. 3

[60] Abhay Zala, Jaemin Cho, Satwik Kottur, Xilun Chen, Barlas Oguz, Yashar Mehdad, and Mohit Bansal. Hierarchical videomoment retrieval and step-captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23056–23065, 2023. 2, 5, 17

[61] Xiangyu Zeng, Kunchang Li, Chenting Wang, Xinhao Li, Tianxiang Jiang, Ziang Yan, Songze Li, Yansong Shi, Zhengrong Yue, Yi Wang, et al. Timesuite: Improving mllms for long video understanding via grounded tuning. arXiv preprint arXiv:2410.19702, 2024. 3, 5, 7

[62] Yingsen Zeng, Zepeng Huang, Yujie Zhong, Chengjian Feng, Jie Hu, Lin Ma, and Yang Liu. Distime: Distribution-based time representation for video large language models. arXiv preprint arXiv:2505.24329, 2025. 3

[63] Pan Zhang, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Rui Qian, Lin Chen, Qipeng Guo, Haodong Duan, Bin Wang, Linke Ouyang, et al. Internlm-xcomposer-2.5: A versatile large vision language model supporting long-contextual input and output. arXiv preprint arXiv:2407.03320, 2024. 17

[64] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024. 17

[65] Songyang Zhang, Houwen Peng, Jianlong Fu, and Jiebo Luo. Learning 2d temporal adjacent networks for moment localization with natural language. In Proceedings of the AAAI conference on artificial intelligence, pages 12870–12877, 2020. 2

[66] Yongheng Zhang, Xu Liu, Ruihan Tao, Qiguang Chen, Hao Fei, Wanxiang Che, and Libo Qin. Vitcot: Video-text interleaved chain-of-thought for boosting video understanding in large language models. arXiv preprint arXiv:2507.09876, 2025. 1