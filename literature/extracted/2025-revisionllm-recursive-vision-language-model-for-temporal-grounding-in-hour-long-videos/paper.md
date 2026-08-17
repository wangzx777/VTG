# ReVisionLLM: Recursive Vision-Language Model for Temporal Grounding in Hour-Long Videos

Tanveer Hannan<sup>1,2\*</sup> Md Mohaiminul Islam<sup>3</sup> Jindong Gu<sup>4</sup> Thomas Seidl<sup>1,2</sup> Gedas Bertasius<sup>3</sup> <sup>1</sup> LMU Munich <sup>2</sup> MCML <sup>3</sup> UNC Chapel Hill <sup>4</sup> University of Oxford

## Abstract

Large language models (LLMs) excel at retrieving information from lengthy text, but their vision-language counterparts (VLMs) face difficulties with hour-long videos, especially for temporal grounding. Specifically, these VLMs are constrained by frame limitations, often losing essential temporal details needed for accurate event localization in extended video content. We propose ReVisionLLM, a recursive vision-language model designed to locate events in hour-long videos. Inspired by human search strategies, our model initially targets broad segments of interest, progressively revising itsfocus to pinpoint exact temporal boundaries. Our model can seamlessly handle videos of vastly different lengths—from minutes to hours. We also introduce a hierarchical training strategy that starts with short clips to capture distinct events and progressively extends to longer videos. To our knowledge, ReVision-LLM is the first VLM capable of temporal grounding in hour-long videos, outperforming previous state-of-the-art methods across multiple datasets by a significant margin (e.g., +2.6% R1@0.1 on MAD). The code is available at https://github.com/Tanveer81/ReVisionLLM

## 1. Introduction

Large language models (LLMs) are particularly adept at handling extensive text documents, such as full-length books, and retrieving relevant information [1–3, 19, 32]. However, achieving similar capabilities in video, i.e., locating fine-grained temporal events in hour-long videos, remains a critical challenge. This task, known as longvideo temporal grounding, requires accurately identifying the start and end of events based on a user’s textual query. This capability could be paramount for video content search, sports analytics, surveillance, and many other applications. However, current vision-language models (VLMs) struggle with this demanding task.

Recently, non-LLM-based models [13, 14, 37, 39] have made progress in long temporal video grounding. However, these methods typically involve multiple networks and complex post-processing steps. Additionally, these models generally lack the flexibility to handle textual user instructions. In contrast, the recent VLMs [16, 17, 40, 43] can effectively process textual user queries but are ineffective for temporal localization in long videos (Fig. 1). In particular, such VLM-based approaches tend to underperform even on short video (e.g., 2 minutes) localization tasks compared to non-LLM approaches [24, 29, 33, 36].

![](assets/images/2c07b436fbff31f1a2224a52f4817a59da4f53d38b300a17e83b2b096759f38f.jpg)  
Figure 1. Existing vision-language models (VLMs) such as VTimeLLM [16] are not equipped to process hour-long videos effectively and struggle to pinpoint precise temporal boundaries for events within extended video durations. In contrast, ReVision-LLM is the first VLM designed to address this limitation, enabling accurate temporal grounding in hour-long video content.

Extending LLM-based solutions to hour-long video inputs for temporal localization presents several important challenges. Video data is much denser than text, leading to a massive number of input tokens for the LLM. To handle this, many VLMs downsample frames and operate on a limited number of frames [16, 21, 28, 54, 63], which leads to a significant loss of information in long videos. Moreover, training VLMs on hour-long videos requires immense memory and computational resources, which presents practical challenges for scalability and training efficiency. Furthermore, current VLMs often exhibit poor confidence calibration [23, 38], leading to frequent false positives with high confidence. This issue is amplified in long videos, where distinguishing actual events from numerous false detections becomes increasingly difficult.

To address these challenges, we introduce the Recursive

Vision-Language Model ReVisionLLM, a VLM with hierarchical perception that processes videos recursively. Cognitive studies [5, 52] suggest that when searching content, humans maintain a mental representation of the target and direct attention to the most promising areas, refining their search. In line with these principles, given a long video input, our recursive model first identifies broad video segments of interest and then progressively revises its focus, narrowing in on the event’s exact temporal boundaries. In Fig. 2, we show the operating principle of our model. At the top hierarchy, the model operates broadly, identifying relevant segments (e.g., 5 minutes) from a 2-hour-long video. As it moves down the intermediate hierarchies, it narrows its focus to increasingly fine-grained temporal segments at the lowest hierarchy, pinpointing precise event boundaries (e.g., 3.5 seconds). Such a recursive processing structure of our model allows it to scale effectively to hour-long videos.

We first train our model on short video segments, then progress to training on hour-long videos. In the short video training phase, we introduce contrastive segments (i.e., video clips that do not contain the queried event) to improve confidence calibration. This helps the model learn to identify both the presence and absence of events, enhancing its confidence in visual input and aiding in accurate event localization within long videos. For efficient training on hour-long videos, we employ a temporal feature reduction strategy that compresses video segments into compact representations. This method reduces the input tokens required by the LLM. As a result, our model achieves both high accuracy and efficiency, making it well-suited for analyzing lengthy videos. Our contributions can be summarized as follows:

• We extend the existing VLMs to enable temporal grounding capabilities in hour-long videos.

• We propose a vision-language model that recursively processes hour-long videos for effective and efficient hourlong video processing.

• We propose a progressive training strategy, where the model is first trained to identify events in short video segments, then progressively scales to hour-long videos, enabling it to effectively handle longer, more complex video sequences.

• Our model significantly outperforms previous state-ofthe-art approaches, surpassing specialized models and other Vision-Language Models (VLMs) on multiple datasets by a substantial margin. For instance, ReVisionLLM outperforms the previous best method [13] by 2.6% R1@.1 on the MAD [45] dataset. Moreover, our model can efficiently solve this task, processing, on average, 43% fewer frames compared to the the existing VLM model [16].

![](assets/images/5fc2f687482fa9572ea47a35cf8be2ea6c4b6bb878ce0b05e67ca14a41fbd4cc.jpg)  
Figure 2. Recursive Video Grounding. ReVisionLLM is a recursive vision-language model designed for localizing events in hourlong videos. Inspired by human search strategies, it first scans the entire video to identify relevant intermediate segments and then zooms in to precisely locate event boundaries. Here, we show one intermediate hierarchy for brevity.

## 2. Related Work

Vision Language Models. VLMs excel in tasks such as video summarization [61, 65], decision-making [11, 48], captioning and general question-answering [7, 46, 63], temporal localization and object trajectory detection [16, 17, 27, 43, 51, 60]. Existing VLMs integrate visual input by either training-free methods [55], full fine-tuning [59], or adapter fine-tuning [15, 16, 25, 30, 43]. However, most VLMs are limited to short videos and lack extensive temporal awareness. Our work extends VLMs to enable capabilities for temporal localization in hour-long videos. We contribute a novel adapter fine-tuning approach for hierarchical perception. Unlike prior models that aggregate frame features through spatial-temporal pooling [35] or align visual-text embeddings [26], we introduce a temporal feature reduction method to effectively scale training to hour-long videos.

Temporal Grounding VLMs. Most existing VLMs face challenges with temporal grounding. Recent models [16, 17, 27, 40, 41, 43, 47, 49, 51, 60] have been specifically developed to address this issue using specialized architectures and datasets. For example, VTimeLLM [16] proposes a temporal fine-tuning stage, TimeChat [43] integrates timestamps with visual features, LITA [17] introduces time tokens for temporal understanding, and Hawkeye [51] uses dense video captions for segment matching. However, these models are limited by their training context length, which confines them to short video segments and hinders their ability to handle the complex temporal relationships and redundancies found in longer videos. Although they can process arbitrary user queries, they generally underperform compared to traditional, non-LLM-based models. We overcome these limitations by introducing a new recursive vision-language model that enables long video processing.

Long Video Temporal Grounding. Recent advancements in hour-long video temporal grounding have leveraged datasets like MAD [45] and VidChapters7M [58]. These methods typically follow a two-stage approach: proposalfree methods [4, 31, 44, 62, 64] segment videos to predict candidate moments and rank them, while proposal-based methods [14, 39] generate proposal clips or anchors for grounding models. Approaches like CONE [14] and M-Guidance [4] utilize detection transformers [6] to enhance grounding, while RGNet [13] unifies clip retrieval [53] and grounding with an end-to-end transformer-based approach. Recently, SnAG [37] introduced late fusion techniques for more efficient processing. Prior models lack the instruction-following capability and rely on techniques that match video and text inputs. In contrast, our work integrates VLMs, which naturally allows our model to follow instructions and process textual user queries in the context of a long temporal grounding framework.

## 3. Method

## 3.1. Problem Overview

Given a long, untrimmed video input and an event defined by a text query, we aim to predict the precise temporal boundary of the event. Formally, as our inputs, we consider a long-range video sequence $V = [ v ^ { t } ] _ { t = 1 , . . . , T }$ comprised of $T$ RGB frames, where $v ^ { t }$ is the $t ^ { t h }$ frame. The event is defined by a query sentence $S$ with $N _ { s }$ words where the sentence corresponds to a target event’s start (s) and end (e) times, denoted as ${ \boldsymbol \tau } = ( s , e )$

## 3.2. The ReVisionLLM Model

We now describe our proposed ReVisionLLM model, which contains three high-level components: (1) a Multimodal Encoder, (2) a Hierarchical Adapter, and (3) a Large Language Model. We illustrate our approach in Fig. 3 and describe each component below.

Multimodal Encoder. We utilize an off-the-shelf video encoder (e.g., CLIP ViT-L/14 [56]) to extract features from an hour-long video. To reduce the input context length and capture the global properties of long video inputs, we extract global features (e.g., CLS token) for each frame, $f ^ { t } \in$ $\mathbb { R } ^ { D }$ , where $D$ represents the feature dimension. These CLS tokens form a set of temporal features, $F = [ f ^ { t } ] _ { t = 1 , . . . , T }$ for the whole video. We use the same CLIP ViT-L/14 text encoder to extract textual features $Q \in \mathbb { R } ^ { N _ { s } \times D }$ of the query sentence S.

Hierarchical Adapter. While LLMs excel at retrieving information from long text documents, performing retrieval from a large number of video frames remains challenging, limiting their effectiveness for visual grounding. To address this, we employ a recursive approach that processes hour-long videos at different temporal resolutions. Initially, the entire video is processed as a whole to identify segments of interest (Fig. 2-top). Next, these segments undergo finer analysis to pinpoint precise event boundaries (Fig. 2-bottom). At the bottom level, we retain the original temporal resolution, while higher levels use compressed representations to maintain manageable visual input lengths for the LLM. Specifically, our Hierarchical Adapter projects initial video features $\mathcal { F }$ into dense temporal features $\mathcal { D }$ for the bottom hierarchy (Fig. 3-right) and encodes them into downsampled sparse temporal features S for the upper hierarchies (Fig. 3-left).

To obtain both temporal features, we first partition $\mathcal { F }$ into sliding windows of length $\scriptstyle L _ { w } ,$ , producing video segments denoted as $C = [ C ^ { i } ] _ { i = 1 , . . . , | C | }$ . Each segment $C ^ { i }$ is defined by $C ^ { i } = [ f ^ { s _ { i } + t } ] _ { t = 1 , . . . , L _ { w } }$ , where $s _ { i }$ is the start index of each clip, and $C ^ { i } \in \mathbb { R } ^ { L _ { w } \times D }$ . Dense temporal features $\mathcal { D } ^ { i }$ are derived from each segment $C ^ { i }$ through a linear projection layer, $h _ { d } .$ such that $\bar { \mathcal { D } ^ { i } } = h _ { d } ( C ^ { i } ) \in \bar { \mathbb { R } ^ { L _ { w } \times D } }$

To create the sparse features, a two-step process is applied. First, a cross-attention layer (Eq. 1) uses segment feature $C ^ { i }$ as the query and the text feature $Q$ as the key, and outputs text-aligned segment feature $\tilde { C } ^ { i } \in \mathbb { R } ^ { L _ { w } \times \bar { D } }$ This cross-attention mechanism aligns the video segment with the text query, enhancing semantic correspondence between the modalities. Next, a self-attention layer (Eq. 2) takes the concatenation of a sparse feature $S ^ { i } \in \mathbb { R } ^ { D }$ and the text-aligned segment feature $\tilde { C } ^ { i }$ as input and condenses the segment into the sparse feature. This sparse feature is a compact, learnable representation similar to the CLS token in BERT [9].

$$
\tilde {C} ^ {i} = \text {Cross - Attention} (C ^ {i}, Q)
$$

$$
A = \text {Self - Attention} ([ \mathcal {S} ^ {i}; \tilde {C} ^ {i} ])\tag{1}
$$

(2)

For each video segment $C ^ { i }$ , we compute the sparse temporal feature as $S ^ { i } \ = \ A _ { 0 }$ , which condenses the segment (e.g., 2-minutes) into a compact embedding (e.g., 768-dimensional), substantially reducing the input context length. Collectively, this process yields the dense temporal features $\mathcal { D } = [ \mathcal { D } ^ { i } ] _ { i = 1 , . . . , | C | }$ and sparse temporal features $\boldsymbol { S } = [ S ^ { i } ] _ { i = 1 , \ldots , | C | }$ for all segments.

Input for the LLM. We construct a video input feature $[ I ^ { ( \ell ) } ] _ { \ell = 1 , \dots , L }$ with L hierarchies to capture different temporal scales. As illustrated in Figure 3, the lowest hierarchical feature, $I ^ { ( 1 ) }$ is set to the dense features \mathcal {D} while the higher levels $[ I ^ { ( \ell ) } ] _ { \ell = 2 , \dots , L }$ use sparse features \mathcal {S} , obtained from the Hierarchical Adapter. We combine this visual input with an instruction prompt for the LLM. We define the instruction as “ <video> when can we see the <event> happen-$i n g ? ^ { \prime \prime }$ . Here <event> is replaced by the textual event description. The word embedding layer of the LLM converts this prompt into token embeddings, $[ w _ { 1 } , w _ { 2 } , \ldots , w _ { M } ]$ . At <video> position, we insert the video features to create the final input prompt, $P ^ { ( \ell ) } = [ I ^ { ( \ell ) } , w _ { 1 } , \dots , w _ { M } ]$

![](assets/images/c505c099bad0b354a0411d0d2768b6131b8ca257725f83ca2a0b8bc2951fac44.jpg)  
Figure 3. The ReVisionLLM model. (Left) First, we detect segments (e.g., a few minutes) from an hour-long video using sparse temporal features produced by the Hierarchical Adapter. (Right) Then ReVisionLLM produces a precise temporal boundary using dense temporal features within the predicted segments. Note that the green box represents the same event boundary in both sub-figures, zooming in from left to right. The multimodal encoder is omitted for simplicity.

Large Language Model. To perform temporal grounding across multiple hierarchies, we utilize a pre-trained language model (e.g., Vicuna [8]) as the temporal grounding decoder, which predicts the event boundaries at each hierarchy. At hierarchy l, the LLM receives input, $P ^ { ( l ) }$ and outputs start and end times, $\tau ^ { ( l ) }$ , for that level in the form of: ‘From s to e.’. Here s and e denote the start and end frame indexes of the queried event. If the event is absent from the video segment, the model generates “Not Present.”. To progressively refine detection, the decoder processes segments features $P ^ { ( l ) }$ containing the predicted boundaries $\tau ^ { ( < l ) }$ from the previous hierarchy levels. At the initial level $( \tau ^ { ( 0 ) } )$ , no prior boundaries are provided, which requires the model to scan the entire video for initial boundary prediction. The proposed ReVisionLLM learns the likelihood of a target event’s start and end times $\tau ^ { ( l ) }$ conditioned on the hierarchical video representation $P ^ { ( l ) }$ through the following training objective:

$$
p (T ^ {(l)} | P ^ {(l)}) = \prod_ {k = 1} ^ {K} p (T _ {k} ^ {(\ell)} | T _ {<   k} ^ {(\ell)}, P ^ {(l)})\tag{3}
$$

Here, $T _ { k } ^ { ( \ell ) }$ denotes the $k ^ { t h }$ language token of the caption, and $T _ { < k } ^ { ( \ell ) }$ denotes all preceding tokens.

## 3.3. Training

Training a hierarchical video-language model for temporal grounding is challenging, particularly with videos of varying lengths and extensive non-relevant content. To tackle this, we employ a progressive training strategy (Fig. 4) where the model initially learns to identify events in short video segments before scaling up to hour-long videos. This approach allows the model to first focus on recognizing key events in shorter segments, then apply that understanding effectively to longer, more complex videos.

Stage 1: Training with Short Segments. Traditional VLMs are generally trained with only positive video segments. For example, VTimeLLM [16] trains the model to ground events on video segments that are guaranteed to contain the event (similar to the leftmost subplot in Figure 4). This approach, however, can lead to overconfidence in the model’s visual predictions [23, 38]. To address overconfidence—a challenge previously tackled in the text domain through contrastive examples [22], we introduce contrastive video segments where the target event is intentionally absent (Fig. 4-middle). In an hour-long video, many video segments are naturally present that are unrelated to the queried event. Including these segments in training helps the model better calibrate its confidence. At first, we use dense features to train the model to predict precise temporal boundaries (e.g., “From s to e.”) or indicate the absence of events (“Not Present.”). During this phase, we fine-tune the LLM component using LoRA.

After fine-tuning the LLM, we freeze its weights and proceed to fine-tune only the Hierarchical Adapter module to generate sparse temporal features. These sparse features, which are downsampled versions of the original visual data, are crucial for efficient long-video training (see 3.3). To simplify the training objective for these sparse inputs, we focus on identifying the presence of events rather than locating precise boundaries. We modify the input prompt as “<video> Does the <event> happen in the video? Answer yes or no.” In this phase, the model learns to respond “Yes.” for relevant segments or “No.” for irrelevant ones. Using this contrastive training strategy, we calibrate the model’s visual confidence and optimize the sparse features needed for effective hour-long video processing.

![](assets/images/f789d6c3328ed42b2c74ab34515d963b195576335a59043798f814722e6d9787.jpg)  
Figure 4. Progressive Training Method. Our model is trained progressively: first on short video segments and then on hour-long videos. (Left) In the first stage, the model learns to detect whether an event is present in the input video and, if so, predicts its precise start and endpoints. Sparse features help determine an event’s presence, while dense features additionally facilitate exact localization. (Right) In the second stage, we utilize the sparse features learned in Stage 1 to identify event segments within hour-long videos.

Stage 2: Training with Long Videos. In this stage (Fig. 4- right), we leverage sparse features to localize relevant segments within hour-long videos. These sparse features correspond to segments typically much longer (e.g., 3 minutes) than the actual target events, allowing the model to efficiently scan and identify broader regions of interest. We use the original prompt defined in Section 3.2 and keep the weights of the hierarchical perception module fixed, finetuning only the same LoRA module used in Stage 1.

## 3.4. Inference with Calibrated Confidence

Previous methods [14, 39] typically used the CLIP [56] similarity score between visual and textual embeddings to rank and select the top-k predictions. In contrast, we rank predictions based on the internal confidence of our LLM. Specifically, we calculate the entropy of the predicted probability distribution for each word generated by the LLM. We do this scoring on the predictions at the bottom hierarchy (l = 0) where we utilize the dense features D. For $i ^ { t h }$ prediction, let $p ( w | T _ { < k } , \mathcal { D } ^ { ( i ) } )$ denote the probability of the $k ^ { t h }$ word conditioned on prior words $T _ { < k }$ and corresponding segment feature $\mathcal { D } ^ { i }$ . We calculate the entropy $H _ { k } ^ { i }$ as:

$$
H _ {k} ^ {(i)} = - \sum_ {w} p (w | T _ {<   k}, \mathcal {D} ^ {(i)}) \log p (w | T _ {<   k}, \mathcal {D} ^ {(i)})
$$

Here, w represents each possible word in the vocabulary of our LLM. The overall uncertainty score is obtained by averaging the entropy values across all words in the generated sequence. To convert this into a confidence score (R<sup>i</sup>), we take the inverse of the mean entropy:

$$
R ^ {i} = \frac {1}{\frac {1}{K} \sum_ {k = 1} ^ {K} H _ {k} ^ {i}}
$$

where K is the total number of generated words. We calculate the confidence score for all the predictions $[ R ^ { i } ] _ { i = 1 , \dots , N _ { p } }$ , where $N _ { p }$ is the number of predicted boundaries. This confidence score allows us to rank and select the top-K predictions based on the LLM’s internal confidence about its outputs. More details are present in the Supplementary ??.

## 4. Experimental Setup

## 4.1. ReVisionLLM Baselines

Long video temporal grounding remains largely uncharted territory for vision-large language models, leaving a lack of established baselines for meaningful comparison. To address this, we present the following video-language baselines that we have adapted specifically for this task.

1. VTimeLLM [16]. A fully fine-tuned baseline that takes in hour-long video as input and produces temporal boundary localization.

2. VTimeLLM + CONE [14, 16]. This baseline is fully fine-tuned on shorter video segments as ours. We employ CONE’s fine-grained ranking method to select the top-k predictions across all segments. First, mean pooling aggregates the CLS features from CLIP [56] across all frames to create an average frame representation. The similarity score is then computed with a dot product between this averaged frame feature and the query text CLS feature from CLIP.

## 4.2. ReVisionLLM Variations

1. ReVisionLLM. Our default model begins by processing hour-long video segments at the top of the hierarchy, then moves down to shorter segments. In this setup, we train two LoRAs within the LLM: one for the bottom hierarchy and another for the higher levels. This approach is highly efficient for inference, as it reduces the number of input frames processed by the LLM.

<table><tr><td rowspan="2">Model</td><td colspan="7">MAD [45]</td><td colspan="5">VidChapters-7m [58]</td><td rowspan="2">Average↑</td></tr><tr><td>R1@.1</td><td>R5@.1</td><td>R1@.3</td><td>R5@.3</td><td>R1@.5</td><td>R5@.5</td><td>Avg.↑</td><td>R1@.3</td><td>R1@.5</td><td>R1@.7</td><td>R1@.9</td><td>Avg.↑</td></tr><tr><td>M-Guide [4]</td><td>9.3</td><td>18.9</td><td>4.6</td><td>13.1</td><td>2.2</td><td>7.4</td><td>9.3</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>CONE [14]</td><td>8.9</td><td>20.5</td><td>6.9</td><td>16.1</td><td>4.1</td><td>9.6</td><td>11.0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SOONet [39]</td><td>11.3</td><td>23.2</td><td>9.0</td><td>19.6</td><td>5.3</td><td>13.1</td><td>13.6</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SnAG [37]</td><td>10.3</td><td>24.4</td><td>8.5</td><td>20.6</td><td>5.5</td><td>13.7</td><td>13.8</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>RGNet [13]</td><td>12.4</td><td>25.1</td><td>9.5</td><td>18.7</td><td>5.6</td><td>10.9</td><td>13.7</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>BERT [9]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.6</td><td>0.3</td><td>0.1</td><td>0.0</td><td>0.3</td><td>0.3</td></tr><tr><td>VTimeLLM* [64]</td><td>1.4</td><td>3.1</td><td>1.3</td><td>2.5</td><td>0.6</td><td>1.1</td><td>1.7</td><td>10.6</td><td>4.1</td><td>1.6</td><td>0.2</td><td>4.1</td><td>2.9</td></tr><tr><td>CLIP [42]</td><td>6.6</td><td>15.1</td><td>3.1</td><td>9.9</td><td>1.5</td><td>5.4</td><td>6.9</td><td>10.7</td><td>5.2</td><td>2.3</td><td>0.5</td><td>4.7</td><td>5.8</td></tr><tr><td>M-DETR [24]</td><td>3.6</td><td>13.0</td><td>2.8</td><td>9.9</td><td>1.7</td><td>5.6</td><td>6.1</td><td>37.4</td><td>27.3</td><td>17.6</td><td>6.4</td><td>22.1</td><td>14.1</td></tr><tr><td> $Ours^†$ </td><td>17.3</td><td>31.4</td><td>12.7</td><td>23.5</td><td>6.7</td><td>13.1</td><td>17.5</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Ours</td><td>15.0</td><td>25.1</td><td>11.0</td><td>18.8</td><td>5.8</td><td>10.5</td><td>14.4</td><td>33.8</td><td>27.4</td><td>21.8</td><td>15.2</td><td>24.6</td><td>19.5</td></tr></table>

Table 1. Main Results on the MAD and VidChapters-7M Datasets. The best scores are highlighted in bold, while the second-best scores are underlined. ReVisionLLM demonstrates state-of-the-art performance across both datasets. This paper trains VTimeLLM on the datasets and uses CONE[14] ranking method. <sup>†</sup>ReVisionLLM-I variant processes more frames and achieves higher accuracy.

2. ReVisionLLM-U. This is a unified model across all hierarchies with shared weights. With significantly fewer trainable parameters than the default model, it is more efficient to train.

3. ReVisionLLM-I. In this variant, the LLM operates in reverse order, starting from the bottom of the hierarchy and progressively increasing the video length. Like the default model, it uses two LoRAs but requires processing all video frames, trading efficiency for improved performance with added computation.

4. ReVisionLLM-(U+I). This variant has a unified architecture like ReVisionLLM-U and starts processing from the bottom hierarchy like ReVisionLLM-I.

## 4.2.1 Datasets and Metrics

MAD Dataset [45]. This large-scale dataset comprises approximately 1,200 hours of full-length movies, featuring 384,000 natural language queries linked to specific moments within the videos. On average, each video is around 110 minutes long, while the moments are brief—just 4.1 seconds on average—making the moment-to-video ratio very low. This disparity poses a substantial challenge for accurate temporal grounding.

VidChapters-7M [58]. This is a large-scale, userannotated dataset with over 7 million chapters across 817,000 videos, with the longest 12 hours. Each video includes 2 to 30 chapters with durations from 1 second to 10 minutes, making it a challenging dataset for temporal localization due to its length and variety.

Evaluation Metrics. Following prior work [14, 45], we use Recall@k at IoU=θ (Rk@θ) as the primary grounding metric. This measures the proportion of test samples where at least one of the top-k predictions achieves an Intersection over Union (IoU) greater than θ with the ground truth. Additionally, to assess generalization to Text-to-Video retrieval, we use the standard Recall at Rank k (R@k) metric [12], which measures the percentage of ground truth video in the top-k retrieved ones.

Implementation Details. In our approach, we utilize the 7B version of Vicuna v1.5 [8] as the Large Language Model. Training is conducted on the using a total batch size of 128 across 8 A100 GPUs. For optimization, we use AdamW [34] with a cosine learning rate decay and an initial warm-up phase. During the adapter training stage, we run 1 epoch with a learning rate of $1 \times 1 0 ^ { - 3 }$ . In the subsequent hierarchical stage, we train for 5 epochs for MAD and 1 epoch for VidChapters-7M, with a learning rate of $1 \times 1 0 ^ { - 4 }$ . LoRA settings include parameters $r = 6 4$ and $a l p h a = 1 2 8$ . Please refer to the Supplementary ?? for additional implementation details.

## 5. Results

In this section, we present our performance against previous methods, detailed ablation studies, qualitative results, and generalizations of text-to-video retrieval tasks.

Main Results on the MAD Dataset [45]. ReVisionLLM sets a new state-of-the-art on the MAD dataset, outperforming prior models in temporal grounding (Tab. 1). It surpasses the previous best method, RGNet [13], by +2.6% in R1@.1 and +1.5% in R1@.3, achieving competitive scores across other metrics. Our ReVisionLLM-I variant outperforms RGNet by an even larger margin, +4.9% in R1@.1 and +6.3% in R1@.3. As the moments-to-video ratio is extremely low in this dataset, it requires fine-grained event understanding. Our model’s recursive architecture effectively narrows the search for relevant segments, handling these challenges well. While existing methods relying on heuristic CLIP [56] ranking struggle with numerous false detections (evident by low R1 score across all thresholds) in this dataset, ReVisionLLM reliance on LLM’s internal confidence reduces such errors.

Main Results on VidChapters-7M Dataset [58]. On the

VidChapters-7M dataset, ReVisionLLM sets a new stateof-the-art (Table 1), significantly outperforming the previous best model, M-DETR [24], particularly at higher IoU thresholds (+4.2% in R1@.7 and +8.8% in R1@.9) and showing strong results across other metrics. This precision at stricter thresholds demonstrates the superior ability of ReVisionLLM to localize events accurately. The dataset includes a diverse range of YouTube tutorial videos with userqueried steps, underscoring our model’s advancements in improving video content search for online platforms across short clips to extended videos up to 12 hours.

## 5.1. Ablation Studies

We present ablation studies on each module, model variants, video length, and the number of hierarchies by experimenting on the MAD dataset [45].

<table><tr><td>Modules</td><td>R1@.1↑</td><td>R5@.1↑</td><td>R1@.3↑</td><td>R5@.3↑</td></tr><tr><td>Baseline: VTimeLLM [16]</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>(+) CONE [14]</td><td>1.4</td><td>2.4</td><td>1.3</td><td>2.5</td></tr><tr><td>(+) Contrastive Segment</td><td>4.8</td><td>6.7</td><td>4.2</td><td>7.2</td></tr><tr><td>(+) Calibration (-) CONE</td><td>8.4</td><td>12.7</td><td>6.6</td><td>8.9</td></tr><tr><td>(+) Recursive Process*</td><td>15.0</td><td>25.1</td><td>11.0</td><td>18.8</td></tr></table>

Table 2. Cumulative Ablation on Proposed Modules. Each of our proposed modules contributes to a significant improvement in grounding capability, with the recursive process achieving the highest gains. <sup>∗</sup>Indicates the ReVisionLLM model.

Modules. Table 2 highlights the unique contributions of each module in our model for temporal grounding on long videos. We start with the baseline VTimeLLM [16] model, trained on the MAD dataset, which scores zero across all recall metrics due to uniform sampling of 100 frames from hours-long videos, causing a complete loss of temporal details. Next, we train and test the VTimeLLM model on shorter video segments and apply CONE’s ranking strategy [14] for final predictions. This single level of hierarchical processing yields modest improvements (e.g., R1@.1 of 1.4% and R5@.1 of 2.4%). The addition of the Contrastive Segments marks a notable improvement, raising R1@.1 to 4.8% and R5@.1 to 6.7%; by allowing the model to identify absent segments (e.g., “Not Present”), it narrows the temporal search space and enhances segment selection. Replacing CONE with our Grounding LLM’s confidence-based ranking further boosts results (e.g., R1@.1 of 8.4% and R5@.1 of 12.7%), as training on positive and contrastive segments in Stage 1 improves the LLM’s calibration, aligning confidence with prediction accuracy and enhancing ranking effectiveness. Finally, the recursive process achieves the highest performance gains, with R1@.1 of 15.0% and R5@.1 of 25.1%, by progressively refining the temporal focus. Each module contributes critically, with the Recursive Process module delivering the largest gains.

Model Variants. Table 3 compares ReVisionLLM variants to assess the effects of hierarchical processing order, percentage of video input frames for the LLM, and parameter sharing. The baseline (VTimeLLM+CONE) performs poorly (R1@.1 = 1.4%), despite processing 100% of video frames as input to the LLM, underscoring the limitations of non-recursive approaches. In contrast, our default Re-VisionLLM processes recursively from top to bottom, balancing accuracy and efficiency, achieving R1@.1 of 15.0% and R5@.1 of 25.1% with processing only 57% of input frames. The ReVisionLLM-U variant enhances training efficiency by sharing weights across all hierarchies, resulting in a slight performance reduction (R1@.1 = 14.4%, R5@.1 = 24.7%) while using fewer trainable parameters (363M vs. 159M). The ReVisionLLM-I variant reaches the highest accuracy (R1@.1 = 17.4%, R5@.1 = 31.4%) by processing in reverse hierarchical order (bottom to top), with more input frames. ReVisionLLM-(U+I) combines reverse processing with parameter sharing, balancing training efficiency, and high performance (R1@.1 = 16.7%, R5@.1 = 31.0%). Overall, the default ReVisionLLM offers strong accuracy with high frame efficiency, while ReVisionLLM-I provides peak accuracy with increased input frames to the LLM.

<table><tr><td>Model</td><td>Train Params↓</td><td>Input Frames↓</td><td>R1@.1↑</td><td>R5@.1↑</td></tr><tr><td>VTimeLLM+CONE</td><td>363M</td><td>100%</td><td>1.4</td><td>2.4</td></tr><tr><td>ReVisionLLM</td><td>363M</td><td>57%</td><td>15.0</td><td>25.1</td></tr><tr><td>ReVisionLLM-U</td><td>159M</td><td>58%</td><td>14.4</td><td>24.7</td></tr><tr><td>ReVisionLLM-(U+I)</td><td>159M</td><td>100%</td><td>16.7</td><td>31.0</td></tr><tr><td>ReVisionLLM-I</td><td>363M</td><td>100%</td><td>17.4</td><td>31.4</td></tr></table>

Table 3. Ablation on Model Variants. We perform well with fewer frames and trainable parameters than the baseline, which struggles to solve the task effectively. Processing additional frames and more parameters further improves our performance.

![](assets/images/34290c641e877066d5f7d7703dd880afcfaeb623a02b943c197a9ccd4465b5e0.jpg)  
Figure 5. Ablation on Video Length. Our recursive approach maintains strong performance even with videos up to 10 hours long, while the baseline method fails entirely in these cases.

Video Length. Fig. 5 demonstrates ReVisionLLM’s robustness in handling long videos. We extend the videos by repeating them multiple times to create longer sequences, ensuring that the ground truth moment appears only once within the extended video. While the method without recursion fails for 10-hour videos, recursive video processing maintains strong performance. The slight decrease in performance for longer videos reflects the inherent challenges

Event 1: At a crowded desk and manuscripts, a man with dark curly hair and glass answers phone.

![](assets/images/b95b8bacbe067a7cc4be3de83acfe7d119b086dffca3f9bfac4d92310bfd2083.jpg)

![](assets/images/a45be63d9d22bc981d8bac1a4095ae41c60a41fcce421d71c7ffaad9bc63aa5f.jpg)  
Figure 6. Qualitative results on MAD. ReVisionLLM accurately locates precise event boundaries that involve intricate actions (top) and complex visual details (bottom) within hour-long movies. In contrast, our VLM baseline fails entirely to capture these events.

of temporal grounding in extended content rather than a limitation of the model. Beyond approximately five hours, performance stabilizes, as the increasing diversity of scenes has minimal impact on event localization.

Number of Hierarchies. Table 4 shows the impact of the number of hierarchies. Without hierarchy, treating the entire video as a single unit prevents the model from capturing event boundaries, highlighting the limitations of current non-recursive VLMs in long video grounding. With one hierarchy, we segment the video and aggregate predictions with calibrated confidence, showing some improvement but still struggling with high false positives. With 2 and 3 hierarchies, we progressively filter out irrelevant regions and revise predictions recursively, resulting in higher accuracy.

<table><tr><td>Hierarchies</td><td>R1@.1↑</td><td>R5@.1↑</td><td>R1@.3↑</td><td>R5@.3↑</td></tr><tr><td>0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>1</td><td>8.4</td><td>12.7</td><td>6.6</td><td>8.9</td></tr><tr><td>2</td><td>11.9</td><td>17.5</td><td>8.7</td><td>13.2</td></tr><tr><td>3</td><td>15.0</td><td>25.1</td><td>11.0</td><td>18.8</td></tr></table>

Table 4. Ablation on number of Hierarchies. The model’s performance improves with the number of hierarchies in the recursive structure, becoming more effective with each additional level. Without this hierarchical approach, the model fails on the task.

## 5.2. Qualitative Results on MAD Dataset

In Figure 6, we demonstrate two examples of events accurately localized by ReVisionLLM. In the first example, frequent scenes of office work closely resemble the queried event, but ReVisionLLM successfully identifies the specific instance where a person with distinct attributes answers a phone—a task requiring detailed comprehension of both appearance and action. The second example tests the model’s ability to identify a complex visual description spanning 5.8 seconds within a 2-hour movie, highlighting its effectiveness in locating subtle differences within visually similar footage. These examples validate ReVisionLLM’s recursive architecture for precise event localization. More qualitative results are included in the Supplementary ??.

## 5.3. Generalization: Text-to-Video Retrieval

Text-to-video retrieval is the task of identifying the video corresponding to a textual event description from a large set of different videos. We solve this task with our ReVisionLLM by concatenating all the videos into a single hourlong video and applying our model to locate the index of the predicted video. It outperforms previous state-of-the-art models on the MSRVTT [53] dataset, with improvements of +2.2% in R@5 and +0.6% in R@10, while achieving competitive R@1 performance. These results demonstrate Re-VisionLLM’s understanding of video-text correspondence, positioning it as a strong model for general multi-modal retrieval tasks, including large-scale video searches. More details are provided in the Supplementary ??.

<table><tr><td>Method</td><td>R@1 ↑</td><td>R@5 ↑</td><td>R@10 ↑</td></tr><tr><td>X-Pool [12]</td><td>46.9</td><td>72.8</td><td>82.2</td></tr><tr><td>DiffusionRet [20]</td><td>49.0</td><td>75.2</td><td>82.7</td></tr><tr><td>UATVR [10]</td><td>47.5</td><td>73.9</td><td>83.5</td></tr><tr><td>TEFAL [18]</td><td>49.4</td><td>75.9</td><td>83.9</td></tr><tr><td>CLIP-ViP [57]</td><td>50.1</td><td>74.8</td><td>84.6</td></tr><tr><td>T-MASS [50]</td><td>50.2</td><td>75.3</td><td>85.1</td></tr><tr><td>Ours</td><td>49.1</td><td>77.5</td><td>85.7</td></tr></table>

Table 5. ReVisionLLM’s Generalization. Our model generalizes well to the Text-to-Video retrieval task and performs competitively with state-of-the-art models on the MSRVTT [53] dataset.

## 6. Conclusion and Future Work

We introduce ReVisionLLM, the first VLM specifically designed with a recursive structure for temporal event grounding in hour-long videos. Its recursive architecture effectively can locate events within extensive videos and establishes a new state-of-the-art, outperforming specialized models. Future work could focus on integrating audio for better event comprehension and expanding the capabilities to handle even longer videos spanning multiple days.

## References

[1] Greg kamradt on x: ”pressure testing gpt-4-128k with long context recall 128k tokens of context is awesome - but what’s performance like? i wanted to find out so i did a “needle in a haystack” analysis some expected (and unexpected) results here’s what i found: Findings: \* gpt-4’s recall https://t.co/nhmokmfhw5” / x. https://twitter.com/GregKamradt/status/ 1722386725635580292 ? s = 20, . (Accessed on 11/14/2024). 1

[2] Greg kamradt on x: ”claude 2.1 (200k tokens) - pressure testing long context recall we all love increasing context lengths - but what’s performance like? anthropic reached out with early access to claude 2.1 so i repeated the “needle in a haystack” analysis i did on gpt-4 here’s what i found: https://t.co/b36knjtjme” / x. https://twitter.com/ GregKamradt/status/1727018183608193393, . (Accessed on 11/14/2024).

[3] gkamradt/llmtest needleinahaystack: Doing simple retrieval from llm models at various context lengths to measure accuracy. https://github.com/gkamradt/LLMTest\_ NeedleInAHaystack. (Accessed on 11/14/2024). 1

[4] Wayner Barrios, Mattia Soldan, Alberto Mario Ceballos-Arroyo, Fabian Caba Heilbron, and Bernard Ghanem. Localizing moments in long video via multimodal guidance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13667–13678, 2023. 3, 6

[5] Patrick Bourke, Steven Brown, Elton Ngan, and Mario Liotti. Functional brain organization of preparatory attentional control in visual search. Brain research, 1530:32–43, 2013. 2

[6] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In European conference on computer vision, pages 213–229. Springer, 2020. 3

[7] Guo Chen, Yin-Dong Zheng, Jiahao Wang, Jilan Xu, Yifei Huang, Junting Pan, Yi Wang, Yali Wang, Yu Qiao, Tong Lu, et al. Videollm: Modeling video sequence with large language models. arXiv preprint arXiv:2305.13292, 2023. 2

[8] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality, 2023. 4, 6

[9] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding, 2019. 3, 6

[10] Bo Fang, Wenhao Wu, Chang Liu, Yu Zhou, Yuxin Song, Weiping Wang, Xiangbo Shu, Xiangyang Ji, and Jingdong Wang. Uatvr: Uncertainty-adaptive text-video retrieval, 2023. 8

[11] Difei Gao, Lei Ji, Luowei Zhou, Kevin Qinghong Lin, Joya Chen, Zihan Fan, and Mike Zheng Shou. Assistgpt: A general multi-modal assistant that can plan, execute, inspect, and learn. arXiv preprint arXiv:2306.08640, 2023. 2

[12] Satya Krishna Gorti, Noel Vouitsis, Junwei Ma, Keyvan Golestan, Maksims Volkovs, Animesh Garg, and Guangwei

Yu. X-pool: Cross-modal language-video attention for textvideo retrieval, 2022. 6, 8

[13] Tanveer Hannan, Md Mohaiminul Islam, Thomas Seidl, and Gedas Bertasius. Rgnet: A unified retrieval and grounding network for long videos. arXiv preprint arXiv:2312.06729, 2023. 1, 2, 3, 6

[14] Zhijian Hou, Wanjun Zhong, Lei Ji, Difei Gao, Kun Yan, Wing-Kwong Chan, Chong-Wah Ngo, Zheng Shou, and Nan Duan. Cone: An efficient coarse-to-fine alignment framework for long video temporal grounding. arXiv preprint arXiv:2209.10918, 2022. 1, 3, 5, 6, 7

[15] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 2

[16] Bin Huang, Xin Wang, Hong Chen, Zihan Song, and Wenwu Zhu. Vtimellm: Empower llm to grasp video moments. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14271–14280, 2024. 1, 2, 4, 5, 7

[17] De-An Huang, Shijia Liao, Subhashree Radhakrishnan, Hongxu Yin, Pavlo Molchanov, Zhiding Yu, and Jan Kautz. Lita: Language instructed temporal-localization assistant. arXiv preprint arXiv:2403.19046, 2024. 1, 2

[18] Sarah Ibrahimi, Xiaohang Sun, Pichao Wang, Amanmeet Garg, Ashutosh Sanan, and Mohamed Omar. Audioenhanced text-to-video retrieval using text-conditioned feature alignment, 2023. 8

[19] Maor Ivgi, Uri Shaham, and Jonathan Berant. Efficient longtext understanding with short-text models. Transactions of the Association for Computational Linguistics, 11:284–299, 2023. 1

[20] Peng Jin, Hao Li, Zesen Cheng, Kehan Li, Xiangyang Ji, Chang Liu, Li Yuan, and Jie Chen. Diffusionret: Generative text-video retrieval with diffusion model, 2023. 8

[21] Peng Jin, Ryuichi Takanobu, Wancai Zhang, Xiaochun Cao, and Li Yuan. Chat-univi: Unified visual representation empowers large language models with image and video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13700– 13710, 2024. 1

[22] Sanyam Kapoor, Nate Gruver, Manley Roberts, Katherine Collins, Arka Pal, Umang Bhatt, Adrian Weller, Samuel Dooley, Micah Goldblum, and Andrew Gordon Wilson. Large language models must be taught to know what they don’t know. arXiv preprint arXiv:2406.08391, 2024. 4

[23] Vasily Kostumov, Bulat Nutfullin, Oleg Pilipenko, and Eugene Ilyushin. Uncertainty-aware evaluation for visionlanguage models. arXiv preprint arXiv:2402.14418, 2024. 1, 4

[24] Jie Lei, Tamara L Berg, and Mohit Bansal. Detecting moments and highlights in videos via natural language queries. Advances in Neural Information Processing Systems, 34: 11846–11858, 2021. 1, 6, 7

[25] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In In-

ternational conference on machine learning, pages 19730– 19742. PMLR, 2023. 2

[26] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023. 2

[27] Zhaowei Li, Qi Xu, Dong Zhang, Hang Song, Yiqing Cai, Qi Qi, Ran Zhou, Junting Pan, Zefeng Li, Vu Tu, et al. Groundinggpt: Language enhanced multi-modal grounding model. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6657–6678, 2024. 2

[28] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023. 1

[29] Kevin Qinghong Lin, Pengchuan Zhang, Joya Chen, Shraman Pramanick, Difei Gao, Alex Jinpeng Wang, Rui Yan, and Mike Zheng Shou. Univtg: Towards unified videolanguage temporal grounding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2794–2804, 2023. 1

[30] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023. 2

[31] Naiyuan Liu, Xiaohan Wang, Xiaobo Li, Yi Yang, and Yueting Zhuang. Reler@ zju-alibaba submission to the ego4d natural language queries challenge 2022. arXiv preprint arXiv:2207.00383, 2022. 3

[32] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. corr abs/2307.03172 (2023). arXiv preprint arXiv:2307.03172, 10, 2023. 1

[33] Ye Liu, Siyuan Li, Yang Wu, Chang-Wen Chen, Ying Shan, and Xiaohu Qie. Umt: Unified multi-modal transformers for joint video moment retrieval and highlight detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3042–3051, 2022. 1

[34] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization, 2019. 6

[35] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424, 2023. 2

[36] WonJun Moon, Sangeek Hyun, SangUk Park, Dongchan Park, and Jae-Pil Heo. Query-dependent video representation for moment retrieval and highlight detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23023–23033, 2023. 1

[37] Fangzhou Mu, Sicheng Mo, and Yin Li. Snag: Scalable and accurate video grounding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18930–18940, 2024. 1, 3, 6

[38] Changdae Oh, Mijoo Kim, Hyesu Lim, Junhyeok Park, Euiseog Jeong, Zhi-Qi Cheng, and Kyungwoo Song. Towards calibrated robust fine-tuning of vision-language models. arXiv preprint arXiv:2311.01723, 2023. 1, 4

[39] Yulin Pan, Xiangteng He, Biao Gong, Yiliang Lv, Yujun Shen, Yuxin Peng, and Deli Zhao. Scanning only once: An end-to-end framework for fast temporal grounding in long videos. arXiv preprint arXiv:2303.08345, 2023. 1, 3, 5, 6

[40] Long Qian, Juncheng Li, Yu Wu, Yaobo Ye, Hao Fei, Tat-Seng Chua, Yueting Zhuang, and Siliang Tang. Momentor: Advancing video large language model with fine-grained temporal reasoning, 2024. 1, 2

[41] Mengxue Qu, Xiaodong Chen, Wu Liu, Alicia Li, and Yao Zhao. Chatvtg: Video temporal grounding via chat with video dialogue large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1847–1856, 2024. 2

[42] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. pages 8748–8763, 2021. 6

[43] Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. Timechat: A time-sensitive multimodal large language model for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14313–14323, 2024. 1, 2

[44] Mattia Soldan, Mengmeng Xu, Sisi Qu, Jesper Tegner, and Bernard Ghanem. Vlg-net: Video-language graph matching network for video grounding. In ICCV, pages 3224–3234, 2021. 3

[45] Mattia Soldan, Alejandro Pardo, Juan Leon Alc´ azar, Fabian´ Caba, Chen Zhao, Silvio Giancola, and Bernard Ghanem. Mad: A scalable dataset for language grounding in videos from movie audio descriptions. In CVPR, pages 5026–5035, 2022. 2, 3, 6, 7

[46] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, et al. Moviechat: From dense token to sparse memory for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18221–18232, 2024. 2

[47] Yunlong Tang, Daiki Shimada, Jing Bi, and Chenliang Xu. Avicuna: Audio-visual llm with interleaver and contextboundary alignment for temporal referential dialogue. arXiv preprint arXiv:2403.16276, 2024. 2

[48] Junke Wang, Dongdong Chen, Chong Luo, Xiyang Dai, Lu Yuan, Zuxuan Wu, and Yu-Gang Jiang. Chatvideo: A tracklet-centric multimodal and versatile video understanding system. arXiv preprint arXiv:2304.14407, 2023. 2

[49] Junke Wang, Dongdong Chen, Chong Luo, Bo He, Lu Yuan, Zuxuan Wu, and Yu-Gang Jiang. Omnivid: A generative framework for universal video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18209–18220, 2024. 2

[50] Jiamian Wang, Guohao Sun, Pichao Wang, Dongfang Liu, Sohail Dianat, Majid Rabbani, Raghuveer Rao, and Zhiqiang Tao. Text is mass: Modeling as stochastic embedding for text-video retrieval, 2024. 8

[51] Yueqian Wang, Xiaojun Meng, Jianxin Liang, Yuxuan Wang, Qun Liu, and Dongyan Zhao. Hawkeye: Training video-

text llms for grounding text in videos. arXiv preprint arXiv:2403.10228, 2024. 2

[52] Jeremy M Wolfe and Todd S Horowitz. Five factors that guide attention in visual search. Nature human behaviour, 1 (3):0058, 2017. 2

[53] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5288–5296, 2016. 3, 8

[54] Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. Pllava: Parameter-free llava extension from images to videos for video dense captioning. arXiv preprint arXiv:2404.16994, 2024. 1

[55] Mingze Xu, Mingfei Gao, Zhe Gan, Hong-You Chen, Zhengfeng Lai, Haiming Gang, Kai Kang, and Afshin Dehghan. Slowfast-llava: A strong training-free baseline for video large language models. arXiv preprint arXiv:2407.15841, 2024. 2

[56] Hongwei Xue, Yuchong Sun, Bei Liu, Jianlong Fu, Ruihua Song, Houqiang Li, and Jiebo Luo. Clip-vip: Adapting pretrained image-text model to video-language representation alignment. ICLR, 2023. 3, 5, 6

[57] Hongwei Xue, Yuchong Sun, Bei Liu, Jianlong Fu, Ruihua Song, Houqiang Li, and Jiebo Luo. Clip-vip: Adapting pretrained image-text model to video-language representation alignment, 2023. 8

[58] Antoine Yang, Arsha Nagrani, Ivan Laptev, Josef Sivic, and Cordelia Schmid. Vidchapters-7m: Video chapters at scale, 2023. 3, 6

[59] Antoine Yang, Arsha Nagrani, Paul Hongsuck Seo, Antoine Miech, Jordi Pont-Tuset, Ivan Laptev, Josef Sivic, and Cordelia Schmid. Vid2seq: Large-scale pretraining of a visual language model for dense video captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10714–10726, 2023. 2

[60] Shoubin Yu, Jaemin Cho, Prateek Yadav, and Mohit Bansal. Self-chained image-language model for video localization and question answering. Advances in Neural Information Processing Systems, 36, 2024. 2

[61] Ce Zhang, Taixi Lu, Md Mohaiminul Islam, Ziyang Wang, Shoubin Yu, Mohit Bansal, and Gedas Bertasius. A simple llm framework for long-range video question-answering, 2024. 2

[62] Hao Zhang, Aixin Sun, Wei Jing, and Joey Tianyi Zhou. Span-based localizing network for natural language video localization. arXiv preprint arXiv:2004.13931, 2020. 3

[63] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023. 1, 2

[64] Songyang Zhang, Houwen Peng, Jianlong Fu, and Jiebo Luo. Learning 2d temporal adjacent networks for moment localization with natural language. In AAAI, pages 12870–12877, 2020. 3, 6

[65] Yue Zhao, Ishan Misra, Philipp Krahenb ¨ uhl, and Rohit¨ Girdhar. Learning video representations from large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6586– 6597, 2023. 2