### Abstact

The recent surge in interest surrounding text-to-video diffusion models highlights their capability to generate videos with both consistency and diversity. Current methods focus on leveraging large-scale datasets to align with training videos, while certain approaches explore the potential of zero-shot generation. Few-shot generative models adapt text-to-image model with temporal layers. They can capture the temporal motion and appearance with acceptable computational resources. However, the prevalent few-shot based approaches, which employ a singular prompt for training videos, typically result in inadequate control over complex backgrounds and multiple objects. To overcome this limitation, we introduce a novel component: the dual cross-attention layer. This component leverages CLIP for image feature extraction. The image feature and text feature will be processed independently in dual cross-attention layer, aiming to achieve image reference and enrich the training videos with additional information. This dual cross-attention mechanism empowers the diffusion model to learn both image and text information effectively, facilitating the generation of higher quality videos. Furthermore, we propose an innovative sampling method to enhance temporal consistency and stability of generative videos. Compared to other text-to-video generative models, our framework demonstrates superior efficiency in generating high quality videos with diverse styles. The code is available at https://github.com/FatLong666/MAIM.



![fig1](./assets/fig1.png)

<img src="./assets/volcano_7.gif" alt="volcano_7" style="zoom:50%;" />

**Prompt: A volcano erupts with huge smoke, overlook angle.**

<img src="./assets/volcano_13-1710816112876-10.gif" alt="volcano_13" style="zoom:50%;" />

**Prompt: A volcano erupts with smoke and lava, dark and red.**

<img src="./assets/Guan_Yu_rides_a_red_horse_ours.gif" alt="Guan_Yu_rides_a_red_horse_ours" style="zoom:50%;" />

**Prompt: Guan Yu rides a red horse.**

<img src="./assets/flower_12.gif" alt="flower_12" style="zoom:50%;" />

**Prompt: A red flower in bloom, Van Gogh style, oil painting.**

<img src="./assets/flower_5.gif" alt="flower_5" style="zoom:50%;" />

**Prompt: A pink flower in bloom, at sunrise.**

<img src="./assets/helicopter_12.gif" alt="helicopter_12" style="zoom:50%;" />

**Prompt: Two helicopters fly above the sea.**

<img src="./assets/helicopter_17.gif" alt="helicopter_17" style="zoom:50%;" />

**Prompt: Four helicopters fly on the city.**