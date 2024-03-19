### Abstact

The recent surge in interest surrounding text-to-video diffusion models highlights their capability to generate videos with both consistency and diversity. Current methods focus on leveraging large-scale datasets to align with training videos, while certain approaches explore the potential of zero-shot generation. Few-shot generative models adapt text-to-image model with temporal layers. They can capture the temporal motion and appearance with acceptable computational resources. However, the prevalent few-shot based approaches, which employ a singular prompt for training videos, typically result in inadequate control over complex backgrounds and multiple objects. To overcome this limitation, we introduce a novel component: the dual cross-attention layer. This component leverages CLIP for image feature extraction. The image feature and text feature will be processed independently in dual cross-attention layer, aiming to achieve image reference and enrich the training videos with additional information. This dual cross-attention mechanism empowers the diffusion model to learn both image and text information effectively, facilitating the generation of higher quality videos. Furthermore, we propose an innovative sampling method to enhance temporal consistency and stability of generative videos. Compared to other text-to-video generative models, our framework demonstrates superior efficiency in generating high quality videos with diverse styles. The code is available at https://github.com/FatLong666/MAIM.



![fig1](./assets/fig1.png)

![volcano_7](./assets/volcano_7.gif)

![volcano_13](./assets/volcano_13-1710816112876-10.gif)

![volcano_13](./assets/1.gif)

![flower_12](./assets/flower_12.gif)

![flower_5](./assets/flower_5.gif)

![helicopter_12](./assets/helicopter_12.gif)

![helicopter_17](./assets/helicopter_17.gif)