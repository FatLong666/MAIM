## **Make An Image Move: Few-shot based Video Generation Guided by CLIP**

### Dependencies and Installation

- Ubuntu > 18.04

- CUDA=11.3

- Others:

  ```
  # clone the repo
  https://github.com/FatLong666/MAIM.git
  
  # create virtual environment
  conda create -n MAIM python=3.8
  conda activate MAIM
  
  # install packages
  pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
  pip install -r requirements.txt
  pip install xformers==0.0.13
  ```
[ICPR文章.pdf](https://github.com/user-attachments/files/17055596/ICPR.pdf)

  



![fig1](./assets/fig1.png)

<img src="./assets/volcano_7.gif" alt="volcano_7" style="zoom:50%;" />

**Prompt: A volcano erupts with huge smoke, overlook angle.**

<img src="./assets/volcano_13-1710816112876-10.gif" alt="volcano_13" style="zoom:50%;" />

**Prompt: A volcano erupts with smoke and lava, dark and red.**

<img src="./assets/Guan_Yu_rides_a_red_horse_ours.gif" alt="Guan_Yu_rides_a_red_horse_ours" style="zoom:50%;" />

**Prompt: Guan Yu rides a red horse.**

<img src="./assets/flower_12.gif" alt="flower_12" style="zoom:50%;" />

**Prompt: A red flower in bloom, Van Gogh style, oil painting.**

<img src="./assets/helicopter_12.gif" alt="helicopter_12" style="zoom:50%;" />

**Prompt: Two helicopters fly above the sea.**

<img src="./assets/helicopter_17.gif" alt="helicopter_17" style="zoom:50%;" />

**Prompt: Four helicopters fly on the city.**
