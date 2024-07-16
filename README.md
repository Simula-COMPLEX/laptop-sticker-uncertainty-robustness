# Assessing the Uncertainty and Robustness of Object Detection Models for Detecting Stickers on Laptops

> This repository contains the replication package for the paper Assessing the Uncertainty and Robustness of Object Detection Models for Detecting Stickers on Laptops.
> To facilitate reviewing our proposed approach, reviewers please refer to the corresponding data in this repository.
> This repository also contains the source code of our experiments.<br/>


## Abstract

Refurbishing laptops extends their lives while contributing to reducing electronic waste, which promotes building a sustainable future. To this end, the Danish Technological Institute (DTI) focuses on the research and development of several applications, including laptop refurbishing. This has several steps, including cleaning, which involves identifying and removing stickers from laptop surfaces.  DTI trained six sticker detection models (SDMs) based on open-source object detection models to identify such stickers precisely so these stickers can be removed automatically. However, given the diversity in types of stickers (e.g., shapes, colors, locations), identification of the stickers is highly uncertain, thereby requiring explicit quantification of uncertainty associated with the identified stickers. Such uncertainty quantification can help reduce risks in removing stickers, which, for example, could otherwise result in damaging laptop surfaces. For uncertainty quantification, we adopted the Monte Carlo Dropout method to evaluate the six SDMs from DTI using three datasets: the original image dataset from DTI and two datasets generated with vision language models, i.e., DALL-E-3 and Stable Diffusion-3. In addition, we presented novel robustness metrics concerning detection accuracy and uncertainty to assess the robustness of the SDMs based on adversarial datasets generated from the three datasets using a dense adversary method. Our evaluation results show that different SDMs perform differently regarding different metrics. Based on the results, we provide SDM selection guidelines and lessons learned from various perspectives.


<div align=center><img src="https://github.com/Simula-COMPLEX/EpiTESTER/blob/main/assets/overview.png" width="960" /></div>
