# BCC-LabNet: Low-Light Image Enhancement (Lab & Retinex synergy)

> This document extracts **key figures and descriptions** from the manuscript to provide a quick overview of the method and results.

**Highlights:** We perform three-branch decoupling (luminance/contrast/chroma) in CIE-Lab space with **LSA / Cross-Attention**, plus **LabRetinex** physics-guided refinement for joint enhancement and denoising. The method achieves leading PSNR/SSIM on LOLv1, LOLv2-Real, VisDrone, LSRW-Huawei, and strong no-reference scores (NIQE/UICM) and downstream DarkFace detection performance.



![Fig. 1. BCC-LabNet overview: Lab-space decoupling with LSA/Cross-Attention and LabRetinex refinement.](assets/fig_01.png)

*Fig. 1. BCC-LabNet overview: Lab-space decoupling with LSA/Cross-Attention and LabRetinex refinement.*

![Fig. 2. Visual comparison on VisDrone: LYT-Net, CIDNet, BCC-LabNet, and GT.](assets/fig_02.png)

*Fig. 2. Visual comparison on VisDrone: LYT-Net, CIDNet, BCC-LabNet, and GT.*

![Fig. 3. Interpretability analysis: intermediate visualizations (ΔL, attention, chroma residuals) and parametric reconstruction.](assets/fig_03.png)

*Fig. 3. Interpretability analysis: intermediate visualizations (ΔL, attention, chroma residuals) and parametric reconstruction.*

![Fig. 4. Retinex decomposition states: illumination I and reflectance R visualizations.](assets/fig_04.png)

*Fig. 4. Retinex decomposition states: illumination I and reflectance R visualizations.*

![Fig. 5. Downstream object detection on DarkFace (YOLOv11s): comparisons across enhancement methods.](assets/fig_05.png)

*Fig. 5. Downstream object detection on DarkFace (YOLOv11s): comparisons across enhancement methods.*

![Fig. 6. Illustration from the document](assets/fig_06.png)

*Fig. 6. Illustration from the document*
