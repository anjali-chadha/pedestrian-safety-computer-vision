## [MSR-net:Low-light Image Enhancement Using Deep Convolutional Network](https://arxiv.org/abs/1711.02488)

1. Problem: Images captured in low light have very low contrast.
2. Enhancing image using CNN model and **Retinex theory**.
3. Traditional approaches for this problem:
    * Histogram based methods
    * Retinex Based methods
4. Solution: Establish a relationship between muti-scale Retinex and feedforward convolutional neural network.
5. Regard it as a supervised learning problem
      * Input: Dark images
      * Output: Light images
6. Retinex Theory:
      1. Combination of word retina and cortex.
      2. Theory that full color perception with color constancy involved all levels of visual processing
      from retina to the visual cortex.
7. Histogram based methods - most inuitive/common techniques for lightening the dark images.
      * Histogram equalization (HE) - common technique - makes histogram of the whole image as
      balanced as possiblem
      * Gamma Correction - Enhances the contrast and brightness by explanding the dark regions
      and suppressing the bright ones in the mean time.
      * **Drawback** - They treat each pixel individually without the dependence of their neighbourhoods.
      * **Solutions Proposed:** Use different regularization terms on the histogram
8. Retinex based methods: 
      * Retinex theory explains the color perception property of human vision system.
      * Assumption of the theory - image can be decomposed into reflection and illumination.
            
      * SSR (Single Scale Retinex) - based on center/surround retinex
      * MSR (Multi Scale Retinex) - can be computed as weighted sum of several different SSR outputs.
      * MSR outputs looks unnatural.
      * Modified MSR that applies color restoration finction and eliminate the color distortions that
      appear in MSR output.
      * One technique estimates the illumination of each pixel by finding the maximum value in R,G, B channel.
9. CNNs for Image Enhancement
      * MSR is equivalent to CNN with different Gaussian convolutional kernels 
      * Three main components in the model
         * Multi scale logarithmic transforamtion
         * Difference of convolution
         * Color Restoration
      * Main difference between MSR-net and MSR       
         * MSR - Parameters such as variance and other constants depend on artificial settings.
         * MSR-net - Parameters in the model are learned from training.

