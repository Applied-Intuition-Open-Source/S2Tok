# S2Tok model weights: license and terms

The pretrained S2Tok weights (`s2tok_252.pt`, on [Hugging Face](https://huggingface.co/AppliedIntuitionResearch/S2Tok)) are licensed under the
[Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/),
the same license as the code; the full text is in [LICENSE](LICENSE).
**Non-commercial use only.**

The non-commercial restriction is also required by the weights' upstream dependencies:

- they are fine-tuned from the released [ZipSplat](https://github.com/cvg/ZipSplat) checkpoint
  (CC BY-NC 4.0),
- which is initialized from [DA3-Giant](https://huggingface.co/depth-anything/DA3-GIANT)
  (CC BY-NC 4.0),
- depth labels for training were produced with
  [DA3-Nested-Giant-Large](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE) (CC BY-NC 4.0),
- and they are trained on [DL3DV-10K](https://github.com/DL3DV-10K/Dataset) (non-commercial terms
  of use) and [RealEstate10K](https://google.github.io/realestate10k/).

Use of the weights must also comply with the terms of these upstream models and datasets.

**Attribution.** When sharing the weights or anything derived from them, credit "S2Tok, The S2Tok
Authors", link to this repository and to the license, and indicate whether changes were made.

**No warranty.** The weights are provided "as is", without warranties or conditions of any kind;
see Section 5 of the license.
