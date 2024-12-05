# People

nmrlineshapeanalyser is written by [Biffo Abdulkadir Olatunbosun](https://scholar.google.com/citations?user=TievH90AAAAJ&hl=en).

# Active Maintainers

The current maintainer is [Biffo Abdulkadir Olatunbosun](https://github.com/BiffoQ).

# Inspiration

On a cold morning while surviving the usual Dutch weather, I had a conversation with my PhD supervisor [Pedro Braga Groszewicz](https://scholar.google.com/citations?user=05NlC3gAAAAJ&hl=en) regarding how to deconvolute Lorentzian and Gaussian areas from a quite complicated 1D 17O MAS ss-NMR spectrum. He explained the so-called Voigt and Pseudo-Voigt concepts and how they could be helpful in this deconvolution. He mentioned [MestreNova](https://mestrelab.com/main-product/mnova) software, and indeed this software offers a solution to my problem. However, I was not just interested in the solution but also the black box.

So I embarked on this journey of unraveling the black box and did not disengage until I found out how it was done. This journey led to this package.

I would like to thank the [NMRglue](https://github.com/jjhelmus/nmrglue) creator for making it possible to work directly with [Bruker's](https://www.bruker.com/en.html) folder. Also, a big thanks to the creator(s) of [LMFit's](https://lmfit.github.io/lmfit-py/builtin_models.html) built-in models -- the content was pivotal in understanding the concept of the Pseudo-Voigt function and parameters.
