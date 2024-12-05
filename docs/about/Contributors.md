#People

# People

nmrlineshapeanalyser is written by [Biffo Abdulkadir Olatunbosun](https://scholar.google.com/citations?user=TievH90AAAAJ&hl=en)


# Active Maintainers

The current maintainer is [Biffo Abdulkadir Olatunbosun](https://github.com/BiffoQ)

# Inspiration

On a cold morning while surviving the usual Dutch weather, I had a conversation with my PhD supervisor [Pedro Braga Groszewicz](https://scholar.google.com/citations?user=05NlC3gAAAAJ&hl=en) regarding how deconvolute Lorentzian and Gaussian areas from a quite complicated 1D 17O MAS ss-NMR spectrum. He explained the so-called Voigt and Pseudo-Voigt concepts and how they could helpful in this deconvolution. He made mention of [MestreNova](https://mestrelab.com/main-product/mnova) software, and indeed this software offers a solution to my problem. However, I am not just interested in the solution but also the blackbox.

So I embarked on this journey of unravelling the blackbox and I did not disengaged until I found out how it was done. And this journey led to this package.

I would like to thank [NMRglue](https://github.com/jjhelmus/nmrglue) creator for making it possible to work directly with [Bruker's](https://www.bruker.com/en.html) folder. Also, a big thanks to the creator(s) of [LMFit's](https://lmfit.github.io/lmfit-py/builtin_models.html) builtin models -- the content was pivotal in the understanding of the concept of Pseudo-Voigt function and parameters.
