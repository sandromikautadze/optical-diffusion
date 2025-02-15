# Data

The raw dataset is taken from [RealEstate10K](https://google.github.io/realestate10k/index.html), introduced in:

> Zhou, Tinghui, Tucker, Richard, Flynn, John, Fyffe, Graham, and Snavely, Noah.  
> *Stereo Magnification: Learning View Synthesis using Multiplane Images.*  
> **In SIGGRAPH, 2018**.

It contains **camera poses for approximately 10 million frames** from **80,000 video clips**, sourced from **10,000 YouTube videos**.

For memory reasons, the raw data and the processed data are not pushed to the repo. 

For those interested in reproducing our data, the functions used to extract the data, and process the Plücker embeddings, are in `src/data.py`. The extraction and processing part is in the `data_processing.ipynb` file.