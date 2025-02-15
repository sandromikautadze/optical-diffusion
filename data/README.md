# Data

The raw dataset is taken from [RealEstate10K](https://google.github.io/realestate10k/index.html), introduced in:

> Zhou, Tinghui, Tucker, Richard, Flynn, John, Fyffe, Graham, and Snavely, Noah.  
> *Stereo Magnification: Learning View Synthesis using Multiplane Images.*  
> **In SIGGRAPH, 2018**.

It contains **camera poses for approximately 10 million frames** from **80,000 video clips**, sourced from **10,000 YouTube videos**.

The functions used to extract the data, and process the Plücker embeddings, is in `src/data.py`. The actual extraction and processing part is in the `data_processing.ipynb` file.