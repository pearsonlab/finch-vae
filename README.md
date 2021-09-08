### Joint-encoding VAE for zebra finch song spectrograms and neural data

This is code used to perform the VAE analyses in
"Neural dynamics underlying birdsong practice and performance" (in press).

The main files are `ssl/poe_finch.py`, which contains the product of experts
joint-encoding VAE, and `ssl/cca_finch.py`, which contains the baseline
separate-encoding model. Additionally, `finch_exp.py` contains code for testing
model performance, `neural_axis_two_color_plots.py` contains code for
visualizing the shared information found by the models, and `detrend_specs.py`
contains code for remove time-of-day trends from song and neural activity.

If you are interested in applying a joint-encoding VAE to your own data, see the
[PoE VAE repo](https://github.com/jackgoffinet/poe-vae) for more flexible
implementations of various multimodal VAEs with better documentation.
