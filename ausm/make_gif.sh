#!/bin/bash
# Install imagemagick if not there
# sudo apt-get install imagemagick
convert -resize 85% -delay 10 -loop 0 analytical_plots/fig_Sod_AUSM_it*.png ausm_scheme.gif
# convert -resize 15% -delay 10 -loop 0 figures/analytical*.png analytical.gif
