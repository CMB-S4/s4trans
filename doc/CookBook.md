# S4 transient pipeline cookbook
### Cook Book for processing DC0 simulations


These instructions assume that the hd5 files have been converted to healpix fits files using toast in a previous step.<br>
For the ICC the healpix files live in:<br>
+ `/projects/caps/cmbs4/sims/healpix_maps_DC0/f090`
+ `/projects/caps/cmbs4/sims/healpix_maps_DC0/f150`

There are two types of files:
+ `*.fits.gz` (sci data)
+ `*._invcov.fits.gz` (weight)

#### Step 1
Project and filter the healpix and insert sources at different flux levels. ﻿
We use two projections for testing:
- proj_01-04
- small

For each projection we insert sources at different flux levels:
- 200 mJy
- 100 mJy
- 50 mJy
- 30 mJy

To do this is better to work in a directory where we will submit the jobs for the ICC. You can copy all of the relevant files from `/home/felipe/submit_box` on the ICC

```
# Create a directory to submit jobs via slurm
mkdir ~/submit_box
cd submit_box
# Set the environment to create the job submission
source /projects/caps/spt3g/opt/miniconda3/bin/activate

```
