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
We use two projections (patches) for testing:
- `proj_01-04`
- `small`

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

For the small patch example, we insert sources to files in list: `selected_files_40-161-9_small_f150_DC0.cat`
```
create_jobs -c create_jobs -c s4_filter_sims_DC0_200mJy.yaml --loop_list selected_files_40-161-9_small_f150_DC0.cat --submit_dir submit_dir_s4_filter_sims_DC0_200mJy --job_name 200mJy
create_jobs -c create_jobs -c s4_filter_sims_DC0_100mJy.yaml --loop_list selected_files_40-161-9_small_f150_DC0.cat --submit_dir submit_dir_s4_filter_sims_DC0_100mJy --job_name 100mJy
create_jobs -c create_jobs -c s4_filter_sims_DC0_50mJy.yaml  --loop_list selected_files_40-161-9_small_f150_DC0.cat --submit_dir submit_dir_s4_filter_sims_DC0_50mJy  --job_name 50mJy
create_jobs -c create_jobs -c s4_filter_sims_DC0_30mJy.yaml  --loop_list selected_files_40-161-9_small_f150_DC0.cat --submit_dir submit_dir_s4_filter_sims_DC0_30mJy  --job_name 30mJy
```

If we examine the contents of the `s4_filter_sims_DC0_200mJy.yaml` configuration file, we see:
```
# The call that will go inside the singExec script:
cmd_call: >
   filter_sims ${INPUTLIST}
   --proj_name "small" 
   --source_catalog /projects/caps/cmbs4/etc/source_catalog_small_200mJy.cat
   --output_filetypes G3 FITS
   --dbname /projects/caps/cmbs4/dlib/s4sims_DC0.db
   --tablename on_fraction_DC0
   --outdir /projects/caps/cmbs4/sims/filter_sims_DC0_200mJy
   --indirect_write
   --onfracion_thresh 0.9
   --normalize_weight
```
For testing it is important to change `--outdir` to the right directory. <br>
Additionaly for diferent projections change`--proj_name` accordingly to either: `small` or `proj_01-04`. <br>
For `--proj_name small` we need to use: ` --source_catalog /projects/caps/cmbs4/etc/source_catalog_small_200mJy.cat` <br>
For `--proj_name proj_01-04` we need to use: `--source_catalog /projects/caps/cmbs4/etc/source_catalog_200mJy.cat`
The above examples are for 200mJy 

