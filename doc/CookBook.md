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
Additionally for different projections change`--proj_name` accordingly to either: `small` or `proj_01-04`. <br>
For `--proj_name small` we need to use: ` --source_catalog /projects/caps/cmbs4/etc/source_catalog_small_200mJy.cat` <br>
For `--proj_name proj_01-04` we need to use: `--source_catalog /projects/caps/cmbs4/etc/source_catalog_200mJy.cat` <br>
The above examples are for 200mJy<br>
Also the above example uses singularity image `s4trans-0.3.1.sif` derived from docker image `menanteau/s4trans:0.3.1`. The newer image `menanteau/s4trans:0.3.2` 

Note: to select the input files that have good coverage over a "patch" or projection can be done by querying the sqlite database.<br>
For example, to select all files with on_fraction > 0.9 (i.e. 90%) for projection name `proj_01-04`:
```
%> sqlite3 /projects/caps/cmbs4/dlib/s4sims_DC0.db 
select SIMID, fraction from on_fraction_DC0  where proj='proj_01-04' and fraction>0.9;
```

#### Step 2
Run the light curve detections. From the file `source_catalog_small_200mJy.cat ` we know that the center or peak of the flux is centered in observation: `40-161-9`. Therefore we run the detection for a range of +5,-5 observations (sims) around  observation `40-161-9.`

```
limit=200mJy
SIMS_DIR=filter_sims_DC0_${limit}
proj=small
lightcurves_s4.py \
  ${SIMS_DIR}/mapmaker_RISING_SCAN_40-150-10_noiseweighted_map_nside4096_flt_${proj}.g3.gz \
  ${SIMS_DIR}/mapmaker_RISING_SCAN_40-152-10_noiseweighted_map_nside4096_flt_${proj}.g3.gz \
  ${SIMS_DIR}/mapmaker_RISING_SCAN_40-154-10_noiseweighted_map_nside4096_flt_${proj}.g3.gz \
  ${SIMS_DIR}/mapmaker_RISING_SCAN_40-155-9_noiseweighted_map_nside4096_flt_${proj}.g3.gz  \
  ${SIMS_DIR}/mapmaker_RISING_SCAN_40-160-9_noiseweighted_map_nside4096_flt_${proj}.g3.gz  \
  ${SIMS_DIR}/mapmaker_RISING_SCAN_40-161-9_noiseweighted_map_nside4096_flt_${proj}.g3.gz  \
  ${SIMS_DIR}/mapmaker_RISING_SCAN_40-163-9_noiseweighted_map_nside4096_flt_${proj}.g3.gz  \
  ${SIMS_DIR}/mapmaker_RISING_SCAN_40-165-8_noiseweighted_map_nside4096_flt_${proj}.g3.gz  \
  ${SIMS_DIR}/mapmaker_RISING_SCAN_40-166-9_noiseweighted_map_nside4096_flt_${proj}.g3.gz  \
  ${SIMS_DIR}/mapmaker_RISING_SCAN_40-168-8_noiseweighted_map_nside4096_flt_${proj}.g3.gz  \
  ${SIMS_DIR}/mapmaker_RISING_SCAN_40-174-8_noiseweighted_map_nside4096_flt_${proj}.g3.gz  \
        -o s4-DC0-${limit}.g3 \
        -v --no-noise-cut \
        --sn-threshold 8 \
        --ts-min 100 \
        --fit-bands 150GHz
```

The output of this call is `s4-DC0-200mJy.g3`, which can be used to find/plot the objects detected.

The above call will run using `menanteau/s4trans:0.3.2` 
