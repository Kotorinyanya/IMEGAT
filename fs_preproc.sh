#!/usr/bin/env bash
# define compulsory and optional arguments
while getopts ":L:a:d:" o; do
    case "${o}" in
        L)
            L=${OPTARG}
            ;;
        a)
            a=${OPTARG}
            ;;
        d)
            d=${OPTARG}
            ;;
    esac
done

export subject_list=$L
export annotation_file=$a
export output_dir=$d
export get_anatomical_stats=YES

# initialize
mkdir -p ${output_dir}
mkdir -p ${output_dir}/label
export rand_id=$RANDOM
mkdir -p ${output_dir}/temp_${rand_id}
rm -f ${output_dir}/temp_${rand_id}/colortab_?
rm -f ${output_dir}/temp_${rand_id}/list_labels_${annotation_file}?

# Convert annotation to label, and get color lookup tables
rm -f ./${output_dir}/log_annotation2label
mri_annotation2label --subject fsaverage --hemi lh --outdir ./${output_dir}/label --annotation ${annotation_file} >> ./${output_dir}/temp_${rand_id}/log_annotation2label
mri_annotation2label --subject fsaverage --hemi lh --outdir ./${output_dir}/label --annotation ${annotation_file} --ctab ${output_dir}/temp_${rand_id}/colortab_${annotation_file}_L1 >> ./${output_dir}/temp_${rand_id}/log_annotation2label
mri_annotation2label --subject fsaverage --hemi rh --outdir ./${output_dir}/label --annotation ${annotation_file} >> ./${output_dir}/temp_${rand_id}/log_annotation2label
mri_annotation2label --subject fsaverage --hemi rh --outdir ./${output_dir}/label --annotation ${annotation_file} --ctab ${output_dir}/temp_${rand_id}/colortab_${annotation_file}_R1 >> ./${output_dir}/temp_${rand_id}/log_annotation2label

# Remove number columns from ctab
awk '!($1="")' ${output_dir}/temp_${rand_id}/colortab_${annotation_file}_L1 >> ${output_dir}/temp_${rand_id}/colortab_${annotation_file}_L2
awk '!($1="")' ${output_dir}/temp_${rand_id}/colortab_${annotation_file}_R1 >> ${output_dir}/temp_${rand_id}/colortab_${annotation_file}_R2

# Create list with region names
awk '{print $2}' ${output_dir}/temp_${rand_id}/colortab_${annotation_file}_L1 > ${output_dir}/temp_${rand_id}/list_labels_${annotation_file}L1
awk '{print $2}' ${output_dir}/temp_${rand_id}/colortab_${annotation_file}_R1 > ${output_dir}/temp_${rand_id}/list_labels_${annotation_file}R1

# Create lists with regions that actually have corresponding labels
for labelsL in `cat ${output_dir}/temp_${rand_id}/list_labels_${annotation_file}L1`
	do if [[ -e ${output_dir}/label/lh.${labelsL}.label ]]
		then
		echo lh.${labelsL}.label >> ${output_dir}/temp_${rand_id}/list_labels_${annotation_file}L
		grep " ${labelsL} " ${output_dir}/temp_${rand_id}/colortab_${annotation_file}_L2 >> ${output_dir}/temp_${rand_id}/colortab_${annotation_file}_L3
	fi
done
for labelsR in `cat ${output_dir}/temp_${rand_id}/list_labels_${annotation_file}R1`
	do if [[ -e ${output_dir}/label/rh.${labelsR}.label ]]
		then
		echo rh.${labelsR}.label >> ${output_dir}/temp_${rand_id}/list_labels_${annotation_file}R
		grep " ${labelsR} " ${output_dir}/temp_${rand_id}/colortab_${annotation_file}_R2 >> ${output_dir}/temp_${rand_id}/colortab_${annotation_file}_R3
	fi
done

# Create new numbers column
number_labels_R=`wc -l < ${output_dir}/temp_${rand_id}/list_labels_${annotation_file}R`
number_labels_L=`wc -l < ${output_dir}/temp_${rand_id}/list_labels_${annotation_file}L`

for ((i=1;i<=${number_labels_L};i+=1))
	do num=`echo "${i}+1000" | bc`
	printf "$num\n" >> ${output_dir}/temp_${rand_id}/LUT_number_table_${annotation_file}L
	printf "$i\n" >> ${output_dir}/temp_${rand_id}/${annotation_file}_number_tableL
done
for ((i=1;i<=${number_labels_R};i+=1))
	do num=`echo "${i}+2000" | bc`
	printf "$num\n" >> ${output_dir}/temp_${rand_id}/LUT_number_table_${annotation_file}R
	printf "$i\n" >> ${output_dir}/temp_${rand_id}/${annotation_file}_number_tableR
done

# Create ctabs with actual regions
paste ${output_dir}/temp_${rand_id}/${annotation_file}_number_tableL ${output_dir}/temp_${rand_id}/colortab_${annotation_file}_L3 > ${output_dir}/temp_${rand_id}/colortab_${annotation_file}_L
paste ${output_dir}/temp_${rand_id}/LUT_number_table_${annotation_file}L ${output_dir}/temp_${rand_id}/list_labels_${annotation_file}L > ${output_dir}/temp_${rand_id}/LUT_left_${annotation_file}
paste ${output_dir}/temp_${rand_id}/${annotation_file}_number_tableR ${output_dir}/temp_${rand_id}/colortab_${annotation_file}_R3 > ${output_dir}/temp_${rand_id}/colortab_${annotation_file}_R
paste ${output_dir}/temp_${rand_id}/LUT_number_table_${annotation_file}R ${output_dir}/temp_${rand_id}/list_labels_${annotation_file}R > ${output_dir}/temp_${rand_id}/LUT_right_${annotation_file}
cat ${output_dir}/temp_${rand_id}/LUT_left_${annotation_file} ${output_dir}/temp_${rand_id}/LUT_right_${annotation_file} > ${output_dir}/temp_${rand_id}/LUT_${annotation_file}.txt

proc_func(){
    subject=$1

    printf "\n         >>>>         PREPROCESSING ${subject}         <<<< \n"
    printf "\n         >>>>         ${subject} STARTED AT: $(date)\n\n"

	mkdir -p ${output_dir}/${subject}
	mkdir -p ${output_dir}/${subject}/label
	sed '/_H_ROI/d' ${output_dir}/temp_${rand_id}/LUT_${annotation_file}.txt > ${output_dir}/${subject}/LUT_${annotation_file}.txt

	if [[ -e $SUBJECTS_DIR/${subject}/label/lh.${subject}_${annotation_file}.annot ]] && [[ -e $SUBJECTS_DIR/${subject}/label/rh.${subject}_${annotation_file}.annot ]]
		then
		echo ">>>>	Annotation files lh.${subject}_${annotation_file}.annot and rh.${subject}_${annotation_file}.annot already exist in ${subject}/label. Won't perform transformations"
		else

		rm -f ${output_dir}/${subject}/label2annot_${annotation_file}?h.log
		rm -f ${output_dir}/${subject}/log_label2label

        if [[ ! -z "${T}" ]] ; then trgsubject=${T}; else trgsubject=${subject}; fi

		# Convert labels to target space
		for label in `cat ${output_dir}/temp_${rand_id}/list_labels_${annotation_file}R`
			do echo "transforming ${label}"
			mri_label2label --srcsubject fsaverage --srclabel ${output_dir}/label/${label} --trgsubject ${trgsubject} --trglabel ${output_dir}/${subject}/label/${label}.label --regmethod surface --hemi rh >> ${output_dir}/${subject}/log_label2label
		done
		for label in `cat ${output_dir}/temp_${rand_id}/list_labels_${annotation_file}L`
			do echo "transforming ${label}"
			mri_label2label --srcsubject fsaverage --srclabel ${output_dir}/label/${label} --trgsubject ${trgsubject} --trglabel ${output_dir}/${subject}/label/${label}.label --regmethod surface --hemi lh >> ${output_dir}/${subject}/log_label2label
		done

		mkdir -p ${output_dir}/temp_${rand_id}/${subject}

		# Convert labels to annot (in subject space)
		rm -f ${output_dir}/temp_${rand_id}/${subject}/temp_cat_${annotation_file}_R
		rm -f ${output_dir}/temp_${rand_id}/${subject}/temp_cat_${annotation_file}_L
		for labelsR in `cat ${output_dir}/temp_${rand_id}/list_labels_${annotation_file}R`
			do printf " --l ${output_dir}/${subject}/label/${labelsR}" >> ${output_dir}/temp_${rand_id}/${subject}/temp_cat_${annotation_file}_R
		done
		for labelsL in `cat ${output_dir}/temp_${rand_id}/list_labels_${annotation_file}L`
			do if [ -e ${output_dir}/${subject}/label/${labelsL} ]
				then printf " --l ${output_dir}/${subject}/label/${labelsL}" >> ${output_dir}/temp_${rand_id}/${subject}/temp_cat_${annotation_file}_L
			fi
		done

		mris_label2annot --s ${subject} --h rh `cat ${output_dir}/temp_${rand_id}/${subject}/temp_cat_${annotation_file}_R` --a ${subject}_${annotation_file} --ctab ${output_dir}/temp_${rand_id}/colortab_${annotation_file}_R >> ${output_dir}/${subject}/label2annot_${annotation_file}rh.log
		mris_label2annot --s ${subject} --h lh `cat ${output_dir}/temp_${rand_id}/${subject}/temp_cat_${annotation_file}_L` --a ${subject}_${annotation_file} --ctab ${output_dir}/temp_${rand_id}/colortab_${annotation_file}_L >> ${output_dir}/${subject}/label2annot_${annotation_file}lh.log

	fi

#	# Convert annot to volume
#	rm -f ${output_dir}/${subject}/log_aparc2aseg
#	mri_aparc2aseg --s ${subject} --o ${output_dir}/temp_${rand_id}/${annotation_file}.nii.gz  --annot ${subject}_${annotation_file} >> ${output_dir}/${subject}/log_aparc2aseg

#	# Remove hippocampal 'residue' --> voxels assigned to hippocampus in the HCPMMP1.0 parcellation will be very few, corresponding to vertices around the actual structure. These will be given the same voxel values as the hippocampi (as defined by the FS automatic segmentation): 17 (L) and 53 (R)
#	l_hipp_index=`grep 'L_H_ROI.label' ${output_dir}/temp_${rand_id}/LUT_${annotation_file}.txt | cut -c-4`
#	r_hipp_index=`grep 'R_H_ROI.label' ${output_dir}/temp_${rand_id}/LUT_${annotation_file}.txt | cut -c-4`
#
#	fslmaths ${output_dir}/temp_${rand_id}/${annotation_file}.nii.gz -thr $l_hipp_index -uthr $l_hipp_index ${output_dir}/temp_${rand_id}/l_hipp_HCP
#	fslmaths ${output_dir}/temp_${rand_id}/l_hipp_HCP -bin -mul 17 ${output_dir}/temp_${rand_id}/l_hipp_FS
#
#	fslmaths ${output_dir}/temp_${rand_id}/${annotation_file}.nii.gz -thr $r_hipp_index -uthr $r_hipp_index -add ${output_dir}/temp_${rand_id}/l_hipp_HCP ${output_dir}/temp_${rand_id}/l_r_hipp_HCP
#	fslmaths ${output_dir}/temp_${rand_id}/${annotation_file}.nii.gz -thr $r_hipp_index -uthr $r_hipp_index -bin -mul 53 -add ${output_dir}/temp_${rand_id}/l_hipp_FS ${output_dir}/temp_${rand_id}/l_r_hipp_FS
#
#	fslmaths ${output_dir}/temp_${rand_id}/${annotation_file}.nii.gz -sub ${output_dir}/temp_${rand_id}/l_r_hipp_HCP -add ${output_dir}/temp_${rand_id}/l_r_hipp_FS ${output_dir}/${subject}/${annotation_file}.nii.gz

	# Get anatomical stats table
	if [[ ${get_anatomical_stats} == "YES" ]]
		then
		mkdir -p ${output_dir}/${subject}/stats
		mris_anatomical_stats -th3 -mgz -cortex $SUBJECTS_DIR/${subject}/label/lh.cortex.label -a $SUBJECTS_DIR/${subject}/label/lh.${subject}_${annotation_file}.annot -f ${output_dir}/${subject}/stats/lh.${annotation_file}.stats -b ${subject} lh white >> ${output_dir}/${subject}/log_mris_anatomical_stats_lh
		mris_anatomical_stats -th3 -mgz -cortex $SUBJECTS_DIR/${subject}/label/rh.cortex.label -a $SUBJECTS_DIR/${subject}/label/rh.${subject}_${annotation_file}.annot -f ${output_dir}/${subject}/stats/rh.${annotation_file}.stats -b ${subject} rh white >> ${output_dir}/${subject}/log_mris_anatomical_stats_rh

	fi

	if [[ ${colorlut_miss} == "YES" ]]; then printf "\n         >>>>         ERROR: FreeSurferColorLUT.txt file not found. Individual subcortical masks NOT created\n"; fi

    printf "\n         >>>>         ${subject} ENDED AT: $(date)\n\n"
}
export -f proc_func
cat ${subject_list} | parallel proc_func


rm -r ${output_dir}/temp_${rand_id}