bin="$(pwd)/build/bin/mytest"
datasetroot="$(pwd)/data"


echo ${bin}

for pair in middle_pair right_pair #left_pair #middle_pair right_pair
do
	for i in $(seq 21 25);
	do
		item=$(printf %03d $i)
		echo ------------------------------------
		echo Calculating $pair $item ...
		echo ------------------------------------
		${bin} -targetDir ${datasetroot}/liu_dataset/$pair -outputDir $item -mode LIU -lambda 0.30 -seg_k 30.0 -inlier_ratio 0.50
		echo End 

	done
done
