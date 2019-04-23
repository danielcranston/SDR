bin="$(pwd)/build/bin/mytest"
datasetroot="$(pwd)/data"
resultsroot="$(pwd)/results/Test"

echo ${bin}

mkdir -p ${resultsroot}
${bin} -targetDir ${datasetroot}/Test -outputDir ${resultsroot} -mode MiddV3 -lambda 0.30 -seg_k 30.0 -inlier_ratio 0.50
