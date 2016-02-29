#!/bin/bash
for layer in conv1 conv2 conv3 conv4 conv5; do
	for dir in ./*_"$layer"; do cp -v ../proto/stitch_"$layer".prototxt $dir/network.prototxt; done
        for dir in ./*_"$layer"; do cp -v ../jobs/job_lasso_moran_"$layer".sh $dir/job.sh; done
	for decay in 0 -1 -2 -3 -4 -5; do
		mkdir net0_net1_L1_"$decay"_"$layer"
		for dir in ./*"$decay"_"$layer"; do 
			cp -v ../proto/solver_stitch_conv1_sgd_l1_"$decay".prototxt $dir/solver.prototxt
			#sbatch -J "$layer"_"$decay" -o $dir/stitching.log $dir/job.sh
		done
	done
done

for layer in conv1 conv2 conv3 conv4 conv5; do
	for decay in 0 -1 -2 -3 -4 -5; do
		for dir in ./*"$decay"_"$layer"; do
			cd $dir
			sbatch -J "$layer"_"$decay" -o stitching.log ./job.sh
			cd ../
		done
	done
done
