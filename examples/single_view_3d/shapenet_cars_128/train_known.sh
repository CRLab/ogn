$CAFFE_ROOT/build/tools/caffe train --solver=solver_known.prototxt 2>&1| tee logs/viewgen-`date +%F_%R`.log
