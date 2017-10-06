$CAFFE_ROOT/build/tools/caffe train --solver=solver_pred.prototxt --snapshot=snapshots/snapshot_iter_170000.solverstate 2>&1| tee logs/viewgen-`date +%F_%R`.log
