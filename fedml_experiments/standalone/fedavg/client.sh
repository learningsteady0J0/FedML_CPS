sh run_fedavg_standalone_pytorch.sh 0 100 3 16 mnist ./../../../data/mnist cnn homo 150 5 0.03 sgd 0 0 test
sh run_fedavg_standalone_pytorch.sh 0 100 3 16 mnist ./../../../data/mnist cnn hetero 150 5 0.03 sgd 0 0 test
sh run_fedavg_standalone_pytorch.sh 0 100 3 16 mnist ./../../../data/mnist cnn class 150 5 0.03 sgd 0 0 test
sh run_fedavg_standalone_pytorch.sh 0 1000 3 16 mnist ./../../../data/mnist cnn class 150 5 0.03 sgd 0 0 test
