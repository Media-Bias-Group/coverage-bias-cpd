python.exe generate_cp.py --num-train 5 --num-valid 1 --num-test 1 --out cp_loc --change-type loc
python.exe generate_cp.py --num-train 5 --num-valid 1 --num-test 1 --out cp_vel --change-type vel
python.exe generate_cp.py --num-train 5 --num-valid 1 --num-test 1 --out cp_edge --change-type edge
# combine generated change-point time series to form a dataset
python.exe generate_dataset.py