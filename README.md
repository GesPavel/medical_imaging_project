# Running instructions
To run the first part of the assignment, run the following command in the terminal:
```bash
python3 part1.py
```
To run the second part of the assignment, run the following command in the terminal:
```bash
python3 part2.py
```
To run the third part:
1. First run 
```bash
python3 part3_getbb.py
```
and circle the tumor with the bounding box.

2. Then run
```bash
python3 part3_getgt.py
```
and circle the tumor on the MR image as if using 'lasso' tool in Photoshop, to get the ground truth mask.

3. Finally, run
```bash
python3 part3_segment.py
```
to obtain segmentation and its evaluation.

# NOTE: for code to work, the path to data should be altered in consts.py

Also note, that part3_segement.py will not work without samed2 repository in the root folder of the project. To run it properly, Samed2 should be cloned and the checkpoint and memory bank should be placed in the same folder as the project.