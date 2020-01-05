## MESH QUALITY CALCULATION


*Requires pymesh* 
Run the pymesh docker container in this foler by running the following command:
```bash
docker run -it -v $MY_DIR_PATH/mesh:/root --entrypoint bash pymesh/pymesh
```
where `$MY_DIR_PATH$` is the dir to this repository.


## Files
`algo.py` generates the files required to draw plots and calculate stuff. This file requires pymesh to run. Otheriwse, `quality_plots.py` do not.
The generated files will be in `meshes` directory.