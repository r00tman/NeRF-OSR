# NeRF for Outdoor Scene Relighting
[Viktor Rudnev](https://twitter.com/realr00tman), [Mohamed Elgharib](https://people.mpi-inf.mpg.de/~elgharib/), [William Smith](https://www-users.cs.york.ac.uk/wsmith/), [Lingjie Liu](https://lingjie0206.github.io/), [Vladislav Golyanik](https://people.mpi-inf.mpg.de/~golyanik/), [Christian Theobalt](https://www.mpi-inf.mpg.de/~theobalt/)

![](demo/NeRFOSR2.gif)

Codebase for ECCV 2022 paper "[NeRF for Outdoor Scene Relighting](https://4dqv.mpi-inf.mpg.de/NeRF-OSR/)".

Based on [NeRF++ codebase](https://github.com/Kai-46/nerfplusplus) and inherits the same training data preprocessing and format.

## Data

Our datasets and preprocessed Trevi dataset from PhotoTourism can be found [here](https://nextcloud.mpi-klsb.mpg.de/index.php/s/mGXYKpD8raQ8nMk). Put the downloaded folders into `data/` sub-folder in the code directory.

See NeRF++ sections on [data](https://github.com/Kai-46/nerfplusplus#data) and [COLMAP](https://github.com/Kai-46/nerfplusplus#generate-camera-parameters-intrinsics-and-poses-with-colmap-sfm) on how to create adapt a new dataset for training. In addition, we also support masking via adding `mask` directory with monochrome masks alongside `rgb` directory in the dataset folder. For more details, refer to the provided datasets.


So, if you have an image dataset, you would need to do the following:
1. Set the path to your colmap binary in [colmap_runner/run_colmap.py:13](https://github.com/r00tman/NeRF-OSR/blob/main/colmap_runner/run_colmap.py#L13).
2. Create a dataset directory in `data/`, e.g., `data/newdataset` and create `source` and `out` subfolders, e.g., `data/newdataset/source`, `data/newdataset/out`.
3. Copy all the images to `data/newdataset/source`.
4. Run `colmap_runner/run_colmap.py data/newdataset` in the root folder.
5. This will set the data up, undistort images to `data/newdataset/rgb`, and calibrate the camera parameters to `data/newdataset/kai_cameras_normalized.json`.
6. Optionally, you can now generate the masks by using `data/newdataset/rgb/*` images as the source, to filter out, e.g., people, bicycle, cars or any other dynamic objects. We used [this](https://github.com/NVIDIA/semantic-segmentation) repository to generate the masks. The grayscale masks should be placed to `data/newdataset/mask/` subfolder. You can use the provided datasets as reference. 
7. Now that we have all data and calibrations, we need to create `train`, `val`, `test` splits. To do so, first create corresponding subfolders: `data/newdataset/{train,val,test}/rgb`. Then split the images as you like by copying them from `data/newdataset/rgb` to the corresponding split's `rgb` folder, e.g., `data/newdataset/train/rgb/`.
8. Now you want to generate camera parameters for splits by running `colmap_runner/cvt.py` while in the dataset directory. It will automatically copy all camera parameters and masks to the split folders. 
9. The dataset folder is ready. Now you need to create the dataset config. You can copy the config from the provided dataset, e.g., [here](https://github.com/r00tman/NeRF-OSR/blob/main/configs/europa/final.txt), to `configs/newdataset.txt`. Then you would need to change `datadir` to `data`, `scene` to `newdataset`, and `expname` in the config.
10. Now you can launch the training by `python ddp_train_nerf.py --config configs/newdataset.txt`

## Models

We provide pre-trained models [here](https://nextcloud.mpi-klsb.mpg.de/index.php/s/mGXYKpD8raQ8nMk). Put the folders into `logs/` sub-directory. Use the scripts from `scripts/` subfolder for testing.

## Create environment

```
conda env create --file environment.yml
conda activate nerfosr
```

## Training and Testing

Use the scripts from `scripts/` subfolder for training and testing.

## VR Demo

Please find precompiled binaries, source code, and the extracted Site 1 mesh from [here](https://nextcloud.mpi-klsb.mpg.de/index.php/s/mGXYKpD8raQ8nMk).

To run the demo, make sure you have an OpenVR runtime such as SteamVR and launch `run.bat` in `hellovr_opengl` directory.

To extract the mesh from another model, run

```
ddp_mesh_nerf.py --config lk2/final.txt
```

The list of folder name correspondences can be found in the README of the [dataset](https://nextcloud.mpi-klsb.mpg.de/index.php/s/mGXYKpD8raQ8nMk).

Note that in the VR demo executable, we also clip the model to keep only the main building on ll. 1446-1449. The bounds are hard-coded for the Site 1.

To recompile the code, refer to [OpenVR](https://github.com/ValveSoftware/openvr) instructions, as the demo is based on one of the samples.

## Citation

Please cite our work if you use the code.

```
@InProceedings{rudnev2022nerfosr,
      title={NeRF for Outdoor Scene Relighting},
      author={Viktor Rudnev and Mohamed Elgharib and William Smith and Lingjie Liu and Vladislav Golyanik and Christian Theobalt},
      booktitle={European Conference on Computer Vision (ECCV)},
      year={2022}
}
```

## License

Permission is hereby granted, free of charge, to any person or company obtaining a copy of this software and associated documentation files (the "Software") from the copyright holders to use the Software for any non-commercial purpose. Publication, redistribution and (re)selling of the software, of modifications, extensions, and derivates of it, and of other software containing portions of the licensed Software, are not permitted. The Copyright holder is permitted to publically disclose and advertise the use of the software by any licensee.

Packaging or distributing parts or whole of the provided software (including code, models and data) as is or as part of other software is prohibited. Commercial use of parts or whole of the provided software (including code, models and data) is strictly prohibited. Using the provided software for promotion of a commercial entity or product, or in any other manner which directly or indirectly results in commercial gains is strictly prohibited.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
