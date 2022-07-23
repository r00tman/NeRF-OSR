# NeRF for Outdoor Scene Relighting
[Viktor Rudnev](https://twitter.com/realr00tman), [Mohamed Elgharib](https://people.mpi-inf.mpg.de/~elgharib/), [William Smith](https://www-users.cs.york.ac.uk/wsmith/), [Lingjie Liu](https://lingjie0206.github.io/), [Vladislav Golyanik](https://people.mpi-inf.mpg.de/~golyanik/), [Christian Theobalt](https://www.mpi-inf.mpg.de/~theobalt/)

![](demo/NeRFOSR2.gif)

Codebase for ECCV 2022 paper "[NeRF for Outdoor Scene Relighting](https://4dqv.mpi-inf.mpg.de/NeRF-OSR/)".

Based on [NeRF++ codebase](https://github.com/Kai-46/nerfplusplus) and inherits the same training data preprocessing and format.

## Data

Our datasets and preprocessed Trevi dataset from PhotoTourism can be found [here](https://nextcloud.mpi-klsb.mpg.de/index.php/s/mGXYKpD8raQ8nMk). Put the downloaded folders into `data/` sub-folder in the code directory.

See NeRF++ sections on [data](https://github.com/Kai-46/nerfplusplus#data) and [COLMAP](https://github.com/Kai-46/nerfplusplus#generate-camera-parameters-intrinsics-and-poses-with-colmap-sfm) on how to create adapt a new dataset for training. In addition, we also support masking via adding `mask` directory with monochrome masks alongside `rgb` directory in the dataset folder. For more details, refer to the provided datasets.

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
