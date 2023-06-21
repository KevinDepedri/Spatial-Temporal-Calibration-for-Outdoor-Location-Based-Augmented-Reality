# Spatial-Temporal Calibration for Outdoor Location-Based Augmented Reality

by
Lorenzo Orlandi,
Kevin Depedri,
Nicola Conci

This paper has been submitted for publication in *IEEE MMSP 2023*.

In this paper we tackle the problem of accuracy for AR solutions developed using the location-based approach, where the pose of the AR contents that need to be visualized is expressed using geographic coordinates. More in detail, we propose a novel procedure to perform the spatial calibration between the smartphone used to visualize AR contents and an external GNSS-RTK receiver, used to acquire the position of the user in the world with centimeter-accuracy. Later, we face all the issue related to the temporal calibration between the two devices, proposing an experimental solution to model the delay between the devices and to mitigate it. Finally, the entire pipeline used to visualize AR contents depending on their geographical coordinates is described, explaining how each part of it works. 

The first two videos attached to this repository show the impact of the spatial calibration and of the temporal calibration on the visualization of holograms. The third video shows how it is possible to acquire a geo-referenced dataset by reversing the pipeline discussed in the paper. Later, such geo-referenced dataset can be used to reconstruct the 3D model of the acquired scene, which, thanks to the camera pose of the images is inclusive of its scale information and of the pose to geo-localise it. Finally, the fourth video shows how the reconstructed model can be visualized in AR exploiting the pipeline previously defined.

<!-- 
![](manuscript/figures/hawaii-trend.png)

*Caption for the example figure with the main results.* -->


## Abstract
The 3D digitalization of contents and their visualization using Augmented Reality (AR) has gained a significant interest within the scientific community. Researchers from various fields have recognized the potential of these technologies and have been actively exploring their applications and implications. The potential lies in the ability to provide users with easy access to digitized information by seamlessly integrating contents directly into their field of view. One of the most promising approaches for outdoor scenarios is the so-called location-based AR, where contents are displayed by leveraging satellite positioning (as GNSS) combined with inertial (as IMUs) sensors. Although the number of application fields are numerous, the accuracy of the over-imposition of the additional contents still hinders a widespread adoption of such technologies. In this paper we propose the combination of a GNSS device equipped with real-time kinematic position (RTK), and a regular smartphone, implementing a novel offline calibration process that relies on a motion capture system (MoCap). The proposed solution is capable to ensure the temporal consistency, and allows for real-time acquisition at centimeter-level accuracy.

## Results
### Effect of the spatial calibration
[![Spatial callibration](https://img.youtube.com/vi/uJmRfPYCVnI/0.jpg)](https://www.youtube.com/watch?v=uJmRfPYCVnI)

### Effect of the temporal calibration
[![Temporal calibration](https://img.youtube.com/vi/bw9R_e6_UiI/0.jpg)](https://www.youtube.com/watch?v=bw9R_e6_UiI)

### Acquisition of geo-referenced images to build geo-referenced 3D models
[![Dataset acquisition](https://img.youtube.com/vi/Ynfc6QxaYrU/0.jpg)](https://www.youtube.com/watch?v=Ynfc6QxaYrU)

### Visualization of 3D reconstructed models
[![3D model visualization](https://img.youtube.com/vi/MiEEN89DyKM/0.jpg)](https://www.youtube.com/watch?v=MiEEN89DyKM)


## Software implementation
All source code used to generate the results and figures in the paper can be find in the folders of this repository. More in detail, the script used to compute the calibration matrix between the GNSS-RTK receiver and the Smartphone is located in the folder ``spatial_calibration``. In the same way the notebooks used to model the temporal delay between the two devices can be find in the folder ``temporal_calibration``. The folder ``general_data_processing`` encompasses some other notebooks used to process initial data coming from the two devices and are not discussed in the paper.

The calculations and figure generation are all run inside [Jupyter notebooks](http://jupyter.org/) or python scripts powered by the [PyCharm IDE](https://www.jetbrains.com/pycharm/).
The data used for each calibration procedure is present in the relative ``files`` subfolder.


## Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

```bash
git clone https://github.com/KevinDepedri/Spatial-Temporal-Calibration-for-Outdoor-Location-Based-Augmented-Reality
```

or [download a zip archive](https://github.com/KevinDepedri/Spatial-Temporal-Calibration-for-Outdoor-Location-Based-Augmented-Reality/archive/refs/heads/main.zip).

A copy of the repository is also archived at *insert DOI here*


## Dependencies

You'll need a working Python environment to run the code.
The required dependencies are specified in the file `requirements.txt`.

Thus, you can install our dependencies running the following command in the repository folder (where `requirements.txt`is located) to create a separate environment and install all required dependencies in it:

```bash
python -m venv venv
venv/Scripts/pip install -r requirements.txt
```

<!-- 
## License

All source code is made available under a BSD 3-clause license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE.md` for the full license text.

The manuscript text is not open source. The authors reserve the rights to the
article content, which is currently submitted for publication in the IEEE MMSP 2023.
-->

## Contacts

Lorenzo Orlandi - lorenzo.orlandi@{arcoda.it,unitn.it}

Kevin Depedri - {kevin.depedri}@studenti.unitn.it

Nicola Conci - {nicola.conci}@unitn.it
