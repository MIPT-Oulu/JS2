# JS2

Implementation of "A Lightweight CNN and Joint Shape-Joint Space (JS2) Descriptor for Radiological Osteoarthritis Detection", MIUA 2020 paper.

https://arxiv.org/abs/2005.11715

![Summary](Pictures/summary.png)

## Data:
Multicenter Osteoarthritis Study (MOST): http://most.ucsf.edu/

The Osteoarthritis Initiative (OAI): https://nda.nih.gov/oai/

## Setup
1. Obtain OAI and MOST Data from the links above. We can not distribute the data.
2. Extract Landmarks using BoneFinder(BF) software: http://bone-finder.com/.
3. Localize both left and right knee joints using BF landmarks.
4. Save each joint and associated list of landmark points in a seperate `.npy` file (patientID_SIDE.npy, i.e. "9000099_L.npy").
5. Create a conda environment as follows:

    `conda env create -f env.yml`
6. Prepare medial tibia crops and JSW measurements using `prepare_crops.py`. This will save marginal medial tibia ROI and JSW measurements (minJSW, fJSW, JS2, jointsize) as `.npy` files.

## Run
Select model according to models defined in `models.py`

`python main.py --model combined`


## License
Please cite:
```
@article{bayramoglu2020js2,
  title={A Lightweight CNN and Joint Shape-Joint Space (JS2) Descriptor for Radiological Osteoarthritis Detection},
  author={Bayramoglu, Neslihan and Miika, Nieminen and Saarakkala, Simo},
  journal={MIUA 2020, arXiv preprint },
  year={2020}
}```
