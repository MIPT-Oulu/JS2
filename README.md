will be available soon..

# JS2
A Lightweight CNN and Joint Shape-Joint Space (JS2) Descriptor for Radiological Osteoarthritis Detection

Implementation of "A Lightweight CNN and Joint Shape-Joint Space (JS2) Descriptor for Radiological Osteoarthritis Detection", MIUA 2020 paper.

![Summary](Pictures/summary.png)

## Data:
Multicenter Osteoarthritis Study (MOST): http://most.ucsf.edu/

The Osteoarthritis Initiative (OAI): https://nda.nih.gov/oai/

## Setup
1. Obtain OAI and MOST Data from the links above.
2. Extract Landmarks (you can use BoneFinder software: http://bone-finder.com/).
3. Save Dicom image and list of landmark points for each sample in a seperate `.npy` file.
4. Prepare medial tibia crops and JSW measurements using `prepare_crops.py`. This will save marginal medial tibia ROI and JSW measurements (minJSW, fJSW, JS2, jointsize) as `.npy` files.

## Run
1. Select the model `arg.model`

## License
Please cite:
```
@article{bayramoglu2020js2,
  title={A Lightweight CNN and Joint Shape-Joint Space (JS2) Descriptor for Radiological Osteoarthritis Detection},
  author={Bayramoglu, Neslihan and Miika, Nieminen and Saarakkala, Simo},
  journal={MIUA 2020, arXiv preprint },
  year={2020}
}```
