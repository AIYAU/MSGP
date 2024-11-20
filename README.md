# MSGP

Multi-Semantic Guided Prototype Learning for Cross-Modal Few-Shot Apple Leaf Disease Recognition 
## Requirements

To run this project, you will need to add the following environment variables to your .env file

`python = 3.8.8`

`torch == 2.1.2+cu121`

`torchvision == 0.16.2+cu121`

## Dataset
```
- dataset-1
   ├── Alternaria leaf spot
   │   ├── xxx.png...
   ├── Blossom blight leaves
   │   ├── xxx.png...
   ├── Brown spot
   │   ├── xxx.png...
   └── ....
     └── Other Categories

- dataset-2
  ├── Alternaria leaf spot
  │   ├── xxx.png...
  ├── Apple_scab
  │   ├── xxx.png...
  ├── Black_rot
  │   ├── xxx.png...
  └── ....
-    └── Other Categories
```

The two datasets should be stored at locations of your choosing based on the addresses specified in the particular code.

```
## Usage

1. Run -`run_data1.py`- to obtain the result of dataset 1
2. Run -`run_data2.py`- to obtain the result of dataset 2
