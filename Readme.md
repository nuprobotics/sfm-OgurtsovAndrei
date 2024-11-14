[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/uAguylQL)
# SFM practise tasks

## Introduction

Task 1 should be completed in `math_solution.py` file. Tasks 2-5 should be completed in `cv2_solution.py` file.

In file `math_solution.py` you are not allowed to use OpenCV library. 

In file `cv2_solution.py` you are allowed to use OpenCV library.

## Task 1

Create a function that performs triangulation for the 3D points.

**Autotests specs:**
+ Function should be called `triangulation`
+ Function should take camera matrix, rotation vector, translation vector,
  image points for the first image, image points for the second image
+ Function should return numpy array with 3D points of shape (N, 3) where N is the number of points

To test your implementation run:
```bash
./auto_test.sh MathTriangulationTest
```

## Task 2

Create a function that takes pair of images as an input, compute key points and descriptors for them using SIFT. 
The function should return key points for both images and matches between them. Don't forget to 
implement k-ratio test and left-right check. k for k-ratio test should be 0.75.

**Autotests specs:**
+ Function should be called `get_matches`
+ Function should take two images and floating point number as input
+ Function should return two sequences with key points and sequence with matches: 
`typing.Sequence[cv2.KeyPoint], typing.Sequence[cv2.KeyPoint], typing.Sequence[cv2.DMatch]`

To test your implementation run:
```bash
./auto_test.sh GetMatchesTest
```

## Task 3

Create a function that performs triangulation for the 3D points.

**Autotests specs:**
+ Function should be called `triangulation`
+ Function should take camera matrix, rotation vector, translation vector,
  image points for the first image, image points for the second image, matches between them
+ Function should return numpy array with 3D points of shape (N, 3) where N is the number of points

To test your implementation run:
```bash
./auto_test.sh TriangulationTest
```

## Task 4 

Create a function that performs resection.

**Autotests specs:**
+ Function should be called `resection`
+ Function should take two images, camera matrix, matches between image points on the first two images and 3D points as input
+ Function should return rotation matrix and translation vector

To test your implementation run:
```bash
./auto_test.sh ResectionTest
```

## Task 5 

Create a function that translate rotation matrix and translation vector to camera rotation and camera position in world coordinate system

**Autotests specs:**
+ Function should be called `convert_to_world_frame`
+ Function should take translation vector and rotation matrix
+ Function should return camera position and matrix of camera orientation

To test your implementation run:
```bash
./auto_test.sh ConvertToWorldFrameTest
```

