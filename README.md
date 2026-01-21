# 3D-Face-Deformation-

This project was developed as part of the **Topics in Artificial Intelligence** course at **Sapienza University of Rome**,  
Faculty of **Artificial Intelligence and Robotics**.

The objective of the project is to **transfer a facial expression from a target 3D face mesh to a neutral 3D face mesh**, producing a deformed model that replicates the target expression while preserving the identity and geometry of the neutral face.

---

## Problem Description

Given:
- a **target 3D face mesh** representing a specific facial expression
- a **neutral 3D face mesh** with the same topology

the task is to deform the neutral mesh so that it reproduces the expression observed in the target mesh.

The focus of the project is on **expression transfer**, not on identity reconstruction or texture synthesis.

---

## Method Overview

The pipeline operates on aligned 3D meshes and computes a deformation that maps the expression from the target mesh onto the neutral one.  
The resulting mesh maintains the structural characteristics of the neutral face while matching the expression dynamics of the target.

This project explores core concepts in:
- 3D geometry processing
- mesh deformation
- facial expression modeling

---

## Results

A qualitative comparison between the input target expression and the resulting deformed neutral mesh is shown below:

<p align="center">
  <img src="assets/target.png" width="45%" />
  <img src="assets/deformed.png" width="45%" />
</p>

**Left:** Target mesh with expression  
**Right:** Neutral mesh after deformation


