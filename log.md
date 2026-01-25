# Develop log
---

## TODO:
1. Add a slightly real light source
2. Optimize the loop since (now barely support for 10 cubes)
3. Maybe turn to taichi or numba someday
4. Fix when two cubes have the same x, z collide they will crush into each other and the top sink into the bottom utill they coincide


## DONE:
1. Fixed a major bug in moving and rotating the perspective
2. Added the method to start or stop the whole scene
3. Made the gravity more reasonable
4. Added collision detection (GJK)
5. Added EPA algorithm
6. Added a simple demo also a test demo
7. Added a numba decorator in G_Object.py to accelerate the update_position method for the integrator
8. Added method to use q/e to move the perspective
9. Fixed the wheelevent to zoom in and out
10. Added a accmulator to restrict the deltatime