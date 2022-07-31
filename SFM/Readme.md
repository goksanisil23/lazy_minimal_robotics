# Bundle Adjustment
- Scale ambiguity is still there (if there's no known 3D landmark introduced somewhere in the chain)
- Most clearly stated in the abstract of this paper: https://rpg.ifi.uzh.ch/docs/ICCV09_scaramuzza.pdf
- But the reconstruction is consistent within the chose world coordinates. However that construction would look exactly the same if everything in real 3D world had been upscaled by X.

 - Many of the structure from motion datasets provide 3D landmarks, initial camera poses and matched image features already. 