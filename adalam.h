/* SPDX-License-Identifier: Unlicense */

#ifndef I_COREGISTRATION_OPENCL_ADALAM_H
#define I_COREGISTRATION_OPENCL_ADALAM_H

/* AdaLAM (Cavalli et al., 2020) outlier filtering of nearest-neighbour
 * descriptor matches, as kornia.feature.match_adalam. Keypoints carry no
 * orientation or scale here (DISK), so the orientation and scale
 * agreement tests are not applied (the repository's constant LAFs made
 * them always pass). */
struct adalam_config {
    double area_ratio;       /* Image area over seed circle area. */
    double search_expansion; /* Neighbourhood radius over seed radius. */
    int ransac_iters;        /* Hypotheses per neighbourhood. */
    int min_inliers;         /* Inliers a neighbourhood must keep. */
    double min_confidence;   /* Confidence-based RANSAC threshold. */
    double det_thr;          /* Largest scale change of an affinity. */
    int force_seed_mnn;      /* Seeds only among mutual nearest neighbours. */
};

/* kornia's defaults with the overrides of RS_coreg_LoFTR 2_match.py
 * (search_expansion 16, ransac_iters 256, force_seed_mnn off). */
void adalam_defaults(struct adalam_config *c);

/* Match the n1 descriptors d1 of keypoints k1 (x, y pairs) to the n2 of
 * k2 (dimension dim, L2-normalized): nearest neighbour by squared
 * distance with Lowe's ratio of the two nearest as score, then AdaLAM.
 * Writes the accepted matches as index pairs (i1, i2) sorted by i1 then
 * i2 into *pairs (allocated) and their inverse confidence into *dist;
 * returns their number. */
int adalam_match(const float *k1, const float *d1, int n1, const float *k2,
                 const float *d2, int n2, int dim,
                 const struct adalam_config *cfg, int **pairs, float **dist);

#endif /* I_COREGISTRATION_OPENCL_ADALAM_H */
