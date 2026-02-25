#ifndef LANDMARKKEYSZLIST_H
#define LANDMARKKEYSZLIST_H

#ifdef __cplusplus
extern "C"{
#endif

struct LandmarkKeysZList {
	unsigned int landmark_id;
    struct Z z; // This "z" already has "step"
};

#ifdef __cplusplus
}
#endif
#endif
