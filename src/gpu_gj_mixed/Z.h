#ifndef Z_H
#define Z_H

#ifdef __cplusplus
extern "C"{
#endif

struct Z {
	unsigned int step;
    unsigned int landmark_id;
    double z[2];
};

#ifdef __cplusplus
}
#endif
#endif
