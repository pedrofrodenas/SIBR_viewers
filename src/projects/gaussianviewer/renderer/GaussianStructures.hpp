#ifndef SIBR_GAUSSIAN_STRUCTURES_HPP
#define SIBR_GAUSSIAN_STRUCTURES_HPP

#include <core/system/Vector.hpp>  // Defines sibr::Vector3f

struct Scale {
    float scale[3];
};

struct Objects {
    float objects[16];
};

typedef sibr::Vector3f Pos;

template<int D>
struct SHs {
    float shs[(D+1)*(D+1)*3];
};

struct Rot {
    float rot[4];
};

template<int D>
struct RichPoint {
    Pos pos;
    float n[3];
    SHs<D> shs;
    float opacity;
    Scale scale;
    Rot rot;
};

template<int D>
struct ExtendedRichPoint {
    Pos pos;
    float n[3];
    SHs<D> shs;
    float opacity;
    Scale scale;
    Rot rot;
    float obj_dc[16];  // New: object-specific data
};


#endif // SIBR_GAUSSIAN_STRUCTURES_HPP