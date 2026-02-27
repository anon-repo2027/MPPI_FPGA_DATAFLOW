#ifndef REF_PATH_HPP
#define REF_PATH_HPP


struct waypoint{
    float ref_x;
    float ref_y;
    float ref_yaw;
    float ref_v;
};

// extern waypoint ref_path_array[1201];
// extern float ref_path_array[4804];
extern float ref_path_array[1201][4];

#endif