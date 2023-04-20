#ifndef WAIFU2X_NCNN_VULKAN_WAIFU2X_WRAPPED_H
#define WAIFU2X_NCNN_VULKAN_WAIFU2X_WRAPPED_H

#include "waifu2x.h"
#include "pybind11/include/pybind11/pybind11.h"
#include <locale>
#include <codecvt>
#include <utility>
#include <iostream>

// wrapper class of ncnn::Mat
class Image {
public:
    std::string d;
    int w;
    int h;
    int c;

    Image(std::string d, int w, int h, int c);

    void set_data(std::string data);

    pybind11::bytes get_data() const;
};

class Waifu2xWrapped : public Waifu2x {
public:
    Waifu2xWrapped(int gpuid, bool tta_mode, int num_threads);

    int get_tilesize(int _scale, std::string _model) const;

    int get_prepadding(int _scale, int _noise, std::string _model) const;

    // waifu2x parameters
    void set_parameters(int _noise, int _scale, int _prepadding, int _tilesize, pybind11::str py_model);

    int load(const std::string &parampath, const std::string &modelpath);

    int process(const Image &inimage, Image &outimage) const;

    int process_cpu(const Image &inimage, Image &outimage) const;

private:
    int gpuid;
};

int get_gpu_count();

void destroy_gpu_instance();

#endif // WAIFU2X_NCNN_VULKAN_WAIFU2X_WRAPPED_H
