#include "waifu2x_wrapped.h"

// Image Data Structure
Image::Image(std::string d, int w, int h, int c) {
    this->d = std::move(d);
    this->w = w;
    this->h = h;
    this->c = c;
}

void Image::set_data(std::string data) {
    this->d = std::move(data);
}

pybind11::bytes Image::get_data() const {
    return pybind11::bytes(this->d);
}

// Waifu2xWrapped
Waifu2xWrapped::Waifu2xWrapped(int gpuid, bool tta_mode, int num_threads)
        : Waifu2x(gpuid, tta_mode, num_threads) {
    this->gpuid = gpuid;
}

int Waifu2xWrapped::get_tilesize(int _scale, std::string _model) const {
    int tilesize = 0;
    if (this->gpuid == -1) return 400;

    uint32_t heap_budget = ncnn::get_gpu_device(this->gpuid)->get_heap_budget();

    if (_model.find("models-cunet") != std::string::npos) {
        if (heap_budget > 2600)
            tilesize = 400;
        else if (heap_budget > 740)
            tilesize = 200;
        else if (heap_budget > 250)
            tilesize = 100;
        else
            tilesize = 32;
    } else {
        if (heap_budget > 1900)
            tilesize = 400;
        else if (heap_budget > 550)
            tilesize = 200;
        else if (heap_budget > 190)
            tilesize = 100;
        else
            tilesize = 32;
    }

    return tilesize;
}


int Waifu2xWrapped::get_prepadding(int _scale, int _noise, std::string _model) const {
    int prepadding = 0;

    if (_model.find("models-cunet") != std::string::npos) {
        if (_noise == -1) {
            prepadding = 18;
        } else if (_scale == 1) {
            prepadding = 28;
        } else if (_scale == 2 || _scale == 4 || _scale == 8 || _scale == 16 || _scale == 32) {
            prepadding = 18;
        }
    } else if (_model.find("models-upconv_7_anime_style_art_rgb") != std::string::npos) {
        prepadding = 7;
    } else if (_model.find("models-upconv_7_photo") != std::string::npos) {
        prepadding = 7;
    } else {
        std::cout << "unknown model dir type" << std::endl;
        prepadding = 7;
    }

    return prepadding;
}

void Waifu2xWrapped::set_parameters(int _noise, int _scale, int _prepadding, int _tilesize, pybind11::str py_model) {
    Waifu2x::noise = _noise;
    Waifu2x::scale = _scale;
    Waifu2x::tilesize = _tilesize ? _tilesize : Waifu2xWrapped::get_tilesize(_scale, std::string(py_model));
    Waifu2x::prepadding = _prepadding ? _prepadding : Waifu2xWrapped::get_prepadding(_scale, _noise,
                                                                                     std::string(py_model));
}

int Waifu2xWrapped::load(const std::string &parampath,
                         const std::string &modelpath) {
#if _WIN32
    // convert string to wstring
    auto to_wide_string = [&](const std::string& input) {
        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
        return converter.from_bytes(input);
    };
    return Waifu2x::load(to_wide_string(parampath), to_wide_string(modelpath));
#else
    return Waifu2x::load(parampath, modelpath);
#endif
}

int Waifu2xWrapped::process(const Image &inimage, Image &outimage) const {
    int c = inimage.c;
    ncnn::Mat inimagemat =
            ncnn::Mat(inimage.w, inimage.h, (void *) inimage.d.data(), (size_t) c, c);
    ncnn::Mat outimagemat =
            ncnn::Mat(outimage.w, outimage.h, (void *) outimage.d.data(), (size_t) c, c);
    return Waifu2x::process(inimagemat, outimagemat);
}

int Waifu2xWrapped::process_cpu(const Image &inimage, Image &outimage) const {
    int c = inimage.c;
    ncnn::Mat inimagemat =
            ncnn::Mat(inimage.w, inimage.h, (void *) inimage.d.data(), (size_t) c, c);
    ncnn::Mat outimagemat =
            ncnn::Mat(outimage.w, outimage.h, (void *) outimage.d.data(), (size_t) c, c);
    return Waifu2x::process_cpu(inimagemat, outimagemat);
}

int get_gpu_count() { return ncnn::get_gpu_count(); }

void destroy_gpu_instance() { ncnn::destroy_gpu_instance(); }

PYBIND11_MODULE(waifu2x_ncnn_vulkan_wrapper, m) {
    pybind11::class_<Waifu2xWrapped>(m, "Waifu2xWrapped")
            .def(pybind11::init<int, bool, int>())
            .def("load", &Waifu2xWrapped::load)
            .def("process", &Waifu2xWrapped::process)
            .def("process_cpu", &Waifu2xWrapped::process_cpu)
            .def("set_parameters", &Waifu2xWrapped::set_parameters);

    pybind11::class_<Image>(m, "Image")
            .def(pybind11::init<std::string, int, int, int>())
            .def("get_data", &Image::get_data)
            .def("set_data", &Image::set_data);

    m.def("get_gpu_count", &get_gpu_count);

    m.def("destroy_gpu_instance", &destroy_gpu_instance);
}
