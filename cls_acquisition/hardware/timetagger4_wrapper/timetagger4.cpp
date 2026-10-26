// The TimeTagger4 wrapper ORACLE's driver (cls_acquisition/hardware/timetagger.py) reads.
//
// Date:    2026-10-27
// Version: 0.2.0
// Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>
//
// The old DAQ's wrapper (collinear_daq_driver_wrap.cpp, by the collinear
// group), with the same Python interface and three faults mended:
//
//   * inputs 3 and 4 (C and D) took input 2's edge setting, not their own;
//   * a hit's flags were never read, so the rollover marks the card sends
//     past the 24-bit timestamp range were recorded as hits;
//   * a card that failed to open, configure or start was reported only on
//     the console, and the next call used a dead handle.
//
// Every failure now raises a RuntimeError naming the step and the driver's
// message. The module carries ORACLE_WRAPPER, so ORACLE knows the edges it
// asks for are the edges it gets. getPackets() hands back (status, rows):
// status 0 with rows of [packet number, input (from 1), timestamp in TDC
// bins] for every trigger read, 1 when there is none yet.
//
// Depends on: pybind11; the cronologic TimeTagger4 driver (TimeTagger4_interface.h,
// xtdc4_driver_64). Build it with build_wrapper.py, as README.md says.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "TimeTagger4_interface.h"

namespace py = pybind11;

namespace {

const int INPUTS = 4;
const int CHANNEL_MASK = 0xf;               // a hit's input, from 0
const int FLAGS_SHIFT = 4;
const int FLAGS_MASK = 0xf;
const int TIMESTAMP_SHIFT = 8;
const int TIMESTAMP_MASK = 0xffffff;        // 24 bits; past them the card marks a rollover

void Set_Edge(timetagger4_configuration& config, int index, bool rising) {
    config.trigger[index].rising = rising ? 1 : 0;
    config.trigger[index].falling = rising ? 0 : 1;
}

}  // namespace

class Time_Tagger {
public:
    Time_Tagger(std::array<bool, INPUTS> used, bool trigger_rising,
                std::array<bool, INPUTS> rising, double trigger_level,
                std::array<double, INPUTS> levels, std::array<int, INPUTS> starts,
                std::array<int, INPUTS> stops) {
        timetagger4_init_parameters params;
        timetagger4_get_default_init_parameters(&params);
        params.buffer_size[0] = 8 * 1024 * 1024;    // 8 MB of packet buffer
        params.board_id = 0;
        params.card_index = 0;                      // the first TimeTagger4 found
        int error_code = 0;
        const char* message = nullptr;
        device_ = timetagger4_init(&params, &error_code, &message);
        if (error_code != CRONO_OK || device_ == nullptr) {
            device_ = nullptr;
            throw std::runtime_error(std::string("The TimeTagger4 could not be opened: ")
                                     + (message ? message : "no message from the driver"));
        }
        timetagger4_configuration config;
        timetagger4_get_default_configuration(device_, &config);
        // the start input's edge and threshold are trigger[0] and dc_offset[0];
        // input i's are trigger[i + 1] and dc_offset[i + 1], each its own
        Set_Edge(config, 0, trigger_rising);
        config.dc_offset[0] = trigger_level;
        for (int i = 0; i < INPUTS; ++i) {
            config.channel[i].enabled = used[i];
            if (!used[i]) {
                continue;
            }
            config.channel[i].start = starts[i];
            config.channel[i].stop = stops[i];
            config.dc_offset[i + 1] = levels[i];
            Set_Edge(config, i + 1, rising[i]);
        }
        Check(timetagger4_configure(device_, &config), "configured");
        // each read acknowledges the last, whose packets are then invalid
        read_config_.acknowledge_last_read = 1;
    }

    ~Time_Tagger() {
        Close();
    }

    int Start_Reading() {
        Check(timetagger4_start_capture(Device()), "started");
        return 0;
    }

    std::tuple<int, std::vector<std::array<int, 3>>> Get_Packets() {
        std::vector<std::array<int, 3>> rows;
        timetagger4_read_out read_data;
        int status = timetagger4_read(Device(), &read_config_, &read_data);
        if (status != CRONO_OK) {
            return std::make_tuple(status, rows);
        }
        for (volatile crono_packet* p = read_data.first_packet; p <= read_data.last_packet;
             p = crono_next_packet(p)) {
            int hits = 2 * static_cast<int>(p->length);
            if (p->flags & TIMETAGGER4_PACKET_FLAG_ODD_HITS) {
                hits -= 1;
            }
            const volatile unsigned int* data =
                reinterpret_cast<const volatile unsigned int*>(p->data);
            for (int i = 0; i < hits; ++i) {
                unsigned int hit = data[i];
                int channel = hit & CHANNEL_MASK;
                int flags = (hit >> FLAGS_SHIFT) & FLAGS_MASK;
                // a rollover mark carries no event, only the counter's wrap
                if (channel >= INPUTS || (flags & TIMETAGGER4_HIT_FLAG_TIME_OVERFLOW)) {
                    continue;
                }
                rows.push_back({packet_number_, channel + 1,
                                static_cast<int>((hit >> TIMESTAMP_SHIFT) & TIMESTAMP_MASK)});
            }
            ++packet_number_;
        }
        return std::make_tuple(status, rows);
    }

    int Stop_Reading() {
        if (device_ != nullptr) {
            timetagger4_stop_capture(device_);
        }
        return 0;
    }

    int Stop() {
        Close();
        return 0;
    }

private:
    timetagger4_device* device_ = nullptr;
    timetagger4_read_in read_config_ = {};
    int packet_number_ = 0;

    timetagger4_device* Device() {
        if (device_ == nullptr) {
            throw std::runtime_error("The TimeTagger4 has been closed.");
        }
        return device_;
    }

    void Check(int status, const char* what) {
        if (status != CRONO_OK) {
            std::string message = timetagger4_get_last_error_message(device_);
            Close();
            throw std::runtime_error(std::string("The TimeTagger4 could not be ") + what + ": "
                                     + message);
        }
    }

    void Close() {
        if (device_ != nullptr) {
            timetagger4_close(device_);
            device_ = nullptr;
        }
    }
};

PYBIND11_MODULE(timetagger4, m) {
    m.attr("ORACLE_WRAPPER") = 1;
    py::class_<Time_Tagger>(m, "TimeTagger")
        .def(py::init([](bool channel_0_used, bool channel_1_used, bool channel_2_used,
                         bool channel_3_used, bool trigger_rising, bool channel_0_rising,
                         bool channel_1_rising, bool channel_2_rising, bool channel_3_rising,
                         double trigger_level, double channel_0_level, double channel_1_level,
                         double channel_2_level, double channel_3_level, int channel_0_start,
                         int channel_1_start, int channel_2_start, int channel_3_start,
                         int channel_0_stop, int channel_1_stop, int channel_2_stop,
                         int channel_3_stop) {
                 return new Time_Tagger(
                     {channel_0_used, channel_1_used, channel_2_used, channel_3_used},
                     trigger_rising,
                     {channel_0_rising, channel_1_rising, channel_2_rising, channel_3_rising},
                     trigger_level,
                     {channel_0_level, channel_1_level, channel_2_level, channel_3_level},
                     {channel_0_start, channel_1_start, channel_2_start, channel_3_start},
                     {channel_0_stop, channel_1_stop, channel_2_stop, channel_3_stop});
             }),
             py::arg("channel_0_used") = false, py::arg("channel_1_used") = false,
             py::arg("channel_2_used") = false, py::arg("channel_3_used") = false,
             py::arg("trigger_rising") = false, py::arg("channel_0_rising") = false,
             py::arg("channel_1_rising") = false, py::arg("channel_2_rising") = false,
             py::arg("channel_3_rising") = false, py::arg("trigger_level") = -0.1,
             py::arg("channel_0_level") = -0.1, py::arg("channel_1_level") = -0.1,
             py::arg("channel_2_level") = -0.1, py::arg("channel_3_level") = -0.1,
             py::arg("channel_0_start") = 0, py::arg("channel_1_start") = 0,
             py::arg("channel_2_start") = 0, py::arg("channel_3_start") = 0,
             py::arg("channel_0_stop") = 16000000, py::arg("channel_1_stop") = 16000000,
             py::arg("channel_2_stop") = 16000000, py::arg("channel_3_stop") = 16000000)
        .def("startReading", &Time_Tagger::Start_Reading)
        .def("getPackets", &Time_Tagger::Get_Packets)
        .def("stopReading", &Time_Tagger::Stop_Reading)
        .def("stop", &Time_Tagger::Stop);
}
