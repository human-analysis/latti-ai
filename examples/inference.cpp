/*
 * Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

// Unified encrypted inference example.
// Works for any task (MNIST, CIFAR-10, ImageNet, etc.) by specifying --task-dir.
// Demonstrates the InferenceClient / InferenceServer separation.

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "interface/inference_client.h"
#include "interface/inference_server.h"
#include "fhe_ops_lib/utils.h"

using namespace std;
namespace ls = cxx_sdk_v2;
using fhe_ops_lib::print_double_message;

static map<string, vector<double>> read_input_values_json(const string& path) {
    string payload;
    if (path == "-") {
        ostringstream buffer;
        buffer << cin.rdbuf();
        payload = buffer.str();
    } else {
        ifstream input(path);
        if (!input.is_open()) {
            throw runtime_error("Failed to open input JSON: " + path);
        }
        ostringstream buffer;
        buffer << input.rdbuf();
        payload = buffer.str();
    }

    auto root = nlohmann::ordered_json::parse(payload);
    if (!root.is_object()) {
        throw runtime_error("Input JSON must be an object: {name: [values...]}");
    }

    map<string, vector<double>> inputs;
    for (auto& [name, values_json] : root.items()) {
        if (!values_json.is_array()) {
            throw runtime_error("Input JSON value must be an array for input: " + name);
        }
        auto& values = inputs[name];
        values.reserve(values_json.size());
        for (const auto& value : values_json) {
            values.push_back(value.get<double>());
        }
    }
    return inputs;
}

int main(int argc, char* argv[]) {
    string task_dir;
    vector<string> input_args;
    string input_json_path;
    bool use_gpu = false;
    int gpu_device = 0;
    bool verify = false;
    bool plaintext_only = false;
    constexpr double tolerance = 0.1;
    using Clock = std::chrono::steady_clock;
    auto seconds_since = [](Clock::time_point start) {
        return std::chrono::duration<double>(Clock::now() - start).count();
    };
    auto print_timing = [&](const string& label, Clock::time_point start) {
        cout << "[Timing] " << label << ": " << fixed << setprecision(3) << seconds_since(start) << " s" << endl;
    };

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--gpu") == 0) {
            use_gpu = true;
        } else if (strcmp(argv[i], "--gpu-device") == 0 && i + 1 < argc) {
            gpu_device = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--task-dir") == 0 && i + 1 < argc) {
            task_dir = argv[++i];
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_args.push_back(argv[++i]);
        } else if (strcmp(argv[i], "--input-json") == 0 && i + 1 < argc) {
            input_json_path = argv[++i];
        } else if (strcmp(argv[i], "--verify") == 0) {
            verify = true;
        } else if (strcmp(argv[i], "--plaintext-only") == 0) {
            plaintext_only = true;
        }
    }

    if (task_dir.empty() || (input_args.empty() && input_json_path.empty())) {
        cerr << "Usage: " << argv[0]
             << " --task-dir <path> (--input [name=]<path> [--input ...] | --input-json <path|->)"
             << " [--gpu] [--gpu-device <id>] [--verify]"
             << endl;
        return 1;
    }

    // Read input names from task_config
    auto task_config = read_json(task_dir + "/client/task_config.json");
    vector<string> input_names;
    for (auto& [name, _] : task_config["task_input_param"].items()) {
        input_names.push_back(name);
    }

    // Build input CSV map: supports "name=path" or positional mapping
    map<string, string> input_csvs;
    for (size_t i = 0; i < input_args.size(); i++) {
        auto eq_pos = input_args[i].find('=');
        if (eq_pos != string::npos) {
            input_csvs[input_args[i].substr(0, eq_pos)] = input_args[i].substr(eq_pos + 1);
        } else if (i < input_names.size()) {
            input_csvs[input_names[i]] = input_args[i];
        } else {
            cerr << "Too many --input arguments. Expected " << input_names.size() << " inputs." << endl;
            return 1;
        }
    }
    map<string, vector<double>> input_values;
    if (!input_json_path.empty()) {
        input_values = read_input_values_json(input_json_path);
    }

    cout << "========== Encrypted Inference ==========" << endl;
    cout << "Task directory: " << task_dir << endl;
    for (auto& [name, path] : input_csvs) {
        cout << "Input [" << name << "]: " << path << endl;
    }
    for (auto& [name, values] : input_values) {
        cout << "Input [" << name << "]: " << values.size() << " JSON values" << endl;
    }
    cout << "Device:         " << (use_gpu ? "GPU" : "CPU");
    if (use_gpu) cout << " (device " << gpu_device << ")";
    cout << endl;
    if (verify) {
        cout << "Verify mode:    ON (tolerance = " << tolerance << ")" << endl;
    }
    cout << endl;
    const auto total_start = Clock::now();

    // --- Plaintext-only mode: skip crypto, just run plaintext inference ---
    if (plaintext_only) {
        const auto plaintext_start = Clock::now();
        cout << "[Plaintext-only mode] Loading model..." << endl;
        InferenceServer server(task_dir + "/server", use_gpu, gpu_device);
        server.load_model_no_context();
        auto plaintext_outputs = input_values.empty() ? server.evaluate_plaintext(input_csvs)
                                                       : server.evaluate_plaintext_values(input_values);
        cout << "========== Plaintext Results ==========" << endl;
        for (auto& [name, output] : plaintext_outputs) {
            print_double_message(output.data(), ("Plaintext output [" + name + "]").c_str(), output.size());
        }
        print_timing("plaintext-only total", plaintext_start);
        return 0;
    }

    // --- Client side: generate keys and encrypt input ---
    cout << "[Step 1/5] Setting up client (key generation)..." << endl;
    auto step_start = Clock::now();
    InferenceClient client(task_dir + "/client");
    client.setup();
    auto eval_ctx = client.export_eval_context();
    print_timing("step 1 setup/export context", step_start);

    cout << "[Step 2/5] Encrypting input..." << endl;
    step_start = Clock::now();
    auto encrypted_inputs = input_values.empty() ? client.encrypt(input_csvs) : client.encrypt_values(input_values);
    print_timing("step 2 encrypt input", step_start);

    // In actual scenarios, the client sends eval_ctx and encrypted_inputs to the server over the network.

    // --- Server side: load model and run encrypted inference ---
    cout << "[Step 3/5] Server loading model and importing context..." << endl;
    step_start = Clock::now();
    InferenceServer server(task_dir + "/server", use_gpu, gpu_device);
    server.import_eval_context(eval_ctx);
    server.load_model();
    print_timing("step 3 server import/load model", step_start);

    cout << "[Step 4/5] Running encrypted inference..." << endl;
    step_start = Clock::now();
    auto encrypted_outputs = server.evaluate(encrypted_inputs);
    print_timing("step 4 encrypted inference", step_start);

    // In actual scenarios, the server sends encrypted_outputs back to the client over the network.

    // --- Client side: decrypt result ---
    cout << "[Step 5/5] Decrypting result..." << endl;
    step_start = Clock::now();
    auto results = client.decrypt(encrypted_outputs);
    print_timing("step 5 decrypt result", step_start);
    cout << endl;

    // --- Display results ---
    cout << "========== Results ==========" << endl;
    for (auto& [name, result] : results) {
        print_double_message(result.output.data(), ("Encrypted output [" + name + "]").c_str(), result.num_outputs);
    }

    print_timing("total encrypted run", total_start);

    if (verify) {
        cout << endl;
        cout << "========== Plaintext Verification Run ==========" << endl;
        auto plaintext_outputs = input_values.empty() ? server.evaluate_plaintext(input_csvs)
                                                      : server.evaluate_plaintext_values(input_values);
        for (auto& [name, plaintext_output] : plaintext_outputs) {
            print_double_message(plaintext_output.data(), ("Plaintext output [" + name + "]").c_str(),
                                 plaintext_output.size());
        }

        for (auto& [name, result] : results) {
            auto pt_it = plaintext_outputs.find(name);
            if (pt_it == plaintext_outputs.end())
                continue;
            auto& plaintext_output = pt_it->second;

            cout << endl;
            cout << "========== Verification [" << name << "] ==========" << endl;
            int count = min(result.num_outputs, (int)plaintext_output.size());
            double max_abs_err = 0.0;
            double sum_abs_err = 0.0;
            double sum_sq_err = 0.0;
            int max_err_idx = 0;
            for (int i = 0; i < count; i++) {
                double err = result.output[i] - plaintext_output[i];
                double abs_err = fabs(err);
                sum_abs_err += abs_err;
                sum_sq_err += err * err;
                if (abs_err > max_abs_err) {
                    max_abs_err = abs_err;
                    max_err_idx = i;
                }
            }
            double avg_abs_err = count > 0 ? sum_abs_err / count : 0.0;
            double mse = count > 0 ? sum_sq_err / count : 0.0;

            cout << "Elements compared: " << count << endl;
            cout << fixed << setprecision(8);

            cout << endl;
            cout << setw(8) << "Index" << setw(18) << "Encrypted" << setw(18) << "Plaintext" << setw(18) << "Abs Error"
                 << endl;
            cout << string(62, '-') << endl;
            for (int i = 0; i < count; i++) {
                double abs_err = fabs(result.output[i] - plaintext_output[i]);
                cout << setw(8) << i << setw(18) << result.output[i] << setw(18) << plaintext_output[i] << setw(18)
                     << abs_err;
                if (abs_err > tolerance) {
                    cout << "  <-- EXCEEDS TOLERANCE";
                }
                cout << endl;
            }

            cout << string(62, '-') << endl;
            cout << "Max absolute error: " << max_abs_err << " (at index " << max_err_idx << ")" << endl;
            cout << "Avg absolute error: " << avg_abs_err << endl;
            cout << "MSE:                " << mse << endl;
            cout << "Tolerance:          " << tolerance << endl;
            cout << endl;

            if (max_abs_err > tolerance) {
                cout << "Result: FAIL" << endl;
                return 1;
            }
            cout << "Result: PASS" << endl;
        }
    }

    return 0;
}
