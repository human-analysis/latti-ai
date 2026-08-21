/*
 * Persistent encrypted inference runner.
 *
 * Reads one JSON request per stdin line and emits one prefixed JSON response
 * per request. The client/server crypto context and model are initialized once.
 */

#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "interface/inference_client.h"
#include "interface/inference_server.h"

using namespace std;

namespace {

map<string, vector<double>> json_to_input_values(const nlohmann::ordered_json& root) {
    const auto& inputs_json = root.contains("inputs") ? root.at("inputs") : root;
    if (!inputs_json.is_object()) {
        throw runtime_error("Request must be {name: [values...]} or {id, inputs: {name: [values...]}}");
    }

    map<string, vector<double>> inputs;
    for (auto& [name, values_json] : inputs_json.items()) {
        if (!values_json.is_array()) {
            throw runtime_error("Input value must be an array for input: " + name);
        }
        auto& values = inputs[name];
        values.reserve(values_json.size());
        for (const auto& value : values_json) {
            values.push_back(value.get<double>());
        }
    }
    return inputs;
}

nlohmann::ordered_json plaintext_outputs_to_json(const map<string, vector<double>>& outputs) {
    nlohmann::ordered_json result = nlohmann::ordered_json::object();
    for (const auto& [name, values] : outputs) {
        result[name] = values;
    }
    return result;
}

nlohmann::ordered_json decrypted_outputs_to_json(const map<string, DecryptedOutput>& outputs) {
    nlohmann::ordered_json result = nlohmann::ordered_json::object();
    for (const auto& [name, output] : outputs) {
        vector<double> values(output.output.begin(), output.output.begin() + output.num_outputs);
        result[name] = values;
    }
    return result;
}

void emit_response(const nlohmann::ordered_json& response) {
    cout << "__LATTI_RESULT__ " << response.dump() << endl;
}

}  // namespace

int main(int argc, char* argv[]) {
    string task_dir;
    bool use_gpu = false;
    int gpu_device = 0;
    bool plaintext_only = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--gpu") == 0) {
            use_gpu = true;
        } else if (strcmp(argv[i], "--gpu-device") == 0 && i + 1 < argc) {
            gpu_device = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--task-dir") == 0 && i + 1 < argc) {
            task_dir = argv[++i];
        } else if (strcmp(argv[i], "--plaintext-only") == 0) {
            plaintext_only = true;
        }
    }

    if (task_dir.empty()) {
        cerr << "Usage: " << argv[0] << " --task-dir <path> [--plaintext-only] [--gpu] [--gpu-device <id>]" << endl;
        return 1;
    }

    try {
        unique_ptr<InferenceClient> client;
        unique_ptr<InferenceServer> server;

        if (plaintext_only) {
            cout << "[Batch] Loading plaintext model once..." << endl;
            server = make_unique<InferenceServer>(task_dir + "/server", use_gpu, gpu_device);
            server->load_model_no_context();
        } else {
            cout << "[Batch] Setting up client and server once..." << endl;
            client = make_unique<InferenceClient>(task_dir + "/client");
            client->setup();
            auto eval_ctx = client->export_eval_context();

            server = make_unique<InferenceServer>(task_dir + "/server", use_gpu, gpu_device);
            server->import_eval_context(eval_ctx);
            server->load_model();
        }
        cout << "[Batch] Ready." << endl;

        string line;
        while (getline(cin, line)) {
            if (line.empty()) {
                continue;
            }
            if (line == "quit" || line == "exit") {
                nlohmann::ordered_json response;
                response["ok"] = true;
                response["bye"] = true;
                emit_response(response);
                break;
            }

            nlohmann::ordered_json response;
            string request_id;
            try {
                auto request = nlohmann::ordered_json::parse(line);
                if (request.value("quit", false)) {
                    response["ok"] = true;
                    response["bye"] = true;
                    emit_response(response);
                    break;
                }
                if (request.contains("id")) {
                    request_id = request.at("id").get<string>();
                    response["id"] = request_id;
                }

                auto input_values = json_to_input_values(request);
                if (plaintext_only) {
                    auto outputs = server->evaluate_plaintext_values(input_values);
                    response["ok"] = true;
                    response["outputs"] = plaintext_outputs_to_json(outputs);
                } else {
                    auto encrypted_inputs = client->encrypt_values(input_values);
                    auto encrypted_outputs = server->evaluate(encrypted_inputs);
                    auto outputs = client->decrypt(encrypted_outputs);
                    response["ok"] = true;
                    response["outputs"] = decrypted_outputs_to_json(outputs);
                }
            } catch (const exception& exc) {
                if (!request_id.empty()) {
                    response["id"] = request_id;
                }
                response["ok"] = false;
                response["error"] = exc.what();
            }
            emit_response(response);
        }
    } catch (const exception& exc) {
        cerr << "[Batch] Fatal error: " << exc.what() << endl;
        return 1;
    }

    return 0;
}
