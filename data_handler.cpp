// parquet_bridge.cpp
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <queue>
#include <fstream>
#include <iostream>
#include <set>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

extern "C" {

    // Read the "answer" column as strings and return a pointer to an array of C strings.
    // Returns number of rows via out_rows.
//    const char** read_problem_column(const char* path, int* out_rows) {
//        auto infile = arrow::io::ReadableFile::Open(path).ValueOrDie();
//        std::unique_ptr<parquet::arrow::FileReader> reader;
//        parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader);
//
//        std::shared_ptr<arrow::Table> table;
//        reader->ReadTable(&table);
//
//        auto col = table->GetColumnByName("problem");
//        if (!col) {
//            *out_rows = 0;
//            return nullptr;
//        }
//
//        // Flatten to chunks (assuming simple column)
//        std::vector<const char*>* cstrings = new std::vector<const char*>();
//        for (int i = 0; i < col->num_chunks(); ++i) {
//            auto chunk = std::static_pointer_cast<arrow::StringArray>(col->chunk(i));
//            for (int j = 0; j < chunk->length(); ++j) {
//                cstrings->push_back(chunk->GetString(j).c_str());
//            }
//        }
//
//        *out_rows = static_cast<int>(cstrings->size());
//        return cstrings->data();  // pointer to array of C strings
//    }
//
//    const char** read_answer_column(const char* path, int* out_rows) {
//        auto infile = arrow::io::ReadableFile::Open(path).ValueOrDie();
//        std::unique_ptr<parquet::arrow::FileReader> reader;
//        parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader);
//
//        std::shared_ptr<arrow::Table> table;
//        reader->ReadTable(&table);
//
//        auto col = table->GetColumnByName("expected_answer");
//        if (!col) {
//            *out_rows = 0;
//            return nullptr;
//        }
//
//        // Flatten to chunks (assuming simple column)
//        std::vector<const char*>* cstrings = new std::vector<const char*>();
//        for (int i = 0; i < col->num_chunks(); ++i) {
//            auto chunk = std::static_pointer_cast<arrow::StringArray>(col->chunk(i));
//            for (int j = 0; j < chunk->length(); ++j) {
//                cstrings->push_back(chunk->GetString(j).c_str());
//            }
//        }
//
//        *out_rows = static_cast<int>(cstrings->size());
//        return cstrings->data();  // pointer to array of C strings
//    }
    std::vector<std::string> load_fineweb_vocab_training_data(std::string path) {
        // This is just to test our vocab training and tokenization, eventually we can
        // refactor our vocab training to be finetunable
        std::vector<std::string> ret;
        for (int i = 1; i < 2; i++) {
            std::string filename = path + "/ultrafineweb-en-part-000" + std::to_string(i) + "-of-2048.parquet";
            std::cout << "[INFO] Loading vocab data, currently loading: " << filename << std::endl;

            auto infile_res = arrow::io::ReadableFile::Open(filename);
            auto infile = *infile_res;

            auto reader_res = parquet::arrow::OpenFile(infile, arrow::default_memory_pool());
            std::unique_ptr<parquet::arrow::FileReader> reader = std::move(*reader_res);

            std::shared_ptr<arrow::Table> table;
            reader->ReadTable(&table);

            auto col = table->GetColumnByName("content");


            for (int i = 0; i < col->num_chunks(); ++i) {
                auto chunk = std::static_pointer_cast<arrow::StringArray>(col->chunk(i));
                for (int j = 0; j < chunk->length(); ++j) {
                    ret.push_back(chunk->GetString(j));
                }
            }
        }
        return ret;
    }

    // Sloooow to train, but really shouldn't need too huge of input data to train a reasonable tokenizer
    // we'll use off the shelf json solutions but this provides a means to build them for an E2E engine
    // later
//    void train_tokenizer(std::string path_name, std::vector<std::string> training_data, int vocab_size) {
//        // Construct the vocabulary
//        // these will be serilaizable and deseriable for easy encoding and decoding
//        std::unordered_map<int, std::string> forward_vocab;
//        forward_vocab.insert({0, "<eos>"});
//        forward_vocab.insert({1, "<bos>"});
//        forward_vocab.insert({2, "<eoc>"});
//        forward_vocab.insert({3, "<boc>"});
//        std::unordered_map<std::string, int> reverse_vocab;
//        std::unordered_map<std::string, std::pair<int, int>> frequencies;
//
//        // find most frequent string of increasing sizes until vocabulary size reached
//        int n = 1;
//        int n_vocab_found = 0;
//        std::priority_queue<std::pair<int, std::string>> frequency_queue;
//        std::string string_pair;
//        // Add single characters
//        for (int i = 0; i < training_data.size(); i++) {
//            if (frequencies.find(training_data.at(i)) == frequencies.end()) {
//                forward_vocab.insert({forward_vocab.size() - 1, training_data.at(i)});
//            }
//        }
//        while ( n_vocab_found < vocab_size ) {
//            std::cout << "[INFO] Constructing vocabulary, n_vocab_found: " << n_vocab_found << "\n";
//            for (int i = 0; i < training_data.size() - 1; i++) {
//                string_pair = training_data.at(i) + training_data.at( i + 1 );
//                if (frequencies.find(string_pair) == frequencies.end()) {
//                    frequencies.insert({string_pair, std::make_pair(1, i)});
//                }
//                else {
//                    auto[ it, _ ] = frequencies.try_emplace(string_window, 0);
//                    ++it->second->first;
//                    }
//            }
//            std::pair<std::string, std::pair<int, int>> max_pair = find_max_freq(frequencies);
//            forward_vocab.insert({forward_vocab.size(), max_pair.first});
//            training_data.at(max_pair.second.second) = max_pair.first;
//            training_data.erase(training_data.begin() + max_pair.second.second + 1);
//            frequencies.clear();
//        }
//        std::free(&training_string);
//        // Construct the reverse_vocab from the forward_vocab
//        int counter = 0;
//        for (auto& it: forward_vocab) {
//            reverse_vocab.insert({it.second, counter});
//            counter++;
//        }
//        // Serialize the forward and reverse vocab
//        serialize_forward(path_name + "/forward_vocab.bin", forward_vocab);
//        serialize_reverse(path_name + "/reverse_vocab.bin", reverse_vocab);
//    }
//    };

    // New method: use a tokenizer.json to construct the map
    std::unordered_map<std::string, int> deserialize_tokenizer(const std::string& filename) {
        std::ifstream f(filename);
        json data = json::parse(f);
        std::unordered_map<std::string, int> forward_vocab;
        for (auto& [key, value] : data["model"]["vocab"].items()) {
            forward_vocab.insert({key, value});
        }
        return forward_vocab;
    }
    std::unordered_map<int, std::string> make_reverse_tokenizer(std::unordered_map<std::string, int> forward_tokenizer) {
        std::unordered_map<int, std::string> reverse_tokenizer;
        for (auto& pair: forward_tokenizer) {
            reverse_tokenizer.insert({pair.second, pair.first});
        }
        return reverse_tokenizer;
    }

    // uses the lookup table created in train_tokenizer to tokenize input text
    // i think the easiest way to start with this is to scan across the input string with a pointer
    // until you get the max length sub string that you can find in the lookup table
    // and add that as a token to the output int[] array
    std::vector<int> encode_tokens(const std::unordered_map<std::string, int>& forward_tokenizer, const std::string& input_string) {
        std::vector<int> ret;
        int i = 0;
        int j = 0;
        while (i < input_string.size()) {
            while ((forward_tokenizer.find(input_string.substr(i, (j - i)  + 1)) != forward_tokenizer.end()) && (j < input_string.size())) {
                j++;
            }
            if (forward_tokenizer.find(input_string.substr(i, (j - i)))!= forward_tokenizer.end()) {
                ret.push_back(forward_tokenizer.at(input_string.substr(i, (j - i))));
            }
            else {
                ret.push_back(50280);
            }
            i = j;
        }
        return ret;
    }

    // Uses the reverse lookup table created by train_tokenizer to detokenize output text
    // this one is easier, just go through the int[] array and put in whatever char is found
    // in the reverse lookup table
    std::vector<std::string> decode_tokens(std::unordered_map<int, std::string>& reverse_tokenizer, std::vector<int> tokens) {
        std::vector<std::string> ret;
        for (auto &n: tokens) {
            if (reverse_tokenizer.find(n) != reverse_tokenizer.end()) {
                ret.push_back(reverse_tokenizer.at(n));
            }
            else {
                ret.push_back("[UNK]");
            }
        }
        return ret;
    }

    // Serialize Shard
    // header:
    // #0,12
    void serialize_shard() {}
    void deserialize_shard() {}

} // extern "C"
