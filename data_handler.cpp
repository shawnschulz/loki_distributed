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

    void serialize_reverse(const std::string& filename, std::unordered_map<std::string, int> reverse_vocab) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr
                << "[ERROR] Failed to open file for writing."
                << std::endl;
            return;
        }
        for (auto const& p: reverse_vocab) {
            file << p.first << p.second;
        }
        return;
    }

    void serialize_forward(const std::string& filename, std::unordered_map<int, std::string> forward_vocab) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr
                << "[ERROR] Failed to open file for writing."
                << std::endl;
            return;
        }
        for (auto const& p: forward_vocab) {
            file << p.first << p.second;
        }
        return;
    }

    // Blegh just create some different function headers for the deserializtion methods we need
    void deserialize_forward(const std::string& filename) {
    }

    void deserialize_reverse(const std::string& filename) {
    }

    // read column from the parquet file and tokenize it using some tokenization scheme so we can actually
    // put that stuff in a feed forward neural network
    // Use byte pair encoding. for now we'll just single thread this, since it's unlikely to be the bottlenck
    // and it's actually quite tricky to parallelize the BPE algorithm
    // but in the future we'll want a kernel for this too
    void train_tokenizer(char* byte_characters, int input_size, int vocab_size) {
        // Construct the vocabulary
        // these will be serilaizable and deseriable for easy encoding and decoding
        std::unordered_map<int, std::string> forward_vocab;
        forward_vocab.insert({0, "<eos>"});
        forward_vocab.insert({1, "<bos>"});
        forward_vocab.insert({2, "<eoc>"});
        forward_vocab.insert({3, "<boc>"});
        std::unordered_map<std::string, int> reverse_vocab;
        std::unordered_map<std::string, int> frequencies;

        // find most frequent string of increasing sizes until vocabulary size reached
        int n = 1;
        int n_vocab_found = 0;
        while ( n_vocab_found < vocab_size ) {
            for (int i = 0; i < input_size; i++) {
                if ( n + i < input_size - 1 ) {
                    std::string string_window(std::next(byte_characters, i), std::next(byte_characters, n + i));
                    if (frequencies.find(string_window) == frequencies.end()) {
                        frequencies.insert({string_window, 1});
                        if (n = 1) {
                            forward_vocab.insert({forward_vocab.size() + 1, string_window});
                            n_vocab_found++;
                        }
                    }
                    else {
                        auto[ it, _ ] = frequencies.try_emplace(string_window, 0);
                        ++it->second;
                    }
                }
            }
            n++;
        }

        // use a heap to add vocab of frequency > 1 so we can sort
        std::priority_queue<std::pair<int, std::string>> vocab_queue;
        for (auto& it: frequencies) {
            if ((it.first.size() == 1) || (frequencies[it.first] > 1)) {
                vocab_queue.push(std::make_pair(it.second, it.first));
            }
        }


        while ((!vocab_queue.empty()) && (forward_vocab.size() < vocab_size)) {
            forward_vocab.insert({forward_vocab.size() + 1, vocab_queue.top().second});
            vocab_queue.pop();
        }

        // Construct the reverse_vocab from the forward_vocab
        int counter = 0;
        for (auto& it: forward_vocab) {
            reverse_vocab.insert({it.second, counter});
            counter++;
        }
        // Serialize the forward and reverse vocab
        serialize_forward(".model_data/forward_vocab.bin", forward_vocab);
        serialize_reverse(".model_data/reverse_vocab.bin", reverse_vocab);
    }

    // uses the lookup table created in train_tokenizer to tokenize input text
    // i think the easiest way to start with this is to scan across the input string with a pointer
    // until you get the max length sub string that you can find in the lookup table
    // and add that as a token to the output int[] array
    int* encode_tokens(std::vector<std::string>& forward_vocab, std::string input_string) {
    }

    // Uses the reverse lookup table created by train_tokenizer to detokenize output text
    // this one is easier, just go through the int[] array and put in whatever char is found
    // in the reverse lookup table
    void decode_tokens(std::unordered_map<std::string, int>& reverse_vocab, int* input_tokens) {}

    // Serialize Shard
    // header:
    // #0,12
    void serialize_shard() {}
    void deserialize_shard() {}

} // extern "C"
