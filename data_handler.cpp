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
    };
    std::vector<std::string> flatten_training_data(std::vector<std::string>& input) {
        std::vector<std::string> flat_training_data;
        std::string new_string;
        for(int i = 0; i < 50000; i++) {
            std::cout << "[INFO] Flattening vocabulary: " << i << "\n";
            new_string = input.back();
            input.pop_back();
            for (auto s: new_string) {
                flat_training_data.push_back(std::to_string(s));
            }
        }
        return flat_training_data;
    }

    void serialize_reverse(const std::string& filename, std::unordered_map<std::string, int> reverse_vocab) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr
                << "[ERROR] Failed to open file for writing."
                << std::endl;
            return;
        }
        for (auto const& p: reverse_vocab) {
            int length = p.first.size();
            file.write(reinterpret_cast<const char*>(&length), sizeof(length));
            file.write(p.first.data(), length);
            file.write(reinterpret_cast<const char*>(&p.second), sizeof(p.second));
        }
        file.close();
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
            int length = p.second.size();
            file.write(reinterpret_cast<const char*>(&p.first), sizeof(p.first));
            file.write(reinterpret_cast<const char*>(&length), sizeof(length));
            file.write(p.second.data(), length);
        }
        file.close();
        return;
    }

    // Blegh just create some different function headers for the deserializtion methods we need
    std::unordered_map<int, std::string> deserialize_forward(const std::string& filename) {
        std::unordered_map<int, std::string> forward_vocab;
        std::ifstream file(filename, std::ios::binary);
        int key;
        int length;
        if (!file.is_open()) {
            std::cerr
                << "[ERROR] Failed to open file for reading."
                << std::endl;
            return forward_vocab;
        }
        while (file && file.peek() != EOF) {
            file.read(reinterpret_cast<char*>(&key), sizeof(key));
            file.read(reinterpret_cast<char*>(&length), sizeof(length));
            std::string value(length, '\0');
            file.read(&value[0], length);
            forward_vocab.insert({key, value});
        }
        return forward_vocab;
    }

    std::unordered_map<std::string, int> deserialize_reverse(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        std::unordered_map<std::string, int> reverse_vocab;
        std::string key;
        int value;
        while (file && file.peek() != EOF) {
            int length;
            file.read(reinterpret_cast<char*>(&length), sizeof(length));
            std::string key(length, '\0');
            file.read(&key[0], length);
            file.read(reinterpret_cast<char*>(&value), sizeof(value));
            reverse_vocab.insert({key, value});
        }
        return reverse_vocab;
    }
    std::pair<std::string, std::pair<int, int>> find_max_freq(
        std::unordered_map<std::string, std::pair<int, int>> const &x)
    {
        return *std::max_element(x.begin(), x.end(),
                                [](const std::pair<std::string, std::pair<int, int>> &p1,
                                    const std::pair<std::string, std::pair<int, int>> &p2)
                                {
                                    return p1.second.first < p2.second.first;
                                });
    }

    // read column from the parquet file and tokenize it using some tokenization scheme so we can actually
    // put that stuff in a feed forward neural network
    // Use byte pair encoding. for now we'll just single thread this, since it's unlikely to be the bottlenck
    // and it's actually quite tricky to parallelize the BPE algorithm
    // but in the future we'll want a kernel for this too
    void train_tokenizer(std::string path_name, std::vector<std::string>& training_data, int vocab_size) {
        // Construct the vocabulary
        // these will be serilaizable and deseriable for easy encoding and decoding
        std::unordered_map<int, std::string> forward_vocab;
        std::unordered_map<std::string, int> reverse_vocab;
        reverse_vocab.insert({"<eos>",0});
        reverse_vocab.insert({"<bos>",1});
        reverse_vocab.insert({"<eoc>",2});
        reverse_vocab.insert({"<boc>",3});
        std::unordered_map<std::string, std::pair<int, int>> frequencies;

        // find most frequent string of increasing sizes until vocabulary size reached
        int n = 1;
        int n_vocab_found = 0;
        std::priority_queue<std::pair<int, std::string>> frequency_queue;
        std::string string_pair;

        // Add single characters
        std::cout << "[INFO] Adding single characters" << "\n";
        for (int i = 0; i < training_data.size(); i++) {
            if (reverse_vocab.find(training_data.at(i)) == reverse_vocab.end()) {
                reverse_vocab.insert({training_data.at(i), reverse_vocab.size() - 1});
                frequencies.insert({training_data.at(i), std::make_pair(1, i)});
            } else {
                frequencies.at(training_data.at(i)).first += 1;
            }
        }

        std::cout << "[INFO] Adding inital pairs" << "\n";
        // Add initial pairs
        for (int i = 0; i < training_data.size() - 1; i++) {
            string_pair = training_data.at(i) + training_data.at( i + 1 );
            if (frequencies.find(string_pair) == frequencies.end()) {
                frequencies.insert({string_pair, std::make_pair(1, i)});
            }
            else {
                frequencies.at(string_pair).first += 1;
                }
        }
        std::pair<std::string, std::pair<int, int>> max_pair;
        std::string new_string;
        int frequency;
        int data_i;
        std::vector<std::string> temporary_vector;
        while ( reverse_vocab.size() < vocab_size ) {
            std::cout << "[INFO] Constructing vocabulary, n_vocab_found: " << reverse_vocab.size() << "\n";
            max_pair = find_max_freq(frequencies);
            new_string = max_pair.first;
            frequency = max_pair.second.first;
            data_i = max_pair.second.second;

            // decrement the old pairs
            frequencies.at(training_data.at(data_i)).first -= 1;
            frequencies.at(training_data.at(data_i + 1)).first -= 1;

            // Update the training_data and vocabulary with the new pair
            reverse_vocab.insert({max_pair.first, reverse_vocab.size()});
            // Merge all instances ofo the pair in the training data, calcualting the
            // new frequency as we go. this is faster but more memory intensive to use
            // extra space
            for (int i = 0; i < training_data.size() - 1; i++) {
                if (training_data.at(data_i) + training_data.at(data_i + 1) == new_string) {
                    temporary_vector.push_back(new_string);
                } else{
                    temporary_vector.push_back(training_data.at(data_i));
                    temporary_vector.push_back(training_data.at(data_i + 1));
                }
            }
            training_data = temporary_vector;
        }
        training_data.clear();
        training_data.shrink_to_fit();
        // Construct the reverse_vocab from the forward_vocab
        int counter = 0;
        for (auto& it: reverse_vocab) {
            forward_vocab.insert({counter, it.first});
            counter++;
        }
        // Serialize the forward and reverse vocab
        serialize_forward(path_name + "/forward_vocab.bin", forward_vocab);
        serialize_reverse(path_name + "/reverse_vocab.bin", reverse_vocab);
    }

    // uses the lookup table created in train_tokenizer to tokenize input text
    // i think the easiest way to start with this is to scan across the input string with a pointer
    // until you get the max length sub string that you can find in the lookup table
    // and add that as a token to the output int[] array
    int* encode_tokens(std::unordered_map<int, std::string>& forward_vocab, std::string input_string) {
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
