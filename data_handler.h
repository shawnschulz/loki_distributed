// Read the "answer" column as strings and return a pointer to an array of C strings.
// Returns number of rows via out_rows.
const char** read_problem_column(const char* path, int* out_rows);

const char** read_answer_column(const char* path, int* out_rows);

const float* tokenize(const char* path, const char* column_name, int* out_rows, int vocabulary_size);

const float* detokenize();
