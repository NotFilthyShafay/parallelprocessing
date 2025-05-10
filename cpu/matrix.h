class Matrix {
private:
    vector<float> data;
    size_t rows, cols;

public:
    Matrix(size_t r, size_t c) : rows(r), cols(c), data(r * c, 0.0f) {}
    
    float& operator()(size_t i, size_t j) {
        return data[i * cols + j];
    }
    
    const float& operator()(size_t i, size_t j) const {
        return data[i * cols + j];
    }
    
    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
    
    void randomize() {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<float> dis(0.0f, 1.0f);
        
        for (auto& val : data) {
            val = dis(gen);
        }
    }
};