#include <iostream>
#include <iomanip>
#include "tl/tl.hpp"


int main() {
    // 1. Setup: Create a 4D "Batch" of images
    // Dimensions: [Batch Size, Channels, Height, Width]
    // Example: 2 images, 3 color channels (RGB), 4x4 pixels
    std::vector<int> shape = {2, 3, 4, 4};
    
    std::cout << "--- Initializing 4D Image Batch ---" << std::endl;
    auto batch = tl::ones<float>(shape);

    // 2. Advanced Indexing: Modify a specific "pixel" 
    // Image 1, Green Channel (1), Row 2, Col 3
    batch[1][1][2][3] = 5.5f;
    batch[0][0][0][0] = 0.1f;

    // 3. Scalar Math: Normalize the batch (e.g., divide by 255.0)
    std::cout << "\n--- Normalizing Batch (Scalar Division) ---" << std::endl;
    auto normalized = batch * 0.5f; 

    // 4. Element-wise Math: Create a "Filter" and apply it
    // Let's create a mask of the same shape
    auto mask = tl::zeros<float>(shape);
    mask[1][1][2][3] = 2.0f; // Only let this one pixel through with a gain
    mask[0][0][0][0] = 10.0f;

    std::cout << "--- Applying Mask (Element-wise Multiplication) ---" << std::endl;
    auto result = normalized * mask;

    // 5. Advanced Usage: Chaining operations and Slicing (if implemented)
    // Here we use the + operator and scalar addition together
    auto final_output = (result + 1.0f) - tl::ones<float>(shape);

    // 6. Verification: Accessing data through the View system
    std::cout << "\nChecking specific values:" << std::endl;
    std::cout << "Original pixel at [1,1,2,3]: " << batch[1][1][2][3] << std::endl;
    std::cout << "Processed pixel at [1,1,2,3]: " << final_output[1][1][2][3] << " (Expected 5.5)" << std::endl;

    // 7. Printing: Using the recursive N-D printer
    // We'll print just a small slice conceptually
    std::cout << "\n--- Final Tensor Structure (Recursive Print) ---" << std::endl;
    tl::print(final_output);

    return 0;
}