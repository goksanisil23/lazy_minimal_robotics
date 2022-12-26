#include <algorithm>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

namespace oakd
{
class OakdImageReader
{
  public:
    OakdImageReader(const char *imageDirectoryPath)
    {
        // Collect the paths of the JPEG files in the specified directory

        for (const auto &entry : std::filesystem::directory_iterator(imageDirectoryPath))
        {
            if (entry.is_regular_file() &&
                ((entry.path().extension() == ".png") || (entry.path().extension() == ".jpg")))
            {
                image_files.push_back(entry.path());
            }
        }

        // Sort the paths alphabetically
        std::sort(image_files.begin(), image_files.end());
        image_file_iterator = image_files.begin();
    }

    bool GetNextImage(cv::Mat &image)
    {
        if (image_file_iterator != image_files.end())
        {
            image = cv::imread(*image_file_iterator);
            if (image.empty())
            {
                std::cout << "Error: Failed to load image " << *image_file_iterator << std::endl;
                return false;
            }
            image_file_iterator++;
            return true;
        }
        else
        {
            return false;
        }
    }

    std::vector<std::filesystem::path>           image_files;
    std::vector<std::filesystem::path>::iterator image_file_iterator;
};

} // namespace oakd