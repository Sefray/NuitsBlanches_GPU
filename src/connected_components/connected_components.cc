#include "pipeline.hh"

#include <map>

namespace cpu
{
    struct Box
    {
        int xmin;
        int ymin;
        int xmax;
        int ymax;

        int size;
    };

    int compute_union(int *image, int width, int height)
    {
        int labbeleds = 0;
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                if (image[x + y * width] == 1)
                {
                    labbeleds++;

                    int nl = y - 1 >= 0 ? std::abs(image[x + (y - 1) * width]) : 0;
                    int ol = x - 1 >= 0 ? std::abs(image[x - 1 + y * width]) : 0;

                    int cl;
                    if (nl && ol)
                        cl = std::min(ol, nl);
                    else if (nl || ol)
                        cl = std::max(ol, nl);
                    else
                        cl = -(x + y * width + 2);

                    image[x + y * width] = cl;

                    // Union
                    if (y - 1 >= 0 && nl && nl != std::abs(cl))
                        image[nl - 2] = std::abs(cl);
                    if (x - 1 >= 0 && ol && ol != std::abs(cl))
                        image[ol - 2] = std::abs(cl);
                }

        return labbeleds;
    }

    std::set<std::vector<int>> compute_find(int *image, int width, int height, int labels, int minimum_pixel)
    {
        std::map<int, Box> boxes;

        int l;
        for (int y = 0; y < height && labels; y++)
            for (int x = 0; x < width && labels; x++)
            {
                if ((l = image[x + y * width]))
                {
                    while (l > 0)
                        l = image[l - 2];

                    if (boxes.find(l) != boxes.end())
                    {
                        auto &box = boxes.at(l);

                        box.xmin = std::min(x, box.xmin);
                        box.ymin = std::min(y, box.ymin);
                        box.xmax = std::max(x, box.xmax);
                        box.ymax = std::max(y, box.ymax);

                        box.size++;
                    }
                    else
                    {
                        Box box = {.xmin = x, .ymin = y, .xmax = x, .ymax = y, .size = 1};
                        boxes[l] = box;
                    }

                    labels--;
                }
            }

        std::set<std::vector<int>> ret;

        for (auto &box : boxes)
        {
            if (box.second.size > minimum_pixel)
                ret.insert({box.second.xmin, box.second.ymin, box.second.xmax - box.second.xmin + 1, box.second.ymax - box.second.ymin + 1});
        }

        return ret;
    }

    std::set<std::vector<int>> get_connected_components(int *image, int width, int height, int minimum_pixel)
    {
        auto labbeleds = compute_union(image, width, height);
        auto ret = compute_find(image, width, height, labbeleds, minimum_pixel);
        return ret;
    }
}