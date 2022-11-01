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

    void compute_union(int *image, int width, int height)
    {
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                if (image[x + y * width] == 1)
                {
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
    }

    std::set<std::vector<int>> compute_find(int *image, int width, int height, int minimum_pixel, bool from_union)
    {
        std::map<int, Box> boxes;

        int l;
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                if ((l = image[x + y * width]))
                {
                    if (from_union)
                        while (l > 0)
                            l = image[l - 2];
                    else
                        l -= 2;

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

    std::set<std::vector<int>> get_connected_components_uf(int *image, int width, int height, int minimum_pixel)
    {
        compute_union(image, width, height);
        auto ret = compute_find(image, width, height, minimum_pixel, true);
        return ret;
    }

    void init_label(int *image, int width, int height)
    {
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                image[x + y * width] *= (x + y * width + 2);
    }

    int get_min_neighbourg(int *image, int width, int height, int x, int y)
    {
        int min = image[y * width + x];
        for (int j = -1; j < 2; j++)
        {
            int cy = y + j;
            if (!(0 <= cy && cy < height))
                continue;

            for (int i = -1; i < 2; i++)
            {
                int cx = x + i;
                if (!(0 <= cx && cx < width))
                    continue;

                int pos = cx + cy * width;
                if (image[pos])
                    min = std::min(image[pos], min);
            }
        }

        return min;
    }

    bool propaged_label(int *image, int width, int height)
    {
        bool changed = false;

        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                int pos = x + y * width;
                if (image[pos] != 0)
                {
                    int min = get_min_neighbourg(image, width, height, x, y);
                    int cmin = image[pos];

                    if (min < cmin)
                    {
                        changed = true;
                        image[pos] = min;
                    }
                }
            }

        return changed;
    }

    std::set<std::vector<int>> get_connected_components_l(int *image, int width, int height, int minimum_pixel)
    {
        init_label(image, width, height);
        display_img_stdout(image, width, height);

        bool changed = true;
        while (changed)
        {
            changed = propaged_label(image, width, height);
            display_img_stdout(image, width, height);
        }

        // relabel(L); // Make label continuous
        // ==> Why ?

        auto ret = compute_find(image, width, height, minimum_pixel, false);

        return ret;
    }

    std::set<std::vector<int>> get_connected_components(int *image, int width, int height, enum mode_cc mode, int minimum_pixel)
    {
        if (mode == slide)
            return get_connected_components_l(image, width, height, minimum_pixel);
        else // if (mode == union_find)
            return get_connected_components_uf(image, width, height, minimum_pixel);
    }
}