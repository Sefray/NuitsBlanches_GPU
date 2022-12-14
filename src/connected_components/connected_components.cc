#include "pipeline.hh"

#include <map>

namespace cpu
{
  int relabel(int* image, int width, int height)
  {
    int r = 0;
    for (int y = 0; y < height; y++)
      for (int x = 0; x < width; x++)
      {
        int p = x + y * width;
        if (image[p] < 0)
        {
          r++;
          image[p] = -r;
        }
      }

    return r;
  }

  std::set<std::vector<int>> compute_find(int* image, int* image_values, int width, int height, int high_pick_threshold,
                                          int minimum_pixel, int nb_boxes)
  {
    Box* boxes = static_cast<Box*>(std::calloc(nb_boxes + 1, sizeof(Box)));

    int l;
    for (int y = 0; y < height; y++)
      for (int x = 0; x < width; x++)
      {
        if ((l = image[x + y * width]))
        {
          while (l > 0)
            l = image[l - 2];

          auto& box = boxes[-l];
          if (box.size != 0)
          {
            box.xmin = std::min(x, box.xmin);
            box.ymin = std::min(y, box.ymin);
            box.xmax = std::max(x, box.xmax);
            box.ymax = std::max(y, box.ymax);
          }
          else
          {
            box.xmin = x;
            box.ymin = y;
            box.xmax = x;
            box.ymax = y;
          }

          box.high_pick = std::max(box.high_pick, image_values[x + y * width]);
          box.size++;
        }
      }

    std::set<std::vector<int>> ret;

    for (int i = 1; i < nb_boxes + 1; i++)
    {
      auto& box = boxes[i];
      if (box.size >= minimum_pixel && box.high_pick >= high_pick_threshold)
        ret.insert({box.xmin, box.ymin, box.xmax - box.xmin + 1, box.ymax - box.ymin + 1});
    }

    std::free(boxes);

    return ret;
  }

  void init_label(int* image, int width, int height)
  {
    for (int y = 0; y < height; y++)
      for (int x = 0; x < width; x++)
        image[x + y * width] *= -(x + y * width + 2);
  }

  int get_min_neighbor(int* image, int width, int height, int x, int y)
  {
    int min = std::abs(image[y * width + x]);
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
          min = std::min(std::abs(image[pos]), min);
      }
    }

    return min;
  }

  bool propaged_label(int* image, int width, int height)
  {
    bool changed = false;

    for (int p = 0; p < width * height;)
    {
      for (int s = 0; s < 8; s++)
      {
        int x = p % width;
        int y = p / width;

        if (image[p] != 0)
        {
          int min  = get_min_neighbor(image, width, height, x, y);
          int cmin = std::abs(image[p]);

          if (min < cmin)
          {
            changed  = true;
            image[p] = min;
          }
        }

        p++;
        x++;
        if (!(x %= width))
        {
          y++;
          if (!(y %= height))
            break;
        }
      }
    }

    return changed;
  }

  std::set<std::vector<int>> get_connected_components(int* image, int* image_values, int width, int height,
                                                      int high_pick_threshold, int minimum_pixel)
  {
    init_label(image, width, height);

    bool changed = true;
    while (changed)
      changed = propaged_label(image, width, height);

    int nb_label = relabel(image, width, height);

    auto ret = compute_find(image, image_values, width, height, high_pick_threshold, minimum_pixel, nb_label);

    return ret;
  }
} // namespace cpu