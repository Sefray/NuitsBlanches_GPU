#include "pipeline.hh"

#include <map>

namespace cpu
{
  void compute_union(int* image, int width, int height)
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

  std::set<std::vector<int>> compute_find(int* image, int width, int height, int minimum_pixel, int nb_boxes)
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
          box.size++;
        }
      }

    std::set<std::vector<int>> ret;

    for (int i = 1; i < nb_boxes + 1; i++)
    {
      auto& box = boxes[i];
      if (box.size > minimum_pixel)
        ret.insert({box.xmin, box.ymin, box.xmax - box.xmin + 1, box.ymax - box.ymin + 1});
    }

    std::free(boxes);

    return ret;
  }

  std::set<std::vector<int>> get_connected_components_uf(int* image, int width, int height, int minimum_pixel)
  {
    compute_union(image, width, height);
    int  nb_label = relabel(image, width, height);
    auto ret      = compute_find(image, width, height, minimum_pixel, nb_label);
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

    // for (int y = 0; y < height; y++)
    //   for (int x = 0; x < width; x++)
    //   {

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

  std::set<std::vector<int>> get_connected_components_l(int* image, int width, int height, int minimum_pixel)
  {
    init_label(image, width, height);

    bool changed = true;
    while (changed)
      changed = propaged_label(image, width, height);

    int nb_label = relabel(image, width, height);

    auto ret = compute_find(image, width, height, minimum_pixel, nb_label);

    return ret;
  }

  std::set<std::vector<int>> get_connected_components(int* image, int width, int height, enum mode_cc mode,
                                                      int minimum_pixel)
  {
    if (mode == slide)
      return get_connected_components_l(image, width, height, minimum_pixel);
    else // if (mode == union_find)
      return get_connected_components_uf(image, width, height, minimum_pixel);
  }
} // namespace cpu