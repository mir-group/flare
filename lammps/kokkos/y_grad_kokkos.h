#ifndef Y_GRAD_KOKKOS_H
#define Y_GRAD_KOKKOS_H

#include <vector>
#include <complex>
#include <Kokkos_Core.hpp>



// TODO: parallelize over components if bottleneck
template<class ViewType>
KOKKOS_INLINE_FUNCTION
void get_Y_kokkos(int i, int j, ViewType &Y, const double x, const double y, const double z, const int l) {

  const double Pi = 3.14159265358979323846;
  const double two_pi = 2. * Pi;
  const double c1 = std::sqrt(3 / Pi);
  const double c2 = std::sqrt(15 / Pi);
  const double c3 = std::sqrt(35 / two_pi);
  const double c4 = std::sqrt(5 / Pi);
  const double c5 = std::sqrt(105 / Pi);
  const double c6 = std::sqrt(21 / two_pi);
  const double c7 = std::sqrt(7 / Pi);
  const double c8 = std::sqrt(5 / two_pi);
  const double c9 = std::sqrt(35 / Pi);
  const double c10 = std::sqrt(77 / two_pi);
  const double c11 = std::sqrt(385 / Pi);
  const double c12 = std::sqrt(385 / two_pi);

  const Kokkos::complex<double> cc1 = Kokkos::complex<double>(0, 1);
  const Kokkos::complex<double> cc2 = Kokkos::complex<double>(0, -1);

  unsigned int counter = 0;

  const double r2 = x * x + y * y + z * z;
  const double r2s = std::sqrt(r2);
  const double r3 = std::pow(r2, 1.5);
  const double r4 = r2 * r2;
  const double r5 = std::pow(r2, 2.5);
  const double r6 = r3 * r3;
  const double r7 = std::pow(r2, 3.5);
  const double r8 = r4 * r4;
  const double r9 = r4 * r5;
  const double r10 = r5 * r5;
  const double r11 = r5 * r6;
  const double r12 = r6 * r6;
  const double x2 = x * x;
  const double y2 = y * y;
  const double z2 = z * z;
  const double x3 = x * x * x;
  const double y3 = y * y * y;
  const double x4 = x2 * x2;
  const double y4 = y2 * y2;
  const double z4 = z2 * z2;
  const double x5 = x4 * x;
  const double y5 = y4 * y;
  const double x6 = x3 * x3;
  const double y6 = y3 * y3;
  const double z6 = z2 * z2 * z2;
  const double x8 = std::pow(x, 8);
  const double y8 = std::pow(y, 8);
  const double z8 = std::pow(z, 8);
  const double x2py21 = x2 + y2;
  const double x2py2 = std::pow(x2 + y2, 2);
  const double x2py3 = std::pow(x2 + y2, 3);
  const double x2py4 = x2py2 * x2py2;
  const double xy = x * y;
  const double xy2 = x * y2;

  { // l_counter == 0
    Y(i,j,counter,0) = 1 / (2. * std::sqrt(Pi));
    Y(i,j,counter,1) = 0;
    Y(i,j,counter,2) = 0;
    Y(i,j,counter,3) = 0;
    counter++;
  }
  if (l == 0)
    return;

  { // l_counter == 1
    Y(i,j,counter,0) = (c1 * y) / (2. * r2s);
    Y(i,j,counter,1) = -(c1 * x * y) / (2. * r3);
    Y(i,j,counter,2) = (c1 * (x2 + z2)) / (2. * r3);
    Y(i,j,counter,3) = -(c1 * y * z) / (2. * r3);
    counter++;

    Y(i,j,counter,0) = (c1 * z) / (2. * r2s);
    Y(i,j,counter,1) = -(c1 * x * z) / (2. * r3);
    Y(i,j,counter,2) = -(c1 * y * z) / (2. * r3);
    Y(i,j,counter,3) = (c1 * x2py21) / (2. * r3);
    counter++;

    Y(i,j,counter,0) = (c1 * x) / (2. * r2s);
    Y(i,j,counter,1) = (c1 * (y2 + z2)) / (2. * r3);
    Y(i,j,counter,2) = -(c1 * x * y) / (2. * r3);
    Y(i,j,counter,3) = -(c1 * x * z) / (2. * r3);
    counter++;
  }
  if (l == 1)
    return;

  { // l_counter == 2
    Y(i,j,counter,0) = (c2 * x * y) / (2. * (r2));
    Y(i,j,counter,1) = (c2 * y * (-x2 + y2 + z2)) / (2. * r4);
    Y(i,j,counter,2) = (c2 * x * (x2 - y2 + z2)) / (2. * r4);
    Y(i,j,counter,3) = -((c2 * x * y * z) / r4);
    counter++;

    Y(i,j,counter,0) = (c2 * y * z) / (2. * (r2));
    Y(i,j,counter,1) = -((c2 * x * y * z) / r4);
    Y(i,j,counter,2) = (c2 * z * (x2 - y2 + z2)) / (2. * r4);
    Y(i,j,counter,3) = (c2 * y * (x2 + y2 - z2)) / (2. * r4);
    counter++;

    Y(i,j,counter,0) = -(c4 * (x2 + y2 - 2 * z2)) / (4. * (r2));
    Y(i,j,counter,1) = (-3 * c4 * x * z2) / (2. * r4);
    Y(i,j,counter,2) = (-3 * c4 * y * z2) / (2. * r4);
    Y(i,j,counter,3) = (3 * c4 * x2py21 * z) / (2. * r4);
    counter++;

    Y(i,j,counter,0) = (c2 * x * z) / (2. * (r2));
    Y(i,j,counter,1) = (c2 * z * (-x2 + y2 + z2)) / (2. * r4);
    Y(i,j,counter,2) = -((c2 * x * y * z) / r4);
    Y(i,j,counter,3) = (c2 * x * (x2 + y2 - z2)) / (2. * r4);
    counter++;

    Y(i,j,counter,0) = (c2 * (x - y) * (x + y)) / (4. * (r2));
    Y(i,j,counter,1) = (c2 * x * (2 * y2 + z2)) / (2. * r4);
    Y(i,j,counter,2) = -(c2 * y * (2 * x2 + z2)) / (2. * r4);
    Y(i,j,counter,3) = -(c2 * (x - y) * (x + y) * z) / (2. * r4);
    counter++;
  }
  if (l == 2)
    return;

  { // l_counter == 3
    Y(i,j,counter,0) = -(c3 * y * (-3 * x2 + y2)) / (4. * r3);
    Y(i,j,counter,1) = (-3 * c3 * x * y * (x2 - 3 * y2 - 2 * z2)) / (4. * r5);
    Y(i,j,counter,2) = (3 * c3 * (x4 - y2 * z2 + x2 * (-3 * y2 + z2))) / (4. * r5);
    Y(i,j,counter,3) = (3 * c3 * y * (-3 * x2 + y2) * z) / (4. * r5);
    counter++;

    Y(i,j,counter,0) = (c5 * x * y * z) / (2. * r3);
    Y(i,j,counter,1) = (c5 * y * z * (-2 * x2 + y2 + z2)) / (2. * r5);
    Y(i,j,counter,2) = (c5 * x * z * (x2 - 2 * y2 + z2)) / (2. * r5);
    Y(i,j,counter,3) = (c5 * x * y * (x2 + y2 - 2 * z2)) / (2. * r5);
    counter++;

    Y(i,j,counter,0) = -(c6 * y * (x2 + y2 - 4 * z2)) / (4. * r3);
    Y(i,j,counter,1) = (c6 * x * y * (x2 + y2 - 14 * z2)) / (4. * r5);
    Y(i,j,counter,2) =
        -(c6 * (x4 + 11 * y2 * z2 - 4 * z4 + x2 * (y2 - 3 * z2))) / (4. * r5);
    Y(i,j,counter,3) = (c6 * y * z * (11 * x2py21 - 4 * z2)) / (4. * r5);
    counter++;

    Y(i,j,counter,0) = (c7 * z * (-3 * x2py21 + 2 * z2)) / (4. * r3);
    Y(i,j,counter,1) = (3 * c7 * x * z * (x2 + y2 - 4 * z2)) / (4. * r5);
    Y(i,j,counter,2) = (3 * c7 * y * z * (x2 + y2 - 4 * z2)) / (4. * r5);
    Y(i,j,counter,3) = (-3 * c7 * x2py21 * (x2 + y2 - 4 * z2)) / (4. * r5);
    counter++;

    Y(i,j,counter,0) = -(c6 * x * (x2 + y2 - 4 * z2)) / (4. * r3);
    Y(i,j,counter,1) =
        -(c6 * (y4 - 3 * y2 * z2 - 4 * z4 + x2 * (y2 + 11 * z2))) / (4. * r5);
    Y(i,j,counter,2) = (c6 * x * y * (x2 + y2 - 14 * z2)) / (4. * r5);
    Y(i,j,counter,3) = (c6 * x * z * (11 * x2py21 - 4 * z2)) / (4. * r5);
    counter++;

    Y(i,j,counter,0) = (c5 * (x - y) * (x + y) * z) / (4. * r3);
    Y(i,j,counter,1) = -(c5 * x * z * (x2 - 5 * y2 - 2 * z2)) / (4. * r5);
    Y(i,j,counter,2) = (c5 * y * z * (-5 * x2 + y2 - 2 * z2)) / (4. * r5);
    Y(i,j,counter,3) = (c5 * (x - y) * (x + y) * (x2 + y2 - 2 * z2)) / (4. * r5);
    counter++;

    Y(i,j,counter,0) = (c3 * (x3 - 3 * xy2)) / (4. * r3);
    Y(i,j,counter,1) =
        (3 * c3 * (-(y2 * (y2 + z2)) + x2 * (3 * y2 + z2))) / (4. * r5);
    Y(i,j,counter,2) = (3 * c3 * x * y * (-3 * x2 + y2 - 2 * z2)) / (4. * r5);
    Y(i,j,counter,3) = (-3 * c3 * (x3 - 3 * xy2) * z) / (4. * r5);
    counter++;
  }
  if (l == 3)
    return;

  { // l_counter == 4
    Y(i,j,counter,0) = (3 * c9 * x * (x - y) * y * (x + y)) / (4. * r4);
    Y(i,j,counter,1) =
        (-3 * c9 * y * (x4 + y2 * (y2 + z2) - 3 * x2 * (2 * y2 + z2))) /
        (4. * r6);
    Y(i,j,counter,2) = (3 * c9 * x * (x4 + y4 - 3 * y2 * z2 + x2 * (-6 * y2 + z2))) /
                  (4. * r6);
    Y(i,j,counter,3) = (-3 * c9 * x * (x - y) * y * (x + y) * z) / r6;
    counter++;

    Y(i,j,counter,0) = (-3 * c3 * y * (-3 * x2 + y2) * z) / (4. * r4);
    Y(i,j,counter,1) =
        (3 * c3 * x * y * z * (-3 * x2 + 5 * y2 + 3 * z2)) / (2. * r6);
    Y(i,j,counter,2) =
        (3 * c3 * z * (3 * x4 + y4 - 3 * y2 * z2 + 3 * x2 * (-4 * y2 + z2))) /
        (4. * r6);
    Y(i,j,counter,3) =
        (-3 * c3 * y * (-3 * x2 + y2) * (x2 + y2 - 3 * z2)) / (4. * r6);
    counter++;

    Y(i,j,counter,0) = (-3 * c4 * x * y * (x2 + y2 - 6 * z2)) / (4. * r4);
    Y(i,j,counter,1) =
        (3 * c4 * y * (x4 - y4 - 21 * x2 * z2 + 5 * y2 * z2 + 6 * z4)) /
        (4. * r6);
    Y(i,j,counter,2) = (3 * c4 * x * (-x4 + y4 + (5 * x2 - 21 * y2) * z2 + 6 * z4)) /
                  (4. * r6);
    Y(i,j,counter,3) = (3 * c4 * x * y * z * (4 * x2py21 - 3 * z2)) / r6;
    counter++;

    Y(i,j,counter,0) = (-3 * c8 * y * z * (3 * x2py21 - 4 * z2)) / (4. * r4);
    Y(i,j,counter,1) = (3 * c8 * x * y * z * (3 * x2py21 - 11 * z2)) / (2. * r6);
    Y(i,j,counter,2) =
        (3 * c8 * z * (-3 * x4 + 3 * y4 + (x2 - 21 * y2) * z2 + 4 * z4)) /
        (4. * r6);
    Y(i,j,counter,3) =
        (-3 * c8 * y * (3 * x2py2 - 21 * x2py21 * z2 + 4 * z4)) / (4. * r6);
    counter++;

    Y(i,j,counter,0) =
        (9 * x2py2 - 72 * x2py21 * z2 + 24 * z4) / (16. * std::sqrt(Pi) * r4);
    Y(i,j,counter,1) = (15 * x * z2 * (3 * x2py21 - 4 * z2)) / (4. * std::sqrt(Pi) * r6);
    Y(i,j,counter,2) = (15 * y * z2 * (3 * x2py21 - 4 * z2)) / (4. * std::sqrt(Pi) * r6);
    Y(i,j,counter,3) =
        (15 * x2py21 * z * (-3 * x2py21 + 4 * z2)) / (4. * std::sqrt(Pi) * r6);
    counter++;

    Y(i,j,counter,0) = (-3 * c8 * x * z * (3 * x2py21 - 4 * z2)) / (4. * r4);
    Y(i,j,counter,1) =
        (3 * c8 * z * (3 * x4 - 3 * y4 - 21 * x2 * z2 + y2 * z2 + 4 * z4)) /
        (4. * r6);
    Y(i,j,counter,2) = (3 * c8 * x * y * z * (3 * x2py21 - 11 * z2)) / (2. * r6);
    Y(i,j,counter,3) =
        (-3 * c8 * x * (3 * x2py2 - 21 * x2py21 * z2 + 4 * z4)) / (4. * r6);
    counter++;

    Y(i,j,counter,0) = (-3 * c4 * (x - y) * (x + y) * (x2 + y2 - 6 * z2)) / (8. * r4);
    Y(i,j,counter,1) =
        (-3 * c4 * x * (y4 - 9 * y2 * z2 - 3 * z4 + x2 * (y2 + 4 * z2))) /
        (2. * r6);
    Y(i,j,counter,2) =
        (3 * c4 * y * (x4 + 4 * y2 * z2 - 3 * z4 + x2 * (y2 - 9 * z2))) /
        (2. * r6);
    Y(i,j,counter,3) =
        (3 * c4 * (x - y) * (x + y) * z * (4 * x2py21 - 3 * z2)) / (2. * r6);
    counter++;

    Y(i,j,counter,0) = (3 * c3 * (x3 - 3 * xy2) * z) / (4. * r4);
    Y(i,j,counter,1) =
        (-3 * c3 * z * (x4 + 3 * y2 * (y2 + z2) - 3 * x2 * (4 * y2 + z2))) /
        (4. * r6);
    Y(i,j,counter,2) =
        (-3 * c3 * x * y * z * (5 * x2 - 3 * y2 + 3 * z2)) / (2. * r6);
    Y(i,j,counter,3) = (3 * c3 * (x3 - 3 * xy2) * (x2 + y2 - 3 * z2)) / (4. * r6);
    counter++;

    Y(i,j,counter,0) = (3 * c9 * (x4 - 6 * x2 * y2 + y4)) / (16. * r4);
    Y(i,j,counter,1) =
        (3 * c9 * x * (-4 * y4 - 3 * y2 * z2 + x2 * (4 * y2 + z2))) / (4. * r6);
    Y(i,j,counter,2) =
        (3 * c9 * y * (-4 * x4 + y2 * z2 + x2 * (4 * y2 - 3 * z2))) / (4. * r6);
    Y(i,j,counter,3) = (-3 * c9 * (x4 - 6 * x2 * y2 + y4) * z) / (4. * r6);
    counter++;
  }
  if (l == 4)
    return;

  { // if l_counter == 5
    Y(i,j,counter,0) = (3 * c10 * (5 * x4 * y - 10 * x2 * y3 + y5)) / (16. * r5);
    Y(i,j,counter,1) = (-15 * c10 * x * y *
                   (x4 + 5 * y4 + 4 * y2 * z2 - 2 * x2 * (5 * y2 + 2 * z2))) /
                  (16. * r7);
    Y(i,j,counter,2) =
        (15 * c10 *
         (x6 - 10 * x4 * y2 + 5 * x2 * y4 + (x4 - 6 * x2 * y2 + y4) * z2)) /
        (16. * r7);
    Y(i,j,counter,3) =
        (-15 * c10 * (5 * x4 * y - 10 * x2 * y3 + y5) * z) / (16. * r7);
    counter++;

    Y(i,j,counter,0) = (3 * c11 * x * (x - y) * y * (x + y) * z) / (4. * r5);
    Y(i,j,counter,1) = (-3 * c11 * y * z *
                   (2 * x4 + y2 * (y2 + z2) - x2 * (7 * y2 + 3 * z2))) /
                  (4. * r7);
    Y(i,j,counter,2) =
        (3 * c11 * x * z * (x4 + 2 * y4 - 3 * y2 * z2 + x2 * (-7 * y2 + z2))) /
        (4. * r7);
    Y(i,j,counter,3) =
        (3 * c11 * x * (x - y) * y * (x + y) * (x2 + y2 - 4 * z2)) / (4. * r7);
    counter++;

    Y(i,j,counter,0) = (c12 * y * (-3 * x2 + y2) * (x2 + y2 - 8 * z2)) / (16. * r5);
    Y(i,j,counter,1) =
        (3 * c12 * x * y *
         (x4 - 3 * y4 + 28 * y2 * z2 + 16 * z4 - 2 * x2 * (y2 + 14 * z2))) /
        (16. * r7);
    Y(i,j,counter,2) = (-3 * c12 *
                   (x2 * (x2 - 3 * y2) * x2py21 -
                    7 * (x4 - 6 * x2 * y2 + y4) * z2 + 8 * (-x2 + y2) * z4)) /
                  (16. * r7);
    Y(i,j,counter,3) = (-3 * c12 * y * (-3 * x2 + y2) * z * (7 * x2py21 - 8 * z2)) /
                  (16. * r7);
    counter++;

    Y(i,j,counter,0) =
        -(std::sqrt(1155 / Pi) * x * y * z * (x2 + y2 - 2 * z2)) / (4. * r5);
    Y(i,j,counter,1) = (std::sqrt(1155 / Pi) * y * z *
                   (2 * x4 - y4 + y2 * z2 + 2 * z4 + x2 * (y2 - 11 * z2))) /
                  (4. * r7);
    Y(i,j,counter,2) = (std::sqrt(1155 / Pi) * x * z *
                   (-x4 + 2 * y4 - 11 * y2 * z2 + 2 * z4 + x2 * (y2 + z2))) /
                  (4. * r7);
    Y(i,j,counter,3) =
        -(std::sqrt(1155 / Pi) * x * y * (x2py2 - 10 * x2py21 * z2 + 4 * z4)) /
        (4. * r7);
    counter++;

    Y(i,j,counter,0) =
        (std::sqrt(165 / Pi) * y * (x2py2 - 12 * x2py21 * z2 + 8 * z4)) / (16. * r5);
    Y(i,j,counter,1) =
        -(std::sqrt(165 / Pi) * x * y * (x2py2 - 40 * x2py21 * z2 + 64 * z4)) /
        (16. * r7);
    Y(i,j,counter,2) =
        (std::sqrt(165 / Pi) * (x2 * x2py2 - (11 * x2 - 29 * y2) * x2py21 * z2 -
                           4 * (x2 + 17 * y2) * z4 + 8 * z6)) /
        (16. * r7);
    Y(i,j,counter,3) =
        -(std::sqrt(165 / Pi) * y * z * (29 * x2py2 - 68 * x2py21 * z2 + 8 * z4)) /
        (16. * r7);
    counter++;

    Y(i,j,counter,0) =
        (std::sqrt(11 / Pi) * z * (15 * x2py2 - 40 * x2py21 * z2 + 8 * z4)) /
        (16. * r5);
    Y(i,j,counter,1) =
        (-15 * std::sqrt(11 / Pi) * x * z * (x2py2 - 12 * x2py21 * z2 + 8 * z4)) /
        (16. * r7);
    Y(i,j,counter,2) =
        (-15 * std::sqrt(11 / Pi) * y * z * (x2py2 - 12 * x2py21 * z2 + 8 * z4)) /
        (16. * r7);
    Y(i,j,counter,3) =
        (15 * std::sqrt(11 / Pi) * x2py21 * (x2py2 - 12 * x2py21 * z2 + 8 * z4)) /
        (16. * r7);
    counter++;

    Y(i,j,counter,0) =
        (std::sqrt(165 / Pi) * x * (x2py2 - 12 * x2py21 * z2 + 8 * z4)) / (16. * r5);
    Y(i,j,counter,1) =
        (std::sqrt(165 / Pi) * (y2 * x2py2 + (29 * x2 - 11 * y2) * x2py21 * z2 -
                           4 * (17 * x2 + y2) * z4 + 8 * z6)) /
        (16. * r7);
    Y(i,j,counter,2) =
        -(std::sqrt(165 / Pi) * x * y * (x2py2 - 40 * x2py21 * z2 + 64 * z4)) /
        (16. * r7);
    Y(i,j,counter,3) =
        -(std::sqrt(165 / Pi) * x * z * (29 * x2py2 - 68 * x2py21 * z2 + 8 * z4)) /
        (16. * r7);
    counter++;

    Y(i,j,counter,0) =
        -(std::sqrt(1155 / Pi) * (x - y) * (x + y) * z * (x2 + y2 - 2 * z2)) /
        (8. * r5);
    Y(i,j,counter,1) =
        (std::sqrt(1155 / Pi) * x * z *
         (x4 - 5 * y4 + 14 * y2 * z2 + 4 * z4 - 2 * x2 * (2 * y2 + 5 * z2))) /
        (8. * r7);
    Y(i,j,counter,2) =
        -(std::sqrt(1155 / Pi) * y * z *
          (-5 * x4 - 4 * x2 * y2 + y4 + 2 * (7 * x2 - 5 * y2) * z2 + 4 * z4)) /
        (8. * r7);
    Y(i,j,counter,3) = -(std::sqrt(1155 / Pi) * (x - y) * (x + y) *
                    (x2py2 - 10 * x2py21 * z2 + 4 * z4)) /
                  (8. * r7);
    counter++;

    Y(i,j,counter,0) = -(c12 * (x3 - 3 * xy2) * (x2 + y2 - 8 * z2)) / (16. * r5);
    Y(i,j,counter,1) = (-3 * c12 *
                   (-(y2 * (-3 * x2 + y2) * x2py21) +
                    7 * (x4 - 6 * x2 * y2 + y4) * z2 + 8 * (-x2 + y2) * z4)) /
                  (16. * r7);
    Y(i,j,counter,2) =
        (3 * c12 * x * y *
         (3 * x4 + 2 * x2 * y2 - y4 - 28 * (x - y) * (x + y) * z2 - 16 * z4)) /
        (16. * r7);
    Y(i,j,counter,3) =
        (3 * c12 * (x3 - 3 * xy2) * z * (7 * x2py21 - 8 * z2)) / (16. * r7);
    counter++;

    Y(i,j,counter,0) = (3 * c11 * (x4 - 6 * x2 * y2 + y4) * z) / (16. * r5);
    Y(i,j,counter,1) = (-3 * c11 * x * z *
                   (x4 - 22 * x2 * y2 + 17 * y4 - 4 * (x2 - 3 * y2) * z2)) /
                  (16. * r7);
    Y(i,j,counter,2) = (-3 * c11 * y * z *
                   (17 * x4 + y4 - 4 * y2 * z2 + x2 * (-22 * y2 + 12 * z2))) /
                  (16. * r7);
    Y(i,j,counter,3) =
        (3 * c11 * (x4 - 6 * x2 * y2 + y4) * (x2 + y2 - 4 * z2)) / (16. * r7);
    counter++;

    Y(i,j,counter,0) = (3 * c10 * (x5 - 10 * x3 * y2 + 5 * x * y4)) / (16. * r5);
    Y(i,j,counter,1) =
        (15 * c10 *
         (5 * x4 * y2 - 10 * x2 * y4 + y6 + (x4 - 6 * x2 * y2 + y4) * z2)) /
        (16. * r7);
    Y(i,j,counter,2) = (-15 * c10 * x * y *
                   (5 * x4 - 10 * x2 * y2 + y4 + 4 * (x - y) * (x + y) * z2)) /
                  (16. * r7);
    Y(i,j,counter,3) =
        (-15 * c10 * (x5 - 10 * x3 * y2 + 5 * x * y4) * z) / (16. * r7);
    counter++;
  }
  if (l == 5)
    return;

  { // l_counter == 6
    Y(i,j,counter,0) =
        (std::sqrt(3003 / two_pi) * (3 * x5 * y - 10 * x3 * y3 + 3 * x * y5)) /
        (16. * r6);
    Y(i,j,counter,1) = (3 * std::sqrt(3003 / two_pi) * y *
                   (-x6 + y4 * (y2 + z2) + 5 * x4 * (3 * y2 + z2) -
                    5 * x2 * (3 * y4 + 2 * y2 * z2))) /
                  (16. * r8);
    Y(i,j,counter,2) = (3 * std::sqrt(3003 / two_pi) * x *
                   (x6 - 15 * x4 * y2 + 15 * x2 * y4 - y6 +
                    (x4 - 10 * x2 * y2 + 5 * y4) * z2)) /
                  (16. * r8);
    Y(i,j,counter,3) = (-3 * std::sqrt(3003 / two_pi) *
                   (3 * x5 * y - 10 * x3 * y3 + 3 * x * y5) * z) /
                  (8. * r8);
    counter++;

    Y(i,j,counter,0) =
        (3 * std::sqrt(1001 / two_pi) * (5 * x4 * y - 10 * x2 * y3 + y5) * z) /
        (16. * r6);
    Y(i,j,counter,1) =
        (-3 * std::sqrt(1001 / two_pi) * x * y * z *
         (5 * x4 + 13 * y4 + 10 * y2 * z2 - 10 * x2 * (3 * y2 + z2))) /
        (8. * r8);
    Y(i,j,counter,2) = (3 * std::sqrt(1001 / two_pi) * z *
                   (5 * x6 - 55 * x4 * y2 + 35 * x2 * y4 - y6 +
                    5 * (x4 - 6 * x2 * y2 + y4) * z2)) /
                  (16. * r8);
    Y(i,j,counter,3) = (3 * std::sqrt(1001 / two_pi) * (5 * x4 * y - 10 * x2 * y3 + y5) *
                   (x2 + y2 - 5 * z2)) /
                  (16. * r8);
    counter++;

    Y(i,j,counter,0) =
        (-3 * std::sqrt(91 / Pi) * x * (x - y) * y * (x + y) * (x2 + y2 - 10 * z2)) /
        (8. * r6);
    Y(i,j,counter,1) =
        (3 * std::sqrt(91 / Pi) * y *
         (x6 + y2 * (y2 - 10 * z2) * (y2 + z2) - 5 * x4 * (y2 + 7 * z2) -
          5 * x2 * (y4 - 16 * y2 * z2 - 6 * z4))) /
        (8. * r8);
    Y(i,j,counter,2) =
        (-3 * std::sqrt(91 / Pi) * x *
         (x6 + y6 - 35 * y4 * z2 + 30 * y2 * z4 - x4 * (5 * y2 + 9 * z2) -
          5 * x2 * (y4 - 16 * y2 * z2 + 2 * z4))) /
        (8. * r8);
    Y(i,j,counter,3) = (3 * std::sqrt(91 / Pi) * x * (x - y) * y * (x + y) * z *
                   (13 * x2py21 - 20 * z2)) /
                  (4. * r8);
    counter++;

    Y(i,j,counter,0) =
        (std::sqrt(1365 / two_pi) * y * (-3 * x2 + y2) * z * (3 * x2py21 - 8 * z2)) /
        (16. * r6);
    Y(i,j,counter,1) =
        (3 * std::sqrt(1365 / two_pi) * x * y * z *
         (3 * x4 - 5 * y4 + 14 * y2 * z2 + 8 * z4 - 2 * x2 * (y2 + 11 * z2))) /
        (8. * r8);
    Y(i,j,counter,2) = (3 * std::sqrt(1365 / two_pi) * z *
                   (-3 * x6 + 9 * x4 * y2 + 11 * x2 * y4 - y6 +
                    (5 * x4 - 54 * x2 * y2 + 13 * y4) * z2 +
                    8 * (x - y) * (x + y) * z4)) /
                  (16. * r8);
    Y(i,j,counter,3) = (3 * std::sqrt(1365 / two_pi) * y * (-3 * x2 + y2) *
                   (x2py2 - 13 * x2py21 * z2 + 8 * z4)) /
                  (16. * r8);
    counter++;

    Y(i,j,counter,0) =
        (std::sqrt(1365 / two_pi) * x * y * (x2py2 - 16 * x2py21 * z2 + 16 * z4)) /
        (16. * r6);
    Y(i,j,counter,1) =
        (std::sqrt(1365 / two_pi) * y *
         (-((x - y) * (x + y) * x2py2) + (53 * x2 - 15 * y2) * x2py21 * z2 -
          128 * x2 * z4 + 16 * z6)) /
        (16. * r8);
    Y(i,j,counter,2) =
        (std::sqrt(1365 / two_pi) * x *
         ((x - y) * (x + y) * x2py2 - (15 * x2 - 53 * y2) * x2py21 * z2 -
          128 * y2 * z4 + 16 * z6)) /
        (16. * r8);
    Y(i,j,counter,3) = -(std::sqrt(1365 / two_pi) * x * y * z *
                    (19 * x2py2 - 64 * x2py21 * z2 + 16 * z4)) /
                  (8. * r8);
    counter++;

    Y(i,j,counter,0) =
        (std::sqrt(273 / Pi) * y * z * (5 * x2py2 - 20 * x2py21 * z2 + 8 * z4)) /
        (16. * r6);
    Y(i,j,counter,1) = -(std::sqrt(273 / Pi) * x * y * z *
                    (5 * x2py2 - 50 * x2py21 * z2 + 44 * z4)) /
                  (8. * r8);
    Y(i,j,counter,2) =
        (std::sqrt(273 / Pi) * z *
         (5 * (x - y) * (x + y) * x2py2 - 5 * (3 * x2 - 17 * y2) * x2py21 * z2 -
          4 * (3 * x2 + 25 * y2) * z4 + 8 * z6)) /
        (16. * r8);
    Y(i,j,counter,3) = (std::sqrt(273 / Pi) * y *
                   (5 * x2py3 - 85 * x2py2 * z2 + 100 * x2py21 * z4 - 8 * z6)) /
                  (16. * r8);
    counter++;

    Y(i,j,counter,0) = (std::sqrt(13 / Pi) * (-5 * x2py3 + 90 * x2py2 * z2 -
                                   120 * x2py21 * z4 + 16 * z6)) /
                 (32. * r6);
    Y(i,j,counter,1) = (-21 * std::sqrt(13 / Pi) * x * z2 *
                   (5 * x2py2 - 20 * x2py21 * z2 + 8 * z4)) /
                  (16. * r8);
    Y(i,j,counter,2) = (-21 * std::sqrt(13 / Pi) * y * z2 *
                   (5 * x2py2 - 20 * x2py21 * z2 + 8 * z4)) /
                  (16. * r8);
    Y(i,j,counter,3) = (21 * std::sqrt(13 / Pi) * x2py21 * z *
                   (5 * x2py2 - 20 * x2py21 * z2 + 8 * z4)) /
                  (16. * r8);
    counter++;

    Y(i,j,counter,0) =
        (std::sqrt(273 / Pi) * x * z * (5 * x2py2 - 20 * x2py21 * z2 + 8 * z4)) /
        (16. * r6);
    Y(i,j,counter,1) = (std::sqrt(273 / Pi) * z *
                   (-5 * (x - y) * (x + y) * x2py2 +
                    5 * (17 * x2 - 3 * y2) * x2py21 * z2 -
                    4 * (25 * x2 + 3 * y2) * z4 + 8 * z6)) /
                  (16. * r8);
    Y(i,j,counter,2) = -(std::sqrt(273 / Pi) * x * y * z *
                    (5 * x2py2 - 50 * x2py21 * z2 + 44 * z4)) /
                  (8. * r8);
    Y(i,j,counter,3) = (std::sqrt(273 / Pi) * x *
                   (5 * x2py3 - 85 * x2py2 * z2 + 100 * x2py21 * z4 - 8 * z6)) /
                  (16. * r8);
    counter++;

    Y(i,j,counter,0) = (std::sqrt(1365 / two_pi) * (x - y) * (x + y) *
                  (x2py2 - 16 * x2py21 * z2 + 16 * z4)) /
                 (32. * r6);
    Y(i,j,counter,1) = (std::sqrt(1365 / two_pi) * x *
                   (2 * y2 * x2py2 + (19 * x2 - 49 * y2) * x2py21 * z2 +
                    64 * (-x + y) * (x + y) * z4 + 16 * z6)) /
                  (16. * r8);
    Y(i,j,counter,2) = -(std::sqrt(1365 / two_pi) * y *
                    (2 * x2 * x2py2 - (49 * x2 - 19 * y2) * x2py21 * z2 +
                     64 * (x - y) * (x + y) * z4 + 16 * z6)) /
                  (16. * r8);
    Y(i,j,counter,3) = -(std::sqrt(1365 / two_pi) * (x - y) * (x + y) * z *
                    (19 * x2py2 - 64 * x2py21 * z2 + 16 * z4)) /
                  (16. * r8);
    counter++;

    Y(i,j,counter,0) =
        -(std::sqrt(1365 / two_pi) * (x3 - 3 * xy2) * z * (3 * x2py21 - 8 * z2)) /
        (16. * r6);
    Y(i,j,counter,1) = (3 * std::sqrt(1365 / two_pi) * z *
                   (x6 - 11 * x4 * y2 - 9 * x2 * y4 + 3 * y6 +
                    (-13 * x4 + 54 * x2 * y2 - 5 * y4) * z2 +
                    8 * (x - y) * (x + y) * z4)) /
                  (16. * r8);
    Y(i,j,counter,2) =
        (3 * std::sqrt(1365 / two_pi) * x * y * z *
         (5 * x4 - 3 * y4 + 22 * y2 * z2 - 8 * z4 + 2 * x2 * (y2 - 7 * z2))) /
        (8. * r8);
    Y(i,j,counter,3) = (-3 * std::sqrt(1365 / two_pi) * (x3 - 3 * xy2) *
                   (x2py2 - 13 * x2py21 * z2 + 8 * z4)) /
                  (16. * r8);
    counter++;

    Y(i,j,counter,0) =
        (-3 * std::sqrt(91 / Pi) * (x4 - 6 * x2 * y2 + y4) * (x2 + y2 - 10 * z2)) /
        (32. * r6);
    Y(i,j,counter,1) =
        (-3 * std::sqrt(91 / Pi) * x *
         (8 * y2 * (x4 - y4) + (13 * x4 - 150 * x2 * y2 + 85 * y4) * z2 -
          20 * (x2 - 3 * y2) * z4)) /
        (16. * r8);
    Y(i,j,counter,2) =
        (-3 * std::sqrt(91 / Pi) * y *
         (-8 * x6 + 8 * x2 * y4 + (85 * x4 - 150 * x2 * y2 + 13 * y4) * z2 +
          20 * (3 * x2 - y2) * z4)) /
        (16. * r8);
    Y(i,j,counter,3) = (3 * std::sqrt(91 / Pi) * (x4 - 6 * x2 * y2 + y4) * z *
                   (13 * x2py21 - 20 * z2)) /
                  (16. * r8);
    counter++;

    Y(i,j,counter,0) =
        (3 * std::sqrt(1001 / two_pi) * (x5 - 10 * x3 * y2 + 5 * x * y4) * z) /
        (16. * r6);
    Y(i,j,counter,1) = (3 * std::sqrt(1001 / two_pi) * z *
                   (-x6 + 35 * x4 * y2 - 55 * x2 * y4 + 5 * y6 +
                    5 * (x4 - 6 * x2 * y2 + y4) * z2)) /
                  (16. * r8);
    Y(i,j,counter,2) =
        (-3 * std::sqrt(1001 / two_pi) * x * y * z *
         (13 * x4 - 30 * x2 * y2 + 5 * y4 + 10 * (x - y) * (x + y) * z2)) /
        (8. * r8);
    Y(i,j,counter,3) = (3 * std::sqrt(1001 / two_pi) * (x5 - 10 * x3 * y2 + 5 * x * y4) *
                   (x2 + y2 - 5 * z2)) /
                  (16. * r8);
    counter++;

    Y(i,j,counter,0) =
        (std::sqrt(3003 / two_pi) * (x6 - 15 * x4 * y2 + 15 * x2 * y4 - y6)) /
        (32. * r6);
    Y(i,j,counter,1) = (3 * std::sqrt(3003 / two_pi) * x *
                   (6 * y6 + 5 * y4 * z2 - 10 * x2 * y2 * (2 * y2 + z2) +
                    x4 * (6 * y2 + z2))) /
                  (16. * r8);
    Y(i,j,counter,2) = (-3 * std::sqrt(3003 / two_pi) * y *
                   (6 * x6 - 20 * x4 * y2 + 6 * x2 * y4 +
                    (5 * x4 - 10 * x2 * y2 + y4) * z2)) /
                  (16. * r8);
    Y(i,j,counter,3) = (3 * std::sqrt(3003 / two_pi) *
                   (-x6 + 15 * x4 * y2 - 15 * x2 * y4 + y6) * z) /
                  (16. * r8);
    counter++;
  }
  if (l == 6)
    return;

  { // l_counter == 7
    Y(i,j,counter,0) = (-3 * std::sqrt(715 / Pi) * y *
                  (-7 * x6 + 35 * x4 * y2 - 21 * x2 * y4 + y6)) /
                 (64. * r7);
    Y(i,j,counter,1) = (21 * std::sqrt(715 / Pi) * x * y *
                   (-x6 + 7 * y6 + 6 * y4 * z2 + 3 * x4 * (7 * y2 + 2 * z2) -
                    5 * x2 * (7 * y4 + 4 * y2 * z2))) /
                  (64. * r9);
    Y(i,j,counter,2) = (21 * std::sqrt(715 / Pi) *
                   (x8 - 21 * x6 * y2 + 35 * x4 * y4 - 7 * x2 * y6 +
                    (x6 - 15 * x4 * y2 + 15 * x2 * y4 - y6) * z2)) /
                  (64. * r9);
    Y(i,j,counter,3) = (21 * std::sqrt(715 / Pi) * y *
                   (-7 * x6 + 35 * x4 * y2 - 21 * x2 * y4 + y6) * z) /
                  (64. * r9);
    counter++;

    Y(i,j,counter,0) = (3 * std::sqrt(5005 / two_pi) *
                  (3 * x5 * y - 10 * x3 * y3 + 3 * x * y5) * z) /
                 (16. * r7);
    Y(i,j,counter,1) = (3 * std::sqrt(5005 / two_pi) * y * z *
                   (-6 * x6 + 55 * x4 * y2 - 48 * x2 * y4 + 3 * y6 +
                    3 * (5 * x4 - 10 * x2 * y2 + y4) * z2)) /
                  (16. * r9);
    Y(i,j,counter,2) = (3 * std::sqrt(5005 / two_pi) * x * z *
                   (3 * x6 - 48 * x4 * y2 + 55 * x2 * y4 - 6 * y6 +
                    3 * (x4 - 10 * x2 * y2 + 5 * y4) * z2)) /
                  (16. * r9);
    Y(i,j,counter,3) =
        (3 * std::sqrt(5005 / two_pi) * (3 * x5 * y - 10 * x3 * y3 + 3 * x * y5) *
         (x2 + y2 - 6 * z2)) /
        (16. * r9);
    counter++;

    Y(i,j,counter,0) =
        (-3 * c11 * (5 * x4 * y - 10 * x2 * y3 + y5) * (x2 + y2 - 12 * z2)) /
        (64. * r7);
    Y(i,j,counter,1) = (3 * c11 * x * y *
                   (5 * x2py21 * (x4 - 10 * x2 * y2 + 5 * y4) -
                    2 * (105 * x4 - 430 * x2 * y2 + 153 * y4) * z2 +
                    240 * (x - y) * (x + y) * z4)) /
                  (64. * r9);
    Y(i,j,counter,2) = (-3 * c11 *
                   (5 * x2 * x2py21 * (x4 - 10 * x2 * y2 + 5 * y4) +
                    (-55 * x6 + 705 * x4 * y2 - 585 * x2 * y4 + 31 * y6) * z2 -
                    60 * (x4 - 6 * x2 * y2 + y4) * z4)) /
                  (64. * r9);
    Y(i,j,counter,3) = (3 * c11 * (5 * x4 * y - 10 * x2 * y3 + y5) * z *
                   (31 * x2py21 - 60 * z2)) /
                  (64. * r9);
    counter++;

    Y(i,j,counter,0) =
        (-3 * c11 * x * (x - y) * y * (x + y) * z * (3 * x2py21 - 10 * z2)) /
        (8. * r7);
    Y(i,j,counter,1) =
        (3 * c11 * y * z *
         (3 * x2py21 * (2 * x4 - 7 * x2 * y2 + y4) +
          (-55 * x4 + 90 * x2 * y2 - 7 * y4) * z2 + 10 * (3 * x2 - y2) * z4)) /
        (8. * r9);
    Y(i,j,counter,2) =
        (3 * c11 * x * z *
         (-3 * x2py21 * (x4 - 7 * x2 * y2 + 2 * y4) +
          (7 * x4 - 90 * x2 * y2 + 55 * y4) * z2 + 10 * (x2 - 3 * y2) * z4)) /
        (8. * r9);
    Y(i,j,counter,3) = (-3 * c11 * x * (x - y) * y * (x + y) *
                   (3 * x2py2 - 48 * x2py21 * z2 + 40 * z4)) /
                  (8. * r9);
    counter++;

    Y(i,j,counter,0) = (-3 * c9 * y * (-3 * x2 + y2) *
                  (3 * x2py2 - 60 * x2py21 * z2 + 80 * z4)) /
                 (64. * r7);
    Y(i,j,counter,1) =
        (-3 * c9 * x * y *
         (9 * (x2 - 3 * y2) * x2py2 - 6 * (99 * x2 - 109 * y2) * x2py21 * z2 +
          160 * (12 * x2 - 5 * y2) * z4 - 480 * z6)) /
        (64. * r9);
    Y(i,j,counter,2) = (3 * c9 *
                   (9 * x2 * (x2 - 3 * y2) * x2py2 -
                    3 * x2py21 * (57 * x4 - 312 * x2 * y2 + 47 * y4) * z2 +
                    20 * (3 * x4 - 102 * x2 * y2 + 31 * y4) * z4 +
                    240 * (x - y) * (x + y) * z6)) /
                  (64. * r9);
    Y(i,j,counter,3) = (3 * c9 * y * (-3 * x2 + y2) * z *
                   (141 * x2py2 - 620 * x2py21 * z2 + 240 * z4)) /
                  (64. * r9);
    counter++;

    Y(i,j,counter,0) =
        (3 * c3 * x * y * z * (15 * x2py2 - 80 * x2py21 * z2 + 48 * z4)) /
        (16. * r7);
    Y(i,j,counter,1) = (3 * c3 * y * z *
                   (15 * (-2 * x6 - 3 * x4 * y2 + y6) +
                    5 * (79 * x2 - 13 * y2) * x2py21 * z2 -
                    16 * (33 * x2 + 2 * y2) * z4 + 48 * z6)) /
                  (16. * r9);
    Y(i,j,counter,2) =
        (3 * c3 * x * z *
         (15 * (x2 - 2 * y2) * x2py2 - 5 * (13 * x2 - 79 * y2) * x2py21 * z2 -
          16 * (2 * x2 + 33 * y2) * z4 + 48 * z6)) /
        (16. * r9);
    Y(i,j,counter,3) =
        (3 * c3 * x * y *
         (15 * x2py3 - 330 * x2py2 * z2 + 560 * x2py21 * z4 - 96 * z6)) /
        (16. * r9);
    counter++;

    Y(i,j,counter,0) =
        -(c5 * y *
          (5 * x2py3 - 120 * x2py2 * z2 + 240 * x2py21 * z4 - 64 * z6)) /
        (64. * r7);
    Y(i,j,counter,1) =
        (c5 * x * y *
         (5 * x2py3 - 390 * x2py2 * z2 + 1680 * x2py21 * z4 - 928 * z6)) /
        (64. * r9);
    Y(i,j,counter,2) =
        (c5 * (-5 * x2 * x2py3 + 5 * (23 * x2 - 55 * y2) * x2py2 * z2 -
               120 * (x2 - 13 * y2) * x2py21 * z4 -
               16 * (11 * x2 + 69 * y2) * z6 + 64 * z8)) /
        (64. * r9);
    Y(i,j,counter,3) =
        (c5 * y * z *
         (275 * x2py3 - 1560 * x2py2 * z2 + 1104 * x2py21 * z4 - 64 * z6)) /
        (64. * r9);
    counter++;

    Y(i,j,counter,0) =
        (c2 * z *
         (-35 * x2py3 + 210 * x2py2 * z2 - 168 * x2py21 * z4 + 16 * z6)) /
        (32. * r7);
    Y(i,j,counter,1) =
        (7 * c2 * x * z *
         (5 * x2py3 - 120 * x2py2 * z2 + 240 * x2py21 * z4 - 64 * z6)) /
        (32. * r9);
    Y(i,j,counter,2) =
        (7 * c2 * y * z *
         (5 * x2py3 - 120 * x2py2 * z2 + 240 * x2py21 * z4 - 64 * z6)) /
        (32. * r9);
    Y(i,j,counter,3) =
        (-7 * c2 * x2py21 *
         (5 * x2py3 - 120 * x2py2 * z2 + 240 * x2py21 * z4 - 64 * z6)) /
        (32. * r9);
    counter++;

    Y(i,j,counter,0) =
        -(c5 * x *
          (5 * x2py3 - 120 * x2py2 * z2 + 240 * x2py21 * z4 - 64 * z6)) /
        (64. * r7);
    Y(i,j,counter,1) =
        -(c5 * (5 * y2 * x2py3 + 5 * (55 * x2 - 23 * y2) * x2py2 * z2 +
                120 * (-13 * x2 + y2) * x2py21 * z4 +
                16 * (69 * x2 + 11 * y2) * z6 - 64 * z8)) /
        (64. * r9);
    Y(i,j,counter,2) =
        (c5 * x * y *
         (5 * x2py3 - 390 * x2py2 * z2 + 1680 * x2py21 * z4 - 928 * z6)) /
        (64. * r9);
    Y(i,j,counter,3) =
        (c5 * x * z *
         (275 * x2py3 - 1560 * x2py2 * z2 + 1104 * x2py21 * z4 - 64 * z6)) /
        (64. * r9);
    counter++;

    Y(i,j,counter,0) = (3 * c3 * (x - y) * (x + y) * z *
                  (15 * x2py2 - 80 * x2py21 * z2 + 48 * z4)) /
                 (32. * r7);
    Y(i,j,counter,1) =
        (-3 * c3 * x * z *
         (15 * (x2 - 5 * y2) * x2py2 - 10 * (33 * x2 - 59 * y2) * x2py21 * z2 +
          16 * (35 * x2 - 27 * y2) * z4 - 96 * z6)) /
        (32. * r9);
    Y(i,j,counter,2) =
        (-3 * c3 * y * z *
         (15 * (5 * x2 - y2) * x2py2 - 10 * (59 * x2 - 33 * y2) * x2py21 * z2 +
          16 * (27 * x2 - 35 * y2) * z4 + 96 * z6)) /
        (32. * r9);
    Y(i,j,counter,3) =
        (3 * c3 * (x - y) * (x + y) *
         (15 * x2py3 - 330 * x2py2 * z2 + 560 * x2py21 * z4 - 96 * z6)) /
        (32. * r9);
    counter++;

    Y(i,j,counter,0) =
        (3 * c9 * (x3 - 3 * xy2) * (3 * x2py2 - 60 * x2py21 * z2 + 80 * z4)) /
        (64. * r7);
    Y(i,j,counter,1) = (3 * c9 *
                   (-9 * y2 * (-3 * x2 + y2) * x2py2 +
                    3 * x2py21 * (47 * x4 - 312 * x2 * y2 + 57 * y4) * z2 -
                    20 * (31 * x4 - 102 * x2 * y2 + 3 * y4) * z4 +
                    240 * (x - y) * (x + y) * z6)) /
                  (64. * r9);
    Y(i,j,counter,2) =
        (-3 * c9 * x * y *
         (9 * (3 * x2 - y2) * x2py2 - 6 * (109 * x2 - 99 * y2) * x2py21 * z2 +
          160 * (5 * x2 - 12 * y2) * z4 + 480 * z6)) /
        (64. * r9);
    Y(i,j,counter,3) = (-3 * c9 * (x3 - 3 * xy2) * z *
                   (141 * x2py2 - 620 * x2py21 * z2 + 240 * z4)) /
                  (64. * r9);
    counter++;

    Y(i,j,counter,0) =
        (-3 * c11 * (x4 - 6 * x2 * y2 + y4) * z * (3 * x2py21 - 10 * z2)) /
        (32. * r7);
    Y(i,j,counter,1) = (3 * c11 * x * z *
                   (3 * x2py21 * (x4 - 22 * x2 * y2 + 17 * y4) -
                    16 * (3 * x4 - 25 * x2 * y2 + 10 * y4) * z2 +
                    40 * (x2 - 3 * y2) * z4)) /
                  (32. * r9);
    Y(i,j,counter,2) = (3 * c11 * y * z *
                   (3 * x2py21 * (17 * x4 - 22 * x2 * y2 + y4) -
                    16 * (10 * x4 - 25 * x2 * y2 + 3 * y4) * z2 +
                    40 * (-3 * x2 + y2) * z4)) /
                  (32. * r9);
    Y(i,j,counter,3) = (-3 * c11 * (x4 - 6 * x2 * y2 + y4) *
                   (3 * x2py2 - 48 * x2py21 * z2 + 40 * z4)) /
                  (32. * r9);
    counter++;

    Y(i,j,counter,0) =
        (-3 * c11 * (x5 - 10 * x3 * y2 + 5 * x * y4) * (x2 + y2 - 12 * z2)) /
        (64. * r7);
    Y(i,j,counter,1) = (-3 * c11 *
                   (5 * y2 * x2py21 * (5 * x4 - 10 * x2 * y2 + y4) +
                    (31 * x6 - 585 * x4 * y2 + 705 * x2 * y4 - 55 * y6) * z2 -
                    60 * (x4 - 6 * x2 * y2 + y4) * z4)) /
                  (64. * r9);
    Y(i,j,counter,2) = (3 * c11 * x * y *
                   (5 * x2py21 * (5 * x4 - 10 * x2 * y2 + y4) -
                    2 * (153 * x4 - 430 * x2 * y2 + 105 * y4) * z2 -
                    240 * (x - y) * (x + y) * z4)) /
                  (64. * r9);
    Y(i,j,counter,3) = (3 * c11 * (x5 - 10 * x3 * y2 + 5 * x * y4) * z *
                   (31 * x2py21 - 60 * z2)) /
                  (64. * r9);
    counter++;

    Y(i,j,counter,0) = (3 * std::sqrt(5005 / two_pi) *
                  (x6 - 15 * x4 * y2 + 15 * x2 * y4 - y6) * z) /
                 (32. * r7);
    Y(i,j,counter,1) = (-3 * std::sqrt(5005 / two_pi) * x * z *
                   (x6 - 51 * x4 * y2 + 135 * x2 * y4 - 37 * y6 -
                    6 * (x4 - 10 * x2 * y2 + 5 * y4) * z2)) /
                  (32. * r9);
    Y(i,j,counter,2) = (3 * std::sqrt(5005 / two_pi) * y * z *
                   (-37 * x6 + 135 * x4 * y2 - 51 * x2 * y4 + y6 -
                    6 * (5 * x4 - 10 * x2 * y2 + y4) * z2)) /
                  (32. * r9);
    Y(i,j,counter,3) =
        (3 * std::sqrt(5005 / two_pi) * (x6 - 15 * x4 * y2 + 15 * x2 * y4 - y6) *
         (x2 + y2 - 6 * z2)) /
        (32. * r9);
    counter++;

    Y(i,j,counter,0) = (3 * std::sqrt(715 / Pi) *
                  (std::pow(x, 7) - 21 * x5 * y2 + 35 * x3 * y4 - 7 * x * y6)) /
                 (64. * r7);
    Y(i,j,counter,1) =
        (21 * std::sqrt(715 / Pi) *
         (-(y6 * (y2 + z2)) + x6 * (7 * y2 + z2) -
          5 * x4 * (7 * y4 + 3 * y2 * z2) + 3 * x2 * (7 * y6 + 5 * y4 * z2))) /
        (64. * r9);
    Y(i,j,counter,2) = (-21 * std::sqrt(715 / Pi) * x * y *
                   (7 * x6 - 35 * x4 * y2 + 21 * x2 * y4 - y6 +
                    2 * (3 * x4 - 10 * x2 * y2 + 3 * y4) * z2)) /
                  (64. * r9);
    Y(i,j,counter,3) = (-21 * std::sqrt(715 / Pi) *
                   (std::pow(x, 7) - 21 * x5 * y2 + 35 * x3 * y4 - 7 * x * y6) * z) /
                  (64. * r9);
    counter++;
  }
  if (l == 7)
    return;

  { // l_counter == 8
    Y(i,j,counter,0) =
        (3 * std::sqrt(12155 / Pi) * x * y * (x6 - 7 * x4 * y2 + 7 * x2 * y4 - y6)) /
        (32. * r8);
    Y(i,j,counter,1) = (-3 * std::sqrt(12155 / Pi) * y *
                   (x8 + y6 * (y2 + z2) + 35 * x4 * y2 * (2 * y2 + z2) -
                    7 * x6 * (4 * y2 + z2) - 7 * x2 * (4 * y6 + 3 * y4 * z2))) /
                  (32. * r10);
    Y(i,j,counter,2) = (3 * std::sqrt(12155 / Pi) * x *
                   (x8 - 28 * x6 * y2 + 70 * x4 * y4 - 28 * x2 * y6 + y8 +
                    (x6 - 21 * x4 * y2 + 35 * x2 * y4 - 7 * y6) * z2)) /
                  (32. * r10);
    Y(i,j,counter,3) = (3 * std::sqrt(12155 / Pi) * x * y *
                   (-x6 + 7 * x4 * y2 - 7 * x2 * y4 + y6) * z) /
                  (4. * r10);
    counter++;

    Y(i,j,counter,0) = (-3 * std::sqrt(12155 / Pi) * y *
                  (-7 * x6 + 35 * x4 * y2 - 21 * x2 * y4 + y6) * z) /
                 (64. * r8);
    Y(i,j,counter,1) = (3 * std::sqrt(12155 / Pi) * x * y * z *
                   (-7 * x6 + 91 * x4 * y2 - 133 * x2 * y4 + 25 * y6 +
                    7 * (3 * x4 - 10 * x2 * y2 + 3 * y4) * z2)) /
                  (32. * r10);
    Y(i,j,counter,2) = (3 * std::sqrt(12155 / Pi) * z *
                   (7 * x8 - 154 * x6 * y2 + 280 * x4 * y4 - 70 * x2 * y6 + y8 +
                    7 * (x6 - 15 * x4 * y2 + 15 * x2 * y4 - y6) * z2)) /
                  (64. * r10);
    Y(i,j,counter,3) =
        (-3 * std::sqrt(12155 / Pi) * y *
         (-7 * x6 + 35 * x4 * y2 - 21 * x2 * y4 + y6) * (x2 + y2 - 7 * z2)) /
        (64. * r10);
    counter++;

    Y(i,j,counter,0) =
        -(std::sqrt(7293 / two_pi) * (3 * x5 * y - 10 * x3 * y3 + 3 * x * y5) *
          (x2 + y2 - 14 * z2)) /
        (32. * r8);
    Y(i,j,counter,1) = (3 * std::sqrt(7293 / two_pi) * y *
                   (x8 - 14 * x6 * y2 + 14 * x2 * y6 - y8 +
                    (-49 * x6 + 315 * x4 * y2 - 231 * x2 * y4 + 13 * y6) * z2 +
                    14 * (5 * x4 - 10 * x2 * y2 + y4) * z4)) /
                  (32. * r10);
    Y(i,j,counter,2) = (-3 * std::sqrt(7293 / two_pi) * x *
                   (x8 - 14 * x6 * y2 + 14 * x2 * y6 - y8 +
                    (-13 * x6 + 231 * x4 * y2 - 315 * x2 * y4 + 49 * y6) * z2 -
                    14 * (x4 - 10 * x2 * y2 + 5 * y4) * z4)) /
                  (32. * r10);
    Y(i,j,counter,3) =
        (3 * std::sqrt(7293 / two_pi) * (3 * x5 * y - 10 * x3 * y3 + 3 * x * y5) *
         z * (3 * x2py21 - 7 * z2)) /
        (8. * r10);
    counter++;

    Y(i,j,counter,0) = (-3 * std::sqrt(17017 / Pi) * (5 * x4 * y - 10 * x2 * y3 + y5) * z *
                  (x2 + y2 - 4 * z2)) /
                 (64. * r8);
    Y(i,j,counter,1) = (3 * std::sqrt(17017 / Pi) * x * y * z *
                   (5 * x6 - 25 * x4 * y2 - 17 * x2 * y4 + 13 * y6 +
                    (-55 * x4 + 170 * x2 * y2 - 47 * y4) * z2 +
                    40 * (x - y) * (x + y) * z4)) /
                  (32. * r10);
    Y(i,j,counter,2) = (3 * std::sqrt(17017 / Pi) * z *
                   (-5 * x8 + 50 * x6 * y2 + 20 * x4 * y4 - 34 * x2 * y6 + y8 +
                    (15 * x6 - 245 * x4 * y2 + 265 * x2 * y4 - 19 * y6) * z2 +
                    20 * (x4 - 6 * x2 * y2 + y4) * z4)) /
                  (64. * r10);
    Y(i,j,counter,3) = (-3 * std::sqrt(17017 / Pi) * (5 * x4 * y - 10 * x2 * y3 + y5) *
                   (x2py2 - 19 * x2py21 * z2 + 20 * z4)) /
                  (64. * r10);
    counter++;

    Y(i,j,counter,0) = (3 * std::sqrt(1309 / Pi) * x * (x - y) * y * (x + y) *
                  (x2py2 - 24 * x2py21 * z2 + 40 * z4)) /
                 (32. * r8);
    Y(i,j,counter,1) =
        (-3 * std::sqrt(1309 / Pi) * y *
         (x2py2 * (x4 - 6 * x2 * y2 + y4) -
          x2py21 * (79 * x4 - 194 * x2 * y2 + 23 * y4) * z2 +
          16 * (20 * x4 - 25 * x2 * y2 + y4) * z4 + 40 * (-3 * x2 + y2) * z6)) /
        (32. * r10);
    Y(i,j,counter,2) =
        (3 * std::sqrt(1309 / Pi) * x *
         (x2py2 * (x4 - 6 * x2 * y2 + y4) -
          x2py21 * (23 * x4 - 194 * x2 * y2 + 79 * y4) * z2 +
          16 * (x4 - 25 * x2 * y2 + 20 * y4) * z4 + 40 * (x2 - 3 * y2) * z6)) /
        (32. * r10);
    Y(i,j,counter,3) = (-3 * std::sqrt(1309 / Pi) * x * (x - y) * y * (x + y) * z *
                   (7 * x2py2 - 38 * x2py21 * z2 + 20 * z4)) /
                  (4. * r10);
    counter++;

    Y(i,j,counter,0) = (std::sqrt(19635 / Pi) * (3 * x2 * y - y3) * z *
                  (3 * x2py2 - 20 * x2py21 * z2 + 16 * z4)) /
                 (64. * r8);
    Y(i,j,counter,1) =
        (-3 * std::sqrt(19635 / Pi) * x * y * z *
         ((3 * x2 - 5 * y2) * x2py2 - (49 * x2 - 39 * y2) * x2py21 * z2 +
          8 * (11 * x2 - 3 * y2) * z4 - 16 * z6)) /
        (32. * r10);
    Y(i,j,counter,2) = (3 * std::sqrt(19635 / Pi) * z *
                   (x2py2 * (3 * x4 - 12 * x2 * y2 + y4) -
                    x2py21 * (17 * x4 - 132 * x2 * y2 + 27 * y4) * z2 -
                    4 * (x4 + 42 * x2 * y2 - 15 * y4) * z4 +
                    16 * (x - y) * (x + y) * z6)) /
                  (64. * r10);
    Y(i,j,counter,3) = (-3 * std::sqrt(19635 / Pi) * y * (-3 * x2 + y2) *
                   (x2py3 - 27 * x2py2 * z2 + 60 * x2py21 * z4 - 16 * z6)) /
                  (64. * r10);
    counter++;

    Y(i,j,counter,0) = (-3 * std::sqrt(595 / two_pi) * x * y *
                  (x2py3 - 30 * x2py2 * z2 + 80 * x2py21 * z4 - 32 * z6)) /
                 (32. * r8);
    Y(i,j,counter,1) =
        (-3 * std::sqrt(595 / two_pi) * y *
         (-((x - y) * (x + y) * x2py3) + (97 * x2 - 29 * y2) * x2py2 * z2 -
          50 * (11 * x2 - y2) * x2py21 * z4 + 16 * (29 * x2 + 3 * y2) * z6 -
          32 * z8)) /
        (32. * r10);
    Y(i,j,counter,2) =
        (-3 * std::sqrt(595 / two_pi) * x *
         ((x - y) * (x + y) * x2py3 - (29 * x2 - 97 * y2) * x2py2 * z2 +
          50 * (x2 - 11 * y2) * x2py21 * z4 + 16 * (3 * x2 + 29 * y2) * z6 -
          32 * z8)) /
        (32. * r10);
    Y(i,j,counter,3) =
        (3 * std::sqrt(595 / two_pi) * x * y * z *
         (17 * x2py3 - 125 * x2py2 * z2 + 128 * x2py21 * z4 - 16 * z6)) /
        (8. * r10);
    counter++;

    Y(i,j,counter,0) =
        (-3 * std::sqrt(17 / Pi) * y * z *
         (35 * x2py3 - 280 * x2py2 * z2 + 336 * x2py21 * z4 - 64 * z6)) /
        (64. * r8);
    Y(i,j,counter,1) =
        (3 * std::sqrt(17 / Pi) * x * y * z *
         (35 * x2py3 - 665 * x2py2 * z2 + 1568 * x2py21 * z4 - 592 * z6)) /
        (32. * r10);
    Y(i,j,counter,2) = (-3 * std::sqrt(17 / Pi) * z *
                   (35 * (x - y) * (x + y) * x2py3 -
                    35 * (7 * x2 - 31 * y2) * x2py2 * z2 +
                    56 * (x2 - 55 * y2) * x2py21 * z4 +
                    16 * (17 * x2 + 91 * y2) * z6 - 64 * z8)) /
                  (64. * r10);
    Y(i,j,counter,3) = (-3 * std::sqrt(17 / Pi) * y *
                   (35 * x2py4 - 1085 * x2py3 * z2 + 3080 * x2py2 * z4 -
                    1456 * x2py21 * z6 + 64 * z8)) /
                  (64. * r10);
    counter++;

    Y(i,j,counter,0) =
        (std::sqrt(17 / Pi) * (35 * x2py4 - 1120 * x2py3 * z2 + 3360 * x2py2 * z4 -
                          1792 * x2py21 * z6 + 128 * z8)) /
        (256. * r8);
    Y(i,j,counter,1) =
        (9 * std::sqrt(17 / Pi) * x * z2 *
         (35 * x2py3 - 280 * x2py2 * z2 + 336 * x2py21 * z4 - 64 * z6)) /
        (32. * r10);
    Y(i,j,counter,2) =
        (9 * std::sqrt(17 / Pi) * y * z2 *
         (35 * x2py3 - 280 * x2py2 * z2 + 336 * x2py21 * z4 - 64 * z6)) /
        (32. * r10);
    Y(i,j,counter,3) =
        (-9 * std::sqrt(17 / Pi) * x2py21 * z *
         (35 * x2py3 - 280 * x2py2 * z2 + 336 * x2py21 * z4 - 64 * z6)) /
        (32. * r10);
    counter++;

    Y(i,j,counter,0) =
        (-3 * std::sqrt(17 / Pi) * x * z *
         (35 * x2py3 - 280 * x2py2 * z2 + 336 * x2py21 * z4 - 64 * z6)) /
        (64. * r8);
    Y(i,j,counter,1) = (3 * std::sqrt(17 / Pi) * z *
                   (35 * (x - y) * (x + y) * x2py3 -
                    35 * (31 * x2 - 7 * y2) * x2py2 * z2 +
                    56 * (55 * x2 - y2) * x2py21 * z4 -
                    16 * (91 * x2 + 17 * y2) * z6 + 64 * z8)) /
                  (64. * r10);
    Y(i,j,counter,2) =
        (3 * std::sqrt(17 / Pi) * x * y * z *
         (35 * x2py3 - 665 * x2py2 * z2 + 1568 * x2py21 * z4 - 592 * z6)) /
        (32. * r10);
    Y(i,j,counter,3) = (-3 * std::sqrt(17 / Pi) * x *
                   (35 * x2py4 - 1085 * x2py3 * z2 + 3080 * x2py2 * z4 -
                    1456 * x2py21 * z6 + 64 * z8)) /
                  (64. * r10);
    counter++;

    Y(i,j,counter,0) = (-3 * std::sqrt(595 / two_pi) * (x - y) * (x + y) *
                  (x2py3 - 30 * x2py2 * z2 + 80 * x2py21 * z4 - 32 * z6)) /
                 (64. * r8);
    Y(i,j,counter,1) = (-3 * std::sqrt(595 / two_pi) * x *
                   (y2 * x2py3 + (17 * x2 - 46 * y2) * x2py2 * z2 +
                    25 * (-5 * x4 + 2 * x2 * y2 + 7 * y4) * z4 +
                    16 * (8 * x2 - 5 * y2) * z6 - 16 * z8)) /
                  (16. * r10);
    Y(i,j,counter,2) = (3 * std::sqrt(595 / two_pi) * y *
                   (x2 * x2py3 - (46 * x2 - 17 * y2) * x2py2 * z2 +
                    25 * (7 * x2 - 5 * y2) * x2py21 * z4 +
                    16 * (-5 * x2 + 8 * y2) * z6 - 16 * z8)) /
                  (16. * r10);
    Y(i,j,counter,3) =
        (3 * std::sqrt(595 / two_pi) * (x - y) * (x + y) * z *
         (17 * x2py3 - 125 * x2py2 * z2 + 128 * x2py21 * z4 - 16 * z6)) /
        (16. * r10);
    counter++;

    Y(i,j,counter,0) = (std::sqrt(19635 / Pi) * (x3 - 3 * xy2) * z *
                  (3 * x2py2 - 20 * x2py21 * z2 + 16 * z4)) /
                 (64. * r8);
    Y(i,j,counter,1) = (3 * std::sqrt(19635 / Pi) * z *
                   (-(x2py2 * (x4 - 12 * x2 * y2 + 3 * y4)) +
                    x2py21 * (27 * x4 - 132 * x2 * y2 + 17 * y4) * z2 +
                    4 * (-15 * x4 + 42 * x2 * y2 + y4) * z4 +
                    16 * (x - y) * (x + y) * z6)) /
                  (64. * r10);
    Y(i,j,counter,2) =
        (-3 * std::sqrt(19635 / Pi) * x * y * z *
         ((5 * x2 - 3 * y2) * x2py2 - (39 * x2 - 49 * y2) * x2py21 * z2 +
          8 * (3 * x2 - 11 * y2) * z4 + 16 * z6)) /
        (32. * r10);
    Y(i,j,counter,3) = (3 * std::sqrt(19635 / Pi) * (x3 - 3 * xy2) *
                   (x2py3 - 27 * x2py2 * z2 + 60 * x2py21 * z4 - 16 * z6)) /
                  (64. * r10);
    counter++;

    Y(i,j,counter,0) = (3 * std::sqrt(1309 / Pi) * (x4 - 6 * x2 * y2 + y4) *
                  (x2py2 - 24 * x2py21 * z2 + 40 * z4)) /
                 (128. * r8);
    Y(i,j,counter,1) = (3 * std::sqrt(1309 / Pi) * x *
                   (2 * (x - y) * y2 * (x + y) * x2py2 +
                    x2py21 * (7 * x4 - 88 * x2 * y2 + 53 * y4) * z2 -
                    2 * (19 * x4 - 130 * x2 * y2 + 35 * y4) * z4 +
                    20 * (x2 - 3 * y2) * z6)) /
                  (16. * r10);
    Y(i,j,counter,2) = (3 * std::sqrt(1309 / Pi) * y *
                   (-2 * x2 * (x - y) * (x + y) * x2py2 +
                    x2py21 * (53 * x4 - 88 * x2 * y2 + 7 * y4) * z2 -
                    2 * (35 * x4 - 130 * x2 * y2 + 19 * y4) * z4 +
                    20 * (-3 * x2 + y2) * z6)) /
                  (16. * r10);
    Y(i,j,counter,3) = (-3 * std::sqrt(1309 / Pi) * (x4 - 6 * x2 * y2 + y4) * z *
                   (7 * x2py2 - 38 * x2py21 * z2 + 20 * z4)) /
                  (16. * r10);
    counter++;

    Y(i,j,counter,0) = (-3 * std::sqrt(17017 / Pi) * (x5 - 10 * x3 * y2 + 5 * x * y4) * z *
                  (x2 + y2 - 4 * z2)) /
                 (64. * r8);
    Y(i,j,counter,1) = (3 * std::sqrt(17017 / Pi) * z *
                   (x8 - 34 * x6 * y2 + 20 * x4 * y4 + 50 * x2 * y6 - 5 * y8 +
                    (-19 * x6 + 265 * x4 * y2 - 245 * x2 * y4 + 15 * y6) * z2 +
                    20 * (x4 - 6 * x2 * y2 + y4) * z4)) /
                  (64. * r10);
    Y(i,j,counter,2) = (-3 * std::sqrt(17017 / Pi) * x * y * z *
                   (-13 * x6 + x4 * (17 * y2 + 47 * z2) +
                    5 * x2 * (5 * y4 - 34 * y2 * z2 + 8 * z4) -
                    5 * (y6 - 11 * y4 * z2 + 8 * y2 * z4))) /
                  (32. * r10);
    Y(i,j,counter,3) = (-3 * std::sqrt(17017 / Pi) * (x5 - 10 * x3 * y2 + 5 * x * y4) *
                   (x2py2 - 19 * x2py21 * z2 + 20 * z4)) /
                  (64. * r10);
    counter++;

    Y(i,j,counter,0) =
        -(std::sqrt(7293 / two_pi) * (x6 - 15 * x4 * y2 + 15 * x2 * y4 - y6) *
          (x2 + y2 - 14 * z2)) /
        (64. * r8);
    Y(i,j,counter,1) = (-3 * std::sqrt(7293 / two_pi) * x *
                   (3 * y8 - 42 * y6 * z2 - 35 * y4 * z4 + 3 * x6 * (y2 + z2) -
                    7 * x4 * (y4 + 12 * y2 * z2 + z4) -
                    7 * x2 * (y6 - 25 * y4 * z2 - 10 * y2 * z4))) /
                  (16. * r10);
    Y(i,j,counter,2) =
        (3 * std::sqrt(7293 / two_pi) * y *
         (3 * x8 + 3 * y6 * z2 - 7 * y4 * z4 - 7 * x6 * (y2 + 6 * z2) -
          7 * x4 * (y4 - 25 * y2 * z2 + 5 * z4) +
          x2 * (3 * y6 - 84 * y4 * z2 + 70 * y2 * z4))) /
        (16. * r10);
    Y(i,j,counter,3) =
        (3 * std::sqrt(7293 / two_pi) * (x6 - 15 * x4 * y2 + 15 * x2 * y4 - y6) * z *
         (3 * x2py21 - 7 * z2)) /
        (16. * r10);
    counter++;

    Y(i,j,counter,0) = (3 * std::sqrt(12155 / Pi) *
                  (std::pow(x, 7) - 21 * x5 * y2 + 35 * x3 * y4 - 7 * x * y6) * z) /
                 (64. * r8);
    Y(i,j,counter,1) = (-3 * std::sqrt(12155 / Pi) * z *
                   (x8 + 7 * y6 * (y2 + z2) - 7 * x6 * (10 * y2 + z2) +
                    35 * x4 * (8 * y4 + 3 * y2 * z2) -
                    7 * x2 * (22 * y6 + 15 * y4 * z2))) /
                  (64. * r10);
    Y(i,j,counter,2) = (-3 * std::sqrt(12155 / Pi) * x * y * z *
                   (25 * x6 - 133 * x4 * y2 + 91 * x2 * y4 - 7 * y6 +
                    7 * (3 * x4 - 10 * x2 * y2 + 3 * y4) * z2)) /
                  (32. * r10);
    Y(i,j,counter,3) = (3 * std::sqrt(12155 / Pi) *
                   (std::pow(x, 7) - 21 * x5 * y2 + 35 * x3 * y4 - 7 * x * y6) *
                   (x2 + y2 - 7 * z2)) /
                  (64. * r10);
    counter++;

    Y(i,j,counter,0) = (3 * std::sqrt(12155 / Pi) *
                  (x8 - 28 * x6 * y2 + 70 * x4 * y4 - 28 * x2 * y6 + y8)) /
                 (256. * r8);
    Y(i,j,counter,1) = (3 * std::sqrt(12155 / Pi) * x *
                   (8 * y2 * (x6 - 7 * x4 * y2 + 7 * x2 * y4 - y6) +
                    (x6 - 21 * x4 * y2 + 35 * x2 * y4 - 7 * y6) * z2)) /
                  (32. * r10);
    Y(i,j,counter,2) = (3 * std::sqrt(12155 / Pi) * y *
                   (-8 * x2 * (x6 - 7 * x4 * y2 + 7 * x2 * y4 - y6) +
                    (-7 * x6 + 35 * x4 * y2 - 21 * x2 * y4 + y6) * z2)) /
                  (32. * r10);
    Y(i,j,counter,3) = (-3 * std::sqrt(12155 / Pi) *
                   (x8 - 28 * x6 * y2 + 70 * x4 * y4 - 28 * x2 * y6 + y8) * z) /
                  (32. * r10);
    counter++;
  }
  if (l == 8)
    return;

  { // l_counter == 9
    Y(i,j,counter,0) =
        (std::sqrt(230945 / two_pi) * (9 * x8 * y - 84 * x6 * y3 + 126 * x4 * y5 -
                                  36 * x2 * std::pow(y, 7) + std::pow(y, 9))) /
        (256. * r9);
    Y(i,j,counter,1) = (-9 * std::sqrt(230945 / two_pi) * x * y *
                   (x8 - 36 * x6 * y2 + 126 * x4 * y4 - 84 * x2 * y6 + 9 * y8 +
                    8 * (-x6 + 7 * x4 * y2 - 7 * x2 * y4 + y6) * z2)) /
                  (256. * r11);
    Y(i,j,counter,2) =
        (9 * std::sqrt(230945 / two_pi) *
         (x2 * (x8 - 36 * x6 * y2 + 126 * x4 * y4 - 84 * x2 * y6 + 9 * y8) +
          (x8 - 28 * x6 * y2 + 70 * x4 * y4 - 28 * x2 * y6 + y8) * z2)) /
        (256. * r11);
    Y(i,j,counter,3) = (-9 * std::sqrt(230945 / two_pi) *
                   (9 * x8 * y - 84 * x6 * y3 + 126 * x4 * y5 -
                    36 * x2 * std::pow(y, 7) + std::pow(y, 9)) *
                   z) /
                  (256. * r11);
    counter++;

    Y(i,j,counter,0) = (3 * std::sqrt(230945 / Pi) * x * y *
                  (x6 - 7 * x4 * y2 + 7 * x2 * y4 - y6) * z) /
                 (32. * r9);
    Y(i,j,counter,1) = (-3 * std::sqrt(230945 / Pi) * y * z *
                   (2 * x8 - 35 * x6 * y2 + 77 * x4 * y4 - 29 * x2 * y6 + y8 +
                    (-7 * x6 + 35 * x4 * y2 - 21 * x2 * y4 + y6) * z2)) /
                  (32. * r11);
    Y(i,j,counter,2) = (3 * std::sqrt(230945 / Pi) * x * z *
                   (x8 - 29 * x6 * y2 + 77 * x4 * y4 - 35 * x2 * y6 + 2 * y8 +
                    (x6 - 21 * x4 * y2 + 35 * x2 * y4 - 7 * y6) * z2)) /
                  (32. * r11);
    Y(i,j,counter,3) = (3 * std::sqrt(230945 / Pi) * x * y *
                   (x6 - 7 * x4 * y2 + 7 * x2 * y4 - y6) * (x2 + y2 - 8 * z2)) /
                  (32. * r11);
    counter++;

    Y(i,j,counter,0) =
        (3 * std::sqrt(13585 / two_pi) * y *
         (-7 * x6 + 35 * x4 * y2 - 21 * x2 * y4 + y6) * (x2 + y2 - 16 * z2)) /
        (256. * r9);
    Y(i,j,counter,1) =
        (3 * std::sqrt(13585 / two_pi) * x * y *
         (7 * x2py21 * (x6 - 21 * x4 * y2 + 35 * x2 * y4 - 7 * y6) -
          8 * (49 * x6 - 455 * x4 * y2 + 567 * x2 * y4 - 97 * y6) * z2 +
          224 * (3 * x4 - 10 * x2 * y2 + 3 * y4) * z4)) /
        (256. * r11);
    Y(i,j,counter,2) =
        (-3 * std::sqrt(13585 / two_pi) *
         (7 * x2 * x2py21 * (x6 - 21 * x4 * y2 + 35 * x2 * y4 - 7 * y6) +
          (-105 * x8 + 2492 * x6 * y2 - 5110 * x4 * y4 + 1596 * x2 * y6 -
           41 * y8) *
              z2 -
          112 * (x6 - 15 * x4 * y2 + 15 * x2 * y4 - y6) * z4)) /
        (256. * r11);
    Y(i,j,counter,3) = (-3 * std::sqrt(13585 / two_pi) * y *
                   (-7 * x6 + 35 * x4 * y2 - 21 * x2 * y4 + y6) * z *
                   (41 * x2py21 - 112 * z2)) /
                  (256. * r11);
    counter++;

    Y(i,j,counter,0) =
        -(std::sqrt(40755 / two_pi) * (3 * x5 * y - 10 * x3 * y3 + 3 * x * y5) * z *
          (3 * x2py21 - 14 * z2)) /
        (32. * r9);
    Y(i,j,counter,1) =
        (3 * std::sqrt(40755 / two_pi) * y * z *
         (6 * x8 - 49 * x6 * y2 - 7 * x4 * y4 + 45 * x2 * y6 - 3 * y8 +
          11 * (-7 * x6 + 35 * x4 * y2 - 21 * x2 * y4 + y6) * z2 +
          14 * (5 * x4 - 10 * x2 * y2 + y4) * z4)) /
        (32. * r11);
    Y(i,j,counter,2) =
        (-3 * std::sqrt(40755 / two_pi) * x * z *
         (3 * x8 - 45 * x6 * y2 + 7 * x4 * y4 + 49 * x2 * y6 - 6 * y8 -
          11 * (x6 - 21 * x4 * y2 + 35 * x2 * y4 - 7 * y6) * z2 -
          14 * (x4 - 10 * x2 * y2 + 5 * y4) * z4)) /
        (32. * r11);
    Y(i,j,counter,3) =
        (-3 * std::sqrt(40755 / two_pi) * (3 * x5 * y - 10 * x3 * y3 + 3 * x * y5) *
         (x2py2 - 22 * x2py21 * z2 + 28 * z4)) /
        (32. * r11);
    counter++;

    Y(i,j,counter,0) = (3 * std::sqrt(2717 / two_pi) * (5 * x4 * y - 10 * x2 * y3 + y5) *
                  (x2py2 - 28 * x2py21 * z2 + 56 * z4)) /
                 (128. * r9);
    Y(i,j,counter,1) = (-15 * std::sqrt(2717 / two_pi) * x * y *
                   (x8 - 8 * x6 * y2 - 14 * x4 * y4 + 5 * y8 -
                    4 * x2py21 * (23 * x4 - 100 * x2 * y2 + 37 * y4) * z2 +
                    224 * (2 * x4 - 5 * x2 * y2 + y4) * z4 -
                    224 * (x - y) * (x + y) * z6)) /
                  (128. * r11);
    Y(i,j,counter,2) =
        (15 * std::sqrt(2717 / two_pi) *
         (x2 * x2py2 * (x4 - 10 * x2 * y2 + 5 * y4) +
          (-27 * x8 + 308 * x6 * y2 + 70 * x4 * y4 - 252 * x2 * y6 + 13 * y8) *
              z2 +
          28 * (x6 - 25 * x4 * y2 + 35 * x2 * y4 - 3 * y6) * z4 +
          56 * (x4 - 6 * x2 * y2 + y4) * z6)) /
        (128. * r11);
    Y(i,j,counter,3) =
        (-15 * std::sqrt(2717 / two_pi) * (5 * x4 * y - 10 * x2 * y3 + y5) * z *
         (13 * x2py2 - 84 * x2py21 * z2 + 56 * z4)) /
        (128. * r11);
    counter++;

    Y(i,j,counter,0) = (3 * std::sqrt(95095 / Pi) * x * (x - y) * y * (x + y) * z *
                  (x2py2 - 8 * x2py21 * z2 + 8 * z4)) /
                 (32. * r9);
    Y(i,j,counter,1) =
        (-3 * std::sqrt(95095 / Pi) * y * z *
         (x2py2 * (2 * x4 - 7 * x2 * y2 + y4) -
          x2py21 * (39 * x4 - 74 * x2 * y2 + 7 * y4) * z2 +
          88 * x2 * (x - y) * (x + y) * z4 + 8 * (-3 * x2 + y2) * z6)) /
        (32. * r11);
    Y(i,j,counter,2) =
        (3 * std::sqrt(95095 / Pi) * x * z *
         (x2py2 * (x4 - 7 * x2 * y2 + 2 * y4) -
          x2py21 * (7 * x4 - 74 * x2 * y2 + 39 * y4) * z2 -
          88 * (x - y) * y2 * (x + y) * z4 + 8 * (x2 - 3 * y2) * z6)) /
        (32. * r11);
    Y(i,j,counter,3) = (3 * std::sqrt(95095 / Pi) * x * (x - y) * y * (x + y) *
                   (x2py3 - 32 * x2py2 * z2 + 88 * x2py21 * z4 - 32 * z6)) /
                  (32. * r11);
    counter++;

    Y(i,j,counter,0) = (std::sqrt(21945 / two_pi) * y * (-3 * x2 + y2) *
                  (x2py3 - 36 * x2py2 * z2 + 120 * x2py21 * z4 - 64 * z6)) /
                 (128. * r9);
    Y(i,j,counter,1) =
        (3 * std::sqrt(21945 / two_pi) * x * y *
         ((x2 - 3 * y2) * x2py3 - 4 * (29 * x2 - 33 * y2) * x2py2 * z2 +
          16 * (51 * x2 - 31 * y2) * x2py21 * z4 +
          32 * (-29 * x2 + 5 * y2) * z6 + 128 * z8)) /
        (128. * r11);
    Y(i,j,counter,2) = (-3 * std::sqrt(21945 / two_pi) *
                   (x2 * (x2 - 3 * y2) * x2py3 -
                    x2py2 * (35 * x4 - 186 * x2 * y2 + 27 * y4) * z2 +
                    4 * x2py21 * (21 * x4 - 246 * x2 * y2 + 61 * y4) * z4 +
                    8 * (7 * x4 + 102 * x2 * y2 - 41 * y4) * z6 -
                    64 * (x - y) * (x + y) * z8)) /
                  (128. * r11);
    Y(i,j,counter,3) =
        (-3 * std::sqrt(21945 / two_pi) * y * (-3 * x2 + y2) * z *
         (27 * x2py3 - 244 * x2py2 * z2 + 328 * x2py21 * z4 - 64 * z6)) /
        (128. * r11);
    counter++;

    Y(i,j,counter,0) = (-3 * std::sqrt(1045 / two_pi) * x * y * z *
                  (7 * x2py3 - 70 * x2py2 * z2 + 112 * x2py21 * z4 - 32 * z6)) /
                 (32. * r9);
    Y(i,j,counter,1) =
        (-3 * std::sqrt(1045 / two_pi) * y * z *
         (-7 * (2 * x2 - y2) * x2py3 + 7 * (47 * x2 - 9 * y2) * x2py2 * z2 -
          14 * (73 * x2 - 3 * y2) * x2py21 * z4 + 16 * (37 * x2 + 5 * y2) * z6 -
          32 * z8)) /
        (32. * r11);
    Y(i,j,counter,2) =
        (-3 * std::sqrt(1045 / two_pi) * x * z *
         (7 * (x2 - 2 * y2) * x2py3 - 7 * (9 * x2 - 47 * y2) * x2py2 * z2 +
          14 * (3 * x2 - 73 * y2) * x2py21 * z4 + 16 * (5 * x2 + 37 * y2) * z6 -
          32 * z8)) /
        (32. * r11);
    Y(i,j,counter,3) = (-3 * std::sqrt(1045 / two_pi) * x * y *
                   (7 * x2py4 - 266 * x2py3 * z2 + 980 * x2py2 * z4 -
                    672 * x2py21 * z6 + 64 * z8)) /
                  (32. * r11);
    counter++;

    Y(i,j,counter,0) = (3 * std::sqrt(95 / Pi) * y *
                  (7 * x2py4 - 280 * x2py3 * z2 + 1120 * x2py2 * z4 -
                   896 * x2py21 * z6 + 128 * z8)) /
                 (256. * r9);
    Y(i,j,counter,1) = (-3 * std::sqrt(95 / Pi) * x * y *
                   (7 * x2py4 - 896 * x2py3 * z2 + 7280 * x2py2 * z4 -
                    10752 * x2py21 * z6 + 2944 * z8)) /
                  (256. * r11);
    Y(i,j,counter,2) = (3 * std::sqrt(95 / Pi) *
                   (7 * x2 * x2py4 - 7 * (39 * x2 - 89 * y2) * x2py3 * z2 +
                    280 * (3 * x2 - 23 * y2) * x2py2 * z4 +
                    224 * x2py21 * (x2 + 49 * y2) * z6 -
                    128 * (6 * x2 + 29 * y2) * z8 + 128 * std::pow(z, 10))) /
                  (256. * r11);
    Y(i,j,counter,3) = (-3 * std::sqrt(95 / Pi) * y * z *
                   (623 * x2py4 - 6440 * x2py3 * z2 + 10976 * x2py2 * z4 -
                    3712 * x2py21 * z6 + 128 * z8)) /
                  (256. * r11);
    counter++;

    Y(i,j,counter,0) = (std::sqrt(19 / Pi) * z *
                  (315 * x2py4 - 3360 * x2py3 * z2 + 6048 * x2py2 * z4 -
                   2304 * x2py21 * z6 + 128 * z8)) /
                 (256. * r9);
    Y(i,j,counter,1) = (-45 * std::sqrt(19 / Pi) * x * z *
                   (7 * x2py4 - 280 * x2py3 * z2 + 1120 * x2py2 * z4 -
                    896 * x2py21 * z6 + 128 * z8)) /
                  (256. * r11);
    Y(i,j,counter,2) = (-45 * std::sqrt(19 / Pi) * y * z *
                   (7 * x2py4 - 280 * x2py3 * z2 + 1120 * x2py2 * z4 -
                    896 * x2py21 * z6 + 128 * z8)) /
                  (256. * r11);
    Y(i,j,counter,3) = (45 * std::sqrt(19 / Pi) * x2py21 *
                   (7 * x2py4 - 280 * x2py3 * z2 + 1120 * x2py2 * z4 -
                    896 * x2py21 * z6 + 128 * z8)) /
                  (256. * r11);
    counter++;

    Y(i,j,counter,0) = (3 * std::sqrt(95 / Pi) * x *
                  (7 * x2py4 - 280 * x2py3 * z2 + 1120 * x2py2 * z4 -
                   896 * x2py21 * z6 + 128 * z8)) /
                 (256. * r9);
    Y(i,j,counter,1) = (3 * std::sqrt(95 / Pi) *
                   (7 * y2 * x2py4 + 7 * (89 * x2 - 39 * y2) * x2py3 * z2 +
                    280 * x2py2 * (-23 * x2 + 3 * y2) * z4 +
                    224 * x2py21 * (49 * x2 + y2) * z6 -
                    128 * (29 * x2 + 6 * y2) * z8 + 128 * std::pow(z, 10))) /
                  (256. * r11);
    Y(i,j,counter,2) = (-3 * std::sqrt(95 / Pi) * x * y *
                   (7 * x2py4 - 896 * x2py3 * z2 + 7280 * x2py2 * z4 -
                    10752 * x2py21 * z6 + 2944 * z8)) /
                  (256. * r11);
    Y(i,j,counter,3) = (-3 * std::sqrt(95 / Pi) * x * z *
                   (623 * x2py4 - 6440 * x2py3 * z2 + 10976 * x2py2 * z4 -
                    3712 * x2py21 * z6 + 128 * z8)) /
                  (256. * r11);
    counter++;

    Y(i,j,counter,0) = (-3 * std::sqrt(1045 / two_pi) * (x - y) * (x + y) * z *
                  (7 * x2py3 - 70 * x2py2 * z2 + 112 * x2py21 * z4 - 32 * z6)) /
                 (64. * r9);
    Y(i,j,counter,1) =
        (3 * std::sqrt(1045 / two_pi) * x * z *
         (7 * (x2 - 5 * y2) * x2py3 - 14 * (19 * x2 - 37 * y2) * x2py2 * z2 +
          28 * (35 * x2 - 41 * y2) * x2py21 * z4 +
          32 * (-21 * x2 + 11 * y2) * z6 + 64 * z8)) /
        (64. * r11);
    Y(i,j,counter,2) =
        (3 * std::sqrt(1045 / two_pi) * y * z *
         (7 * (5 * x2 - y2) * x2py3 - 14 * (37 * x2 - 19 * y2) * x2py2 * z2 +
          28 * (41 * x2 - 35 * y2) * x2py21 * z4 +
          32 * (-11 * x2 + 21 * y2) * z6 - 64 * z8)) /
        (64. * r11);
    Y(i,j,counter,3) = (-3 * std::sqrt(1045 / two_pi) * (x - y) * (x + y) *
                   (7 * x2py4 - 266 * x2py3 * z2 + 980 * x2py2 * z4 -
                    672 * x2py21 * z6 + 64 * z8)) /
                  (64. * r11);
    counter++;

    Y(i,j,counter,0) = -(std::sqrt(21945 / two_pi) * (x3 - 3 * xy2) *
                   (x2py3 - 36 * x2py2 * z2 + 120 * x2py21 * z4 - 64 * z6)) /
                 (128. * r9);
    Y(i,j,counter,1) = (-3 * std::sqrt(21945 / two_pi) *
                   (-(y2 * (-3 * x2 + y2) * x2py3) +
                    x2py2 * (27 * x4 - 186 * x2 * y2 + 35 * y4) * z2 -
                    4 * x2py21 * (61 * x4 - 246 * x2 * y2 + 21 * y4) * z4 +
                    8 * (41 * x4 - 102 * x2 * y2 - 7 * y4) * z6 +
                    64 * (-x + y) * (x + y) * z8)) /
                  (128. * r11);
    Y(i,j,counter,2) =
        (3 * std::sqrt(21945 / two_pi) * x * y *
         ((3 * x2 - y2) * x2py3 - 4 * (33 * x2 - 29 * y2) * x2py2 * z2 +
          16 * (31 * x2 - 51 * y2) * x2py21 * z4 +
          32 * (-5 * x2 + 29 * y2) * z6 - 128 * z8)) /
        (128. * r11);
    Y(i,j,counter,3) =
        (3 * std::sqrt(21945 / two_pi) * (x3 - 3 * xy2) * z *
         (27 * x2py3 - 244 * x2py2 * z2 + 328 * x2py21 * z4 - 64 * z6)) /
        (128. * r11);
    counter++;

    Y(i,j,counter,0) = (3 * std::sqrt(95095 / Pi) * (x4 - 6 * x2 * y2 + y4) * z *
                  (x2py2 - 8 * x2py21 * z2 + 8 * z4)) /
                 (128. * r9);
    Y(i,j,counter,1) =
        (-3 * std::sqrt(95095 / Pi) * x * z *
         (x2py2 * (x4 - 22 * x2 * y2 + 17 * y4) -
          16 * (x - 3 * y) * (x + 3 * y) * (2 * x2 - y2) * x2py21 * z2 +
          88 * (x4 - 6 * x2 * y2 + y4) * z4 - 32 * (x2 - 3 * y2) * z6)) /
        (128. * r11);
    Y(i,j,counter,2) =
        (-3 * std::sqrt(95095 / Pi) * y * z *
         (x2py2 * (17 * x4 - 22 * x2 * y2 + y4) -
          16 * (3 * x - y) * (3 * x + y) * (x2 - 2 * y2) * x2py21 * z2 +
          88 * (x4 - 6 * x2 * y2 + y4) * z4 + 32 * (3 * x2 - y2) * z6)) /
        (128. * r11);
    Y(i,j,counter,3) = (3 * std::sqrt(95095 / Pi) * (x4 - 6 * x2 * y2 + y4) *
                   (x2py3 - 32 * x2py2 * z2 + 88 * x2py21 * z4 - 32 * z6)) /
                  (128. * r11);
    counter++;

    Y(i,j,counter,0) = (3 * std::sqrt(2717 / two_pi) * (x5 - 10 * x3 * y2 + 5 * x * y4) *
                  (x2py2 - 28 * x2py21 * z2 + 56 * z4)) /
                 (128. * r9);
    Y(i,j,counter,1) =
        (15 * std::sqrt(2717 / two_pi) *
         (y2 * x2py2 * (5 * x4 - 10 * x2 * y2 + y4) +
          x2py21 * (13 * x6 - 265 * x4 * y2 + 335 * x2 * y4 - 27 * y6) * z2 +
          28 * (-3 * x6 + 35 * x4 * y2 - 25 * x2 * y4 + y6) * z4 +
          56 * (x4 - 6 * x2 * y2 + y4) * z6)) /
        (128. * r11);
    Y(i,j,counter,2) = (-15 * std::sqrt(2717 / two_pi) * x * y *
                   (5 * x8 - 14 * x4 * y4 - 8 * x2 * y6 + y8 -
                    4 * x2py21 * (37 * x4 - 100 * x2 * y2 + 23 * y4) * z2 +
                    224 * (x4 - 5 * x2 * y2 + 2 * y4) * z4 +
                    224 * (x - y) * (x + y) * z6)) /
                  (128. * r11);
    Y(i,j,counter,3) =
        (-15 * std::sqrt(2717 / two_pi) * (x5 - 10 * x3 * y2 + 5 * x * y4) * z *
         (13 * x2py2 - 84 * x2py21 * z2 + 56 * z4)) /
        (128. * r11);
    counter++;

    Y(i,j,counter,0) =
        -(std::sqrt(40755 / two_pi) * (x6 - 15 * x4 * y2 + 15 * x2 * y4 - y6) * z *
          (3 * x2py21 - 14 * z2)) /
        (64. * r9);
    Y(i,j,counter,1) = (3 * std::sqrt(40755 / two_pi) * x * z *
                   (x8 - 50 * x6 * y2 + 84 * x4 * y4 + 98 * x2 * y6 - 37 * y8 -
                    22 * (x6 - 21 * x4 * y2 + 35 * x2 * y4 - 7 * y6) * z2 +
                    28 * (x4 - 10 * x2 * y2 + 5 * y4) * z4)) /
                  (64. * r11);
    Y(i,j,counter,2) = (-3 * std::sqrt(40755 / two_pi) * y * z *
                   (-37 * x8 + 98 * x6 * y2 + 84 * x4 * y4 - 50 * x2 * y6 + y8 +
                    22 * (7 * x6 - 35 * x4 * y2 + 21 * x2 * y4 - y6) * z2 +
                    28 * (5 * x4 - 10 * x2 * y2 + y4) * z4)) /
                  (64. * r11);
    Y(i,j,counter,3) =
        (-3 * std::sqrt(40755 / two_pi) * (x6 - 15 * x4 * y2 + 15 * x2 * y4 - y6) *
         (x2py2 - 22 * x2py21 * z2 + 28 * z4)) /
        (64. * r11);
    counter++;

    Y(i,j,counter,0) = (-3 * std::sqrt(13585 / two_pi) *
                  (std::pow(x, 7) - 21 * x5 * y2 + 35 * x3 * y4 - 7 * x * y6) *
                  (x2 + y2 - 16 * z2)) /
                 (256. * r9);
    Y(i,j,counter,1) =
        (-3 * std::sqrt(13585 / two_pi) *
         (-7 * y2 * x2py21 * (-7 * x6 + 35 * x4 * y2 - 21 * x2 * y4 + y6) +
          (41 * x8 - 1596 * x6 * y2 + 5110 * x4 * y4 - 2492 * x2 * y6 +
           105 * y8) *
              z2 -
          112 * (x6 - 15 * x4 * y2 + 15 * x2 * y4 - y6) * z4)) /
        (256. * r11);
    Y(i,j,counter,2) =
        (3 * std::sqrt(13585 / two_pi) * x * y *
         (7 * x2py21 * (7 * x6 - 35 * x4 * y2 + 21 * x2 * y4 - y6) -
          8 * (97 * x6 - 567 * x4 * y2 + 455 * x2 * y4 - 49 * y6) * z2 -
          224 * (3 * x4 - 10 * x2 * y2 + 3 * y4) * z4)) /
        (256. * r11);
    Y(i,j,counter,3) = (3 * std::sqrt(13585 / two_pi) *
                   (std::pow(x, 7) - 21 * x5 * y2 + 35 * x3 * y4 - 7 * x * y6) * z *
                   (41 * x2py21 - 112 * z2)) /
                  (256. * r11);
    counter++;

    Y(i,j,counter,0) = (3 * std::sqrt(230945 / Pi) *
                  (x8 - 28 * x6 * y2 + 70 * x4 * y4 - 28 * x2 * y6 + y8) * z) /
                 (256. * r9);
    Y(i,j,counter,1) =
        (-3 * std::sqrt(230945 / Pi) * x * z *
         (x8 - 92 * x6 * y2 + 518 * x4 * y4 - 476 * x2 * y6 + 65 * y8 -
          8 * (x6 - 21 * x4 * y2 + 35 * x2 * y4 - 7 * y6) * z2)) /
        (256. * r11);
    Y(i,j,counter,2) =
        (-3 * std::sqrt(230945 / Pi) * y * z *
         (65 * x8 - 476 * x6 * y2 + 518 * x4 * y4 - 92 * x2 * y6 + y8 +
          8 * (7 * x6 - 35 * x4 * y2 + 21 * x2 * y4 - y6) * z2)) /
        (256. * r11);
    Y(i,j,counter,3) = (3 * std::sqrt(230945 / Pi) *
                   (x8 - 28 * x6 * y2 + 70 * x4 * y4 - 28 * x2 * y6 + y8) *
                   (x2 + y2 - 8 * z2)) /
                  (256. * r11);
    counter++;

    Y(i,j,counter,0) =
        (std::sqrt(230945 / two_pi) * (std::pow(x, 9) - 36 * std::pow(x, 7) * y2 +
                                  126 * x5 * y4 - 84 * x3 * y6 + 9 * x * y8)) /
        (256. * r9);
    Y(i,j,counter,1) =
        (9 * std::sqrt(230945 / two_pi) *
         (y8 * (y2 + z2) - 28 * x6 * y2 * (3 * y2 + z2) + x8 * (9 * y2 + z2) +
          14 * x4 * (9 * y6 + 5 * y4 * z2) - 4 * x2 * (9 * y8 + 7 * y6 * z2))) /
        (256. * r11);
    Y(i,j,counter,2) = (-9 * std::sqrt(230945 / two_pi) * x * y *
                   (9 * x8 - 84 * x6 * y2 + 126 * x4 * y4 - 36 * x2 * y6 + y8 +
                    8 * (x6 - 7 * x4 * y2 + 7 * x2 * y4 - y6) * z2)) /
                  (256. * r11);
    Y(i,j,counter,3) = (-9 * std::sqrt(230945 / two_pi) *
                   (std::pow(x, 9) - 36 * std::pow(x, 7) * y2 + 126 * x5 * y4 -
                    84 * x3 * y6 + 9 * x * y8) *
                   z) /
                  (256. * r11);
    counter++;
  }
  if (l == 9)
    return;

  { // l_counter == 10
    Y(i,j,counter,0) = (std::sqrt(969969 / two_pi) * x * y * (5 * x4 - 10 * x2 * y2 + y4) *
                  (x4 - 10 * x2 * y2 + 5 * y4)) /
                 (256. * r10);
    Y(i,j,counter,1) =
        (5 * std::sqrt(969969 / two_pi) * y *
         (-std::pow(x, 10) + y8 * (y2 + z2) + 9 * x8 * (5 * y2 + z2) -
          42 * x6 * (5 * y4 + 2 * y2 * z2) + 42 * x4 * (5 * y6 + 3 * y4 * z2) -
          9 * x2 * (5 * y8 + 4 * y6 * z2))) /
        (256. * r12);
    Y(i,j,counter,2) =
        (5 * std::sqrt(969969 / two_pi) * x *
         (std::pow(x, 10) - 45 * x8 * y2 + 210 * x6 * y4 - 210 * x4 * y6 +
          45 * x2 * y8 - std::pow(y, 10) +
          (x8 - 36 * x6 * y2 + 126 * x4 * y4 - 84 * x2 * y6 + 9 * y8) * z2)) /
        (256. * r12);
    Y(i,j,counter,3) =
        (-5 * std::sqrt(969969 / two_pi) * x * y * (5 * x4 - 10 * x2 * y2 + y4) *
         (x4 - 10 * x2 * y2 + 5 * y4) * z) /
        (128. * r12);
    counter++;

    Y(i,j,counter,0) = (std::sqrt(4849845 / two_pi) *
                  (9 * x8 * y - 84 * x6 * y3 + 126 * x4 * y5 -
                   36 * x2 * std::pow(y, 7) + std::pow(y, 9)) *
                  z) /
                 (256. * r10);
    Y(i,j,counter,1) =
        (std::sqrt(4849845 / two_pi) * x * y * z *
         (-9 * x8 + 204 * x6 * y2 - 630 * x4 * y4 + 396 * x2 * y6 - 41 * y8 +
          36 * (x6 - 7 * x4 * y2 + 7 * x2 * y4 - y6) * z2)) /
        (128. * r12);
    Y(i,j,counter,2) =
        (std::sqrt(4849845 / two_pi) * z *
         (9 * std::pow(x, 10) - 333 * x8 * y2 + 1218 * x6 * y4 - 882 * x4 * y6 +
          117 * x2 * y8 - std::pow(y, 10) +
          9 * (x8 - 28 * x6 * y2 + 70 * x4 * y4 - 28 * x2 * y6 + y8) * z2)) /
        (256. * r12);
    Y(i,j,counter,3) = (std::sqrt(4849845 / two_pi) *
                   (9 * x8 * y - 84 * x6 * y3 + 126 * x4 * y5 -
                    36 * x2 * std::pow(y, 7) + std::pow(y, 9)) *
                   (x2 + y2 - 9 * z2)) /
                  (256. * r12);
    counter++;

    Y(i,j,counter,0) =
        -(std::sqrt(255255 / Pi) * x * y * (x6 - 7 * x4 * y2 + 7 * x2 * y4 - y6) *
          (x2 + y2 - 18 * z2)) /
        (64. * r10);
    Y(i,j,counter,1) =
        (std::sqrt(255255 / Pi) * y *
         (std::pow(x, 10) + y6 * (y2 - 18 * z2) * (y2 + z2) -
          9 * x8 * (3 * y2 + 7 * z2) + 42 * x6 * (y4 + 19 * y2 * z2 + 3 * z4) +
          9 * x2 * y4 * (-3 * y4 + 58 * y2 * z2 + 42 * z4) +
          42 * x4 * (y6 - 36 * y4 * z2 - 15 * y2 * z4))) /
        (64. * r12);
    Y(i,j,counter,2) = -(std::sqrt(255255 / Pi) * x *
                    (std::pow(x, 10) - 27 * x8 * y2 + 42 * x6 * y4 + 42 * x4 * y6 -
                     27 * x2 * y8 + std::pow(y, 10) +
                     (-17 * x8 + 522 * x6 * y2 - 1512 * x4 * y4 +
                      798 * x2 * y6 - 63 * y8) *
                         z2 -
                     18 * (x6 - 21 * x4 * y2 + 35 * x2 * y4 - 7 * y6) * z4)) /
                  (64. * r12);
    Y(i,j,counter,3) =
        (std::sqrt(255255 / Pi) * x * y * (x6 - 7 * x4 * y2 + 7 * x2 * y4 - y6) * z *
         (23 * x2py21 - 72 * z2)) /
        (32. * r12);
    counter++;

    Y(i,j,counter,0) = (3 * std::sqrt(85085 / two_pi) * y *
                  (-7 * x6 + 35 * x4 * y2 - 21 * x2 * y4 + y6) * z *
                  (3 * x2py21 - 16 * z2)) /
                 (256. * r10);
    Y(i,j,counter,1) =
        (3 * std::sqrt(85085 / two_pi) * x * y * z *
         (3 * x2py21 * (7 * x6 - 91 * x4 * y2 + 133 * x2 * y4 - 25 * y6) -
          4 * (77 * x6 - 567 * x4 * y2 + 595 * x2 * y4 - 89 * y6) * z2 +
          112 * (3 * x4 - 10 * x2 * y2 + 3 * y4) * z4)) /
        (128. * r12);
    Y(i,j,counter,2) =
        (3 * std::sqrt(85085 / two_pi) * z *
         (-3 * x2py21 *
              (7 * x8 - 154 * x6 * y2 + 280 * x4 * y4 - 70 * x2 * y6 + y8) +
          (91 * x8 - 2436 * x6 * y2 + 5810 * x4 * y4 - 2212 * x2 * y6 +
           75 * y8) *
              z2 +
          112 * (x6 - 15 * x4 * y2 + 15 * x2 * y4 - y6) * z4)) /
        (256. * r12);
    Y(i,j,counter,3) = (3 * std::sqrt(85085 / two_pi) * y *
                   (-7 * x6 + 35 * x4 * y2 - 21 * x2 * y4 + y6) *
                   (3 * x2py2 - 75 * x2py21 * z2 + 112 * z4)) /
                  (256. * r12);
    counter++;

    Y(i,j,counter,0) =
        (3 * std::sqrt(5005 / two_pi) * (3 * x5 * y - 10 * x3 * y3 + 3 * x * y5) *
         (3 * x2py2 - 96 * x2py21 * z2 + 224 * z4)) /
        (256. * r10);
    Y(i,j,counter,1) =
        (3 * std::sqrt(5005 / two_pi) * y *
         (-9 * x2py2 * (x6 - 15 * x4 * y2 + 15 * x2 * y4 - y6) +
          3 * x2py21 * (315 * x6 - 2135 * x4 * y2 + 1617 * x2 * y4 - 93 * y6) *
              z2 -
          128 * (42 * x6 - 175 * x4 * y2 + 84 * x2 * y4 - 3 * y6) * z4 +
          672 * (5 * x4 - 10 * x2 * y2 + y4) * z6)) /
        (256. * r12);
    Y(i,j,counter,2) =
        (3 * std::sqrt(5005 / two_pi) * x *
         (9 * x2py2 * (x6 - 15 * x4 * y2 + 15 * x2 * y4 - y6) -
          3 * x2py21 * (93 * x6 - 1617 * x4 * y2 + 2135 * x2 * y4 - 315 * y6) *
              z2 +
          128 * (3 * x6 - 84 * x4 * y2 + 175 * x2 * y4 - 42 * y6) * z4 +
          672 * (x4 - 10 * x2 * y2 + 5 * y4) * z6)) /
        (256. * r12);
    Y(i,j,counter,3) =
        (-3 * std::sqrt(5005 / two_pi) * (3 * x5 * y - 10 * x3 * y3 + 3 * x * y5) *
         z * (111 * x2py2 - 832 * x2py21 * z2 + 672 * z4)) /
        (128. * r12);
    counter++;

    Y(i,j,counter,0) = (3 * std::sqrt(1001 / two_pi) * (5 * x4 * y - 10 * x2 * y3 + y5) *
                  z * (15 * x2py2 - 140 * x2py21 * z2 + 168 * z4)) /
                 (128. * r10);
    Y(i,j,counter,1) = (-15 * std::sqrt(1001 / two_pi) * x * y * z *
                   (3 * x2py2 * (5 * x4 - 30 * x2 * y2 + 13 * y4) -
                    4 * x2py21 * (85 * x4 - 295 * x2 * y2 + 92 * y4) * z2 +
                    28 * (33 * x4 - 70 * x2 * y2 + 9 * y4) * z4 -
                    336 * (x - y) * (x + y) * z6)) /
                  (64. * r12);
    Y(i,j,counter,2) = (15 * std::sqrt(1001 / two_pi) * z *
                   (3 * x2py2 * (5 * x6 - 55 * x4 * y2 + 35 * x2 * y4 - y6) +
                    (-125 * x8 + 1680 * x6 * y2 + 70 * x4 * y4 -
                     1624 * x2 * y6 + 111 * y8) *
                        z2 +
                    28 * (x6 - 75 * x4 * y2 + 135 * x2 * y4 - 13 * y6) * z4 +
                    168 * (x4 - 6 * x2 * y2 + y4) * z6)) /
                  (128. * r12);
    Y(i,j,counter,3) =
        (15 * std::sqrt(1001 / two_pi) * (5 * x4 * y - 10 * x2 * y3 + y5) *
         (3 * x2py3 - 111 * x2py2 * z2 + 364 * x2py21 * z4 - 168 * z6)) /
        (128. * r12);
    counter++;

    Y(i,j,counter,0) = (-3 * std::sqrt(5005 / Pi) * x * (x - y) * y * (x + y) *
                  (x2py3 - 42 * x2py2 * z2 + 168 * x2py21 * z4 - 112 * z6)) /
                 (64. * r10);
    Y(i,j,counter,1) = (3 * std::sqrt(5005 / Pi) * y *
                   (x2py3 * (x4 - 6 * x2 * y2 + y4) -
                    x2py2 * (135 * x4 - 340 * x2 * y2 + 41 * y4) * z2 +
                    126 * x2py21 * (9 * x4 - 14 * x2 * y2 + y4) * z4 +
                    56 * (-29 * x4 + 24 * x2 * y2 + y4) * z6 +
                    112 * (3 * x2 - y2) * z8)) /
                  (64. * r12);
    Y(i,j,counter,2) =
        (-3 * std::sqrt(5005 / Pi) * x *
         (x2py3 * (x4 - 6 * x2 * y2 + y4) -
          x2py2 * (41 * x4 - 340 * x2 * y2 + 135 * y4) * z2 +
          126 * x2py21 * (x4 - 14 * x2 * y2 + 9 * y4) * z4 +
          56 * (x4 + 24 * x2 * y2 - 29 * y4) * z6 - 112 * (x2 - 3 * y2) * z8)) /
        (64. * r12);
    Y(i,j,counter,3) =
        (3 * std::sqrt(5005 / Pi) * x * (x - y) * y * (x + y) * z *
         (47 * x2py3 - 504 * x2py2 * z2 + 840 * x2py21 * z4 - 224 * z6)) /
        (32. * r12);
    counter++;

    Y(i,j,counter,0) = (3 * std::sqrt(5005 / two_pi) * y * (-3 * x2 + y2) * z *
                  (7 * x2py3 - 84 * x2py2 * z2 + 168 * x2py21 * z4 - 64 * z6)) /
                 (128. * r10);
    Y(i,j,counter,1) =
        (3 * std::sqrt(5005 / two_pi) * x * y * z *
         (7 * (3 * x2 - 5 * y2) * x2py3 - 84 * (7 * x2 - 6 * y2) * x2py2 * z2 +
          84 * (27 * x2 - 13 * y2) * x2py21 * z4 +
          16 * (-111 * x2 + 11 * y2) * z6 + 192 * z8)) /
        (64. * r12);
    Y(i,j,counter,2) = (3 * std::sqrt(5005 / two_pi) * z *
                   (-7 * x2py3 * (3 * x4 - 12 * x2 * y2 + y4) +
                    21 * x2py2 * (11 * x4 - 78 * x2 * y2 + 15 * y4) * z2 -
                    84 * x2py21 * (3 * x4 - 60 * x2 * y2 + 17 * y4) * z4 -
                    8 * (39 * x4 + 366 * x2 * y2 - 161 * y4) * z6 +
                    192 * (x - y) * (x + y) * z8)) /
                  (128. * r12);
    Y(i,j,counter,3) = (3 * std::sqrt(5005 / two_pi) * y * (-3 * x2 + y2) *
                   (7 * x2py4 - 315 * x2py3 * z2 + 1428 * x2py2 * z4 -
                    1288 * x2py21 * z6 + 192 * z8)) /
                  (128. * r12);
    counter++;

    Y(i,j,counter,0) = (3 * c11 * x * y *
                  (7 * x2py4 - 336 * x2py3 * z2 + 1680 * x2py2 * z4 -
                   1792 * x2py21 * z6 + 384 * z8)) /
                 (256. * r10);
    Y(i,j,counter,1) = (3 * c11 * y *
                   (-7 * (x - y) * (x + y) * x2py4 +
                    7 * (153 * x2 - 47 * y2) * x2py3 * z2 -
                    1344 * (8 * x2 - y2) * x2py2 * z4 +
                    112 * (187 * x2 - y2) * x2py21 * z6 -
                    128 * (69 * x2 + 11 * y2) * z8 + 384 * std::pow(z, 10))) /
                  (256. * r12);
    Y(i,j,counter,2) = (3 * c11 * x *
                   (7 * (x - y) * (x + y) * x2py4 -
                    7 * (47 * x2 - 153 * y2) * x2py3 * z2 +
                    1344 * (x2 - 8 * y2) * x2py2 * z4 -
                    112 * (x2 - 187 * y2) * x2py21 * z6 -
                    128 * (11 * x2 + 69 * y2) * z8 + 384 * std::pow(z, 10))) /
                  (256. * r12);
    Y(i,j,counter,3) = (-3 * c11 * x * y * z *
                   (371 * x2py4 - 4704 * x2py3 * z2 + 10416 * x2py2 * z4 -
                    5120 * x2py21 * z6 + 384 * z8)) /
                  (128. * r12);
    counter++;

    Y(i,j,counter,0) = (std::sqrt(1155 / Pi) * y * z *
                  (63 * x2py4 - 840 * x2py3 * z2 + 2016 * x2py2 * z4 -
                   1152 * x2py21 * z6 + 128 * z8)) /
                 (256. * r10);
    Y(i,j,counter,1) = -(std::sqrt(1155 / Pi) * x * y * z *
                    (63 * x2py4 - 1932 * x2py3 * z2 + 8568 * x2py2 * z4 -
                     8640 * x2py21 * z6 + 1792 * z8)) /
                  (128. * r12);
    Y(i,j,counter,2) = (std::sqrt(1155 / Pi) * z *
                   (63 * (x - y) * (x + y) * x2py4 -
                    21 * (37 * x2 - 147 * y2) * x2py3 * z2 +
                    168 * (7 * x2 - 95 * y2) * x2py2 * z4 +
                    864 * x2py21 * (x2 + 21 * y2) * z6 -
                    512 * (2 * x2 + 9 * y2) * z8 + 128 * std::pow(z, 10))) /
                  (256. * r12);
    Y(i,j,counter,3) =
        (std::sqrt(1155 / Pi) * y *
         (63 * std::pow(x2 + y2, 5) - 3087 * x2py4 * z2 + 15960 * x2py3 * z4 -
          18144 * x2py2 * z6 + 4608 * x2py21 * z8 - 128 * std::pow(z, 10))) /
        (256. * r12);
    counter++;

    Y(i,j,counter,0) = (std::sqrt(21 / Pi) * (-63 * std::pow(x2 + y2, 5) + 3150 * x2py4 * z2 -
                                   16800 * x2py3 * z4 + 20160 * x2py2 * z6 -
                                   5760 * x2py21 * z8 + 256 * std::pow(z, 10))) /
                 (512. * r10);
    Y(i,j,counter,1) = (-55 * std::sqrt(21 / Pi) * x * z2 *
                   (63 * x2py4 - 840 * x2py3 * z2 + 2016 * x2py2 * z4 -
                    1152 * x2py21 * z6 + 128 * z8)) /
                  (256. * r12);
    Y(i,j,counter,2) = (-55 * std::sqrt(21 / Pi) * y * z2 *
                   (63 * x2py4 - 840 * x2py3 * z2 + 2016 * x2py2 * z4 -
                    1152 * x2py21 * z6 + 128 * z8)) /
                  (256. * r12);
    Y(i,j,counter,3) = (55 * std::sqrt(21 / Pi) * x2py21 * z *
                   (63 * x2py4 - 840 * x2py3 * z2 + 2016 * x2py2 * z4 -
                    1152 * x2py21 * z6 + 128 * z8)) /
                  (256. * r12);
    counter++;

    Y(i,j,counter,0) = (std::sqrt(1155 / Pi) * x * z *
                  (63 * x2py4 - 840 * x2py3 * z2 + 2016 * x2py2 * z4 -
                   1152 * x2py21 * z6 + 128 * z8)) /
                 (256. * r10);
    Y(i,j,counter,1) = (std::sqrt(1155 / Pi) * z *
                   (-63 * (x - y) * (x + y) * x2py4 +
                    21 * (147 * x2 - 37 * y2) * x2py3 * z2 -
                    168 * (95 * x2 - 7 * y2) * x2py2 * z4 +
                    864 * x2py21 * (21 * x2 + y2) * z6 -
                    512 * (9 * x2 + 2 * y2) * z8 + 128 * std::pow(z, 10))) /
                  (256. * r12);
    Y(i,j,counter,2) = -(std::sqrt(1155 / Pi) * x * y * z *
                    (63 * x2py4 - 1932 * x2py3 * z2 + 8568 * x2py2 * z4 -
                     8640 * x2py21 * z6 + 1792 * z8)) /
                  (128. * r12);
    Y(i,j,counter,3) =
        (std::sqrt(1155 / Pi) * x *
         (63 * std::pow(x2 + y2, 5) - 3087 * x2py4 * z2 + 15960 * x2py3 * z4 -
          18144 * x2py2 * z6 + 4608 * x2py21 * z8 - 128 * std::pow(z, 10))) /
        (256. * r12);
    counter++;

    Y(i,j,counter,0) = (3 * c11 * (x - y) * (x + y) *
                  (7 * x2py4 - 336 * x2py3 * z2 + 1680 * x2py2 * z4 -
                   1792 * x2py21 * z6 + 384 * z8)) /
                 (512. * r10);
    Y(i,j,counter,1) = (3 * c11 * x *
                   (14 * y2 * x2py4 + 7 * (53 * x2 - 147 * y2) * x2py3 * z2 +
                    672 * x2py2 * (-7 * x2 + 11 * y2) * z4 +
                    112 * (93 * x2 - 95 * y2) * x2py21 * z6 +
                    256 * (-20 * x2 + 9 * y2) * z8 + 384 * std::pow(z, 10))) /
                  (256. * r12);
    Y(i,j,counter,2) = (-3 * c11 * y *
                   (14 * x2 * x2py4 - 7 * (147 * x2 - 53 * y2) * x2py3 * z2 +
                    672 * (11 * x2 - 7 * y2) * x2py2 * z4 -
                    112 * (95 * x2 - 93 * y2) * x2py21 * z6 +
                    256 * (9 * x2 - 20 * y2) * z8 + 384 * std::pow(z, 10))) /
                  (256. * r12);
    Y(i,j,counter,3) = (-3 * c11 * (x - y) * (x + y) * z *
                   (371 * x2py4 - 4704 * x2py3 * z2 + 10416 * x2py2 * z4 -
                    5120 * x2py21 * z6 + 384 * z8)) /
                  (256. * r12);
    counter++;

    Y(i,j,counter,0) = (-3 * std::sqrt(5005 / two_pi) * (x3 - 3 * xy2) * z *
                  (7 * x2py3 - 84 * x2py2 * z2 + 168 * x2py21 * z4 - 64 * z6)) /
                 (128. * r10);
    Y(i,j,counter,1) = (3 * std::sqrt(5005 / two_pi) * z *
                   (7 * x2py3 * (x4 - 12 * x2 * y2 + 3 * y4) -
                    21 * x2py2 * (15 * x4 - 78 * x2 * y2 + 11 * y4) * z2 +
                    84 * x2py21 * (17 * x4 - 60 * x2 * y2 + 3 * y4) * z4 +
                    8 * (-161 * x4 + 366 * x2 * y2 + 39 * y4) * z6 +
                    192 * (x - y) * (x + y) * z8)) /
                  (128. * r12);
    Y(i,j,counter,2) =
        (3 * std::sqrt(5005 / two_pi) * x * y * z *
         (7 * (5 * x2 - 3 * y2) * x2py3 - 84 * (6 * x2 - 7 * y2) * x2py2 * z2 +
          84 * (13 * x2 - 27 * y2) * x2py21 * z4 +
          16 * (-11 * x2 + 111 * y2) * z6 - 192 * z8)) /
        (64. * r12);
    Y(i,j,counter,3) = (-3 * std::sqrt(5005 / two_pi) * (x3 - 3 * xy2) *
                   (7 * x2py4 - 315 * x2py3 * z2 + 1428 * x2py2 * z4 -
                    1288 * x2py21 * z6 + 192 * z8)) /
                  (128. * r12);
    counter++;

    Y(i,j,counter,0) = (-3 * std::sqrt(5005 / Pi) * (x4 - 6 * x2 * y2 + y4) *
                  (x2py3 - 42 * x2py2 * z2 + 168 * x2py21 * z4 - 112 * z6)) /
                 (256. * r10);
    Y(i,j,counter,1) = (-3 * std::sqrt(5005 / Pi) * x *
                   (-8 * y2 * (-x + y) * (x + y) * x2py3 +
                    x2py2 * (47 * x4 - 610 * x2 * y2 + 375 * y4) * z2 -
                    504 * x2py21 * (x4 - 8 * x2 * y2 + 3 * y4) * z4 +
                    56 * (15 * x4 - 82 * x2 * y2 + 7 * y4) * z6 -
                    224 * (x2 - 3 * y2) * z8)) /
                  (128. * r12);
    Y(i,j,counter,2) = (-3 * std::sqrt(5005 / Pi) * y *
                   (-8 * x2 * (x - y) * (x + y) * x2py3 +
                    x2py2 * (375 * x4 - 610 * x2 * y2 + 47 * y4) * z2 -
                    504 * x2py21 * (3 * x4 - 8 * x2 * y2 + y4) * z4 +
                    56 * (7 * x4 - 82 * x2 * y2 + 15 * y4) * z6 +
                    224 * (3 * x2 - y2) * z8)) /
                  (128. * r12);
    Y(i,j,counter,3) =
        (3 * std::sqrt(5005 / Pi) * (x4 - 6 * x2 * y2 + y4) * z *
         (47 * x2py3 - 504 * x2py2 * z2 + 840 * x2py21 * z4 - 224 * z6)) /
        (128. * r12);
    counter++;

    Y(i,j,counter,0) = (3 * std::sqrt(1001 / two_pi) * (x5 - 10 * x3 * y2 + 5 * x * y4) *
                  z * (15 * x2py2 - 140 * x2py21 * z2 + 168 * z4)) /
                 (128. * r10);
    Y(i,j,counter,1) =
        (15 * std::sqrt(1001 / two_pi) * z *
         (-3 * x2py2 * (x6 - 35 * x4 * y2 + 55 * x2 * y4 - 5 * y6) +
          x2py21 * (111 * x6 - 1735 * x4 * y2 + 1805 * x2 * y4 - 125 * y6) *
              z2 -
          28 * (13 * x6 - 135 * x4 * y2 + 75 * x2 * y4 - y6) * z4 +
          168 * (x4 - 6 * x2 * y2 + y4) * z6)) /
        (128. * r12);
    Y(i,j,counter,2) = (-15 * std::sqrt(1001 / two_pi) * x * y * z *
                   (3 * x2py2 * (13 * x4 - 30 * x2 * y2 + 5 * y4) -
                    4 * x2py21 * (92 * x4 - 295 * x2 * y2 + 85 * y4) * z2 +
                    28 * (9 * x4 - 70 * x2 * y2 + 33 * y4) * z4 +
                    336 * (x - y) * (x + y) * z6)) /
                  (64. * r12);
    Y(i,j,counter,3) =
        (15 * std::sqrt(1001 / two_pi) * (x5 - 10 * x3 * y2 + 5 * x * y4) *
         (3 * x2py3 - 111 * x2py2 * z2 + 364 * x2py21 * z4 - 168 * z6)) /
        (128. * r12);
    counter++;

    Y(i,j,counter,0) =
        (3 * std::sqrt(5005 / two_pi) * (x6 - 15 * x4 * y2 + 15 * x2 * y4 - y6) *
         (3 * x2py2 - 96 * x2py21 * z2 + 224 * z4)) /
        (512. * r10);
    Y(i,j,counter,1) =
        (3 * std::sqrt(5005 / two_pi) * x *
         (18 * y2 * x2py2 * (3 * x4 - 10 * x2 * y2 + 3 * y4) +
          3 * x2py21 * (37 * x6 - 1113 * x4 * y2 + 2415 * x2 * y4 - 595 * y6) *
              z2 +
          64 * (-13 * x6 + 231 * x4 * y2 - 315 * x2 * y4 + 49 * y6) * z4 +
          672 * (x4 - 10 * x2 * y2 + 5 * y4) * z6)) /
        (256. * r12);
    Y(i,j,counter,2) =
        (-3 * std::sqrt(5005 / two_pi) * y *
         (18 * x2 * x2py2 * (3 * x4 - 10 * x2 * y2 + 3 * y4) -
          3 * x2py21 * (595 * x6 - 2415 * x4 * y2 + 1113 * x2 * y4 - 37 * y6) *
              z2 +
          64 * (49 * x6 - 315 * x4 * y2 + 231 * x2 * y4 - 13 * y6) * z4 +
          672 * (5 * x4 - 10 * x2 * y2 + y4) * z6)) /
        (256. * r12);
    Y(i,j,counter,3) =
        (-3 * std::sqrt(5005 / two_pi) * (x6 - 15 * x4 * y2 + 15 * x2 * y4 - y6) *
         z * (111 * x2py2 - 832 * x2py21 * z2 + 672 * z4)) /
        (256. * r12);
    counter++;

    Y(i,j,counter,0) = (-3 * std::sqrt(85085 / two_pi) *
                  (std::pow(x, 7) - 21 * x5 * y2 + 35 * x3 * y4 - 7 * x * y6) * z *
                  (3 * x2py21 - 16 * z2)) /
                 (256. * r10);
    Y(i,j,counter,1) =
        (3 * std::sqrt(85085 / two_pi) * z *
         (3 * x2py21 *
              (x8 - 70 * x6 * y2 + 280 * x4 * y4 - 154 * x2 * y6 + 7 * y8) +
          (-75 * x8 + 2212 * x6 * y2 - 5810 * x4 * y4 + 2436 * x2 * y6 -
           91 * y8) *
              z2 +
          112 * (x6 - 15 * x4 * y2 + 15 * x2 * y4 - y6) * z4)) /
        (256. * r12);
    Y(i,j,counter,2) =
        (3 * std::sqrt(85085 / two_pi) * x * y * z *
         (3 * x2py21 * (25 * x6 - 133 * x4 * y2 + 91 * x2 * y4 - 7 * y6) -
          4 * (89 * x6 - 595 * x4 * y2 + 567 * x2 * y4 - 77 * y6) * z2 -
          112 * (3 * x4 - 10 * x2 * y2 + 3 * y4) * z4)) /
        (128. * r12);
    Y(i,j,counter,3) = (-3 * std::sqrt(85085 / two_pi) *
                   (std::pow(x, 7) - 21 * x5 * y2 + 35 * x3 * y4 - 7 * x * y6) *
                   (3 * x2py2 - 75 * x2py21 * z2 + 112 * z4)) /
                  (256. * r12);
    counter++;

    Y(i,j,counter,0) = -(std::sqrt(255255 / Pi) *
                   (x8 - 28 * x6 * y2 + 70 * x4 * y4 - 28 * x2 * y6 + y8) *
                   (x2 + y2 - 18 * z2)) /
                 (512. * r10);
    Y(i,j,counter,1) = (std::sqrt(255255 / Pi) * x *
                   (32 * y2 * (-x8 + 6 * x6 * y2 - 6 * x2 * y6 + y8) +
                    (-23 * x8 + 1188 * x6 * y2 - 5418 * x4 * y4 +
                     4452 * x2 * y6 - 567 * y8) *
                        z2 +
                    72 * (x6 - 21 * x4 * y2 + 35 * x2 * y4 - 7 * y6) * z4)) /
                  (256. * r12);
    Y(i,j,counter,2) = (std::sqrt(255255 / Pi) * y *
                   (32 * x2 * (x8 - 6 * x6 * y2 + 6 * x2 * y6 - y8) +
                    (-567 * x8 + 4452 * x6 * y2 - 5418 * x4 * y4 +
                     1188 * x2 * y6 - 23 * y8) *
                        z2 -
                    72 * (7 * x6 - 35 * x4 * y2 + 21 * x2 * y4 - y6) * z4)) /
                  (256. * r12);
    Y(i,j,counter,3) = (std::sqrt(255255 / Pi) *
                   (x8 - 28 * x6 * y2 + 70 * x4 * y4 - 28 * x2 * y6 + y8) * z *
                   (23 * x2py21 - 72 * z2)) /
                  (256. * r12);
    counter++;

    Y(i,j,counter,0) = (std::sqrt(4849845 / two_pi) *
                  (std::pow(x, 9) - 36 * std::pow(x, 7) * y2 + 126 * x5 * y4 -
                   84 * x3 * y6 + 9 * x * y8) *
                  z) /
                 (256. * r10);
    Y(i,j,counter,1) =
        (std::sqrt(4849845 / two_pi) * z *
         (-std::pow(x, 10) + 117 * x8 * y2 - 882 * x6 * y4 + 1218 * x4 * y6 -
          333 * x2 * y8 + 9 * std::pow(y, 10) +
          9 * (x8 - 28 * x6 * y2 + 70 * x4 * y4 - 28 * x2 * y6 + y8) * z2)) /
        (256. * r12);
    Y(i,j,counter,2) = (std::sqrt(4849845 / two_pi) * x * y * z *
                   (-41 * x8 + 396 * x6 * y2 - 630 * x4 * y4 + 204 * x2 * y6 -
                    9 * y8 - 36 * (x6 - 7 * x4 * y2 + 7 * x2 * y4 - y6) * z2)) /
                  (128. * r12);
    Y(i,j,counter,3) = (std::sqrt(4849845 / two_pi) *
                   (std::pow(x, 9) - 36 * std::pow(x, 7) * y2 + 126 * x5 * y4 -
                    84 * x3 * y6 + 9 * x * y8) *
                   (x2 + y2 - 9 * z2)) /
                  (256. * r12);
    counter++;

    Y(i,j,counter,0) =
        (std::sqrt(969969 / two_pi) * (std::pow(x, 10) - 45 * x8 * y2 + 210 * x6 * y4 -
                                  210 * x4 * y6 + 45 * x2 * y8 - std::pow(y, 10))) /
        (512. * r10);
    Y(i,j,counter,1) =
        (5 * std::sqrt(969969 / two_pi) * x *
         (2 * y2 * (5 * x4 - 10 * x2 * y2 + y4) * (x4 - 10 * x2 * y2 + 5 * y4) +
          (x8 - 36 * x6 * y2 + 126 * x4 * y4 - 84 * x2 * y6 + 9 * y8) * z2)) /
        (256. * r12);
    Y(i,j,counter,2) =
        (-5 * std::sqrt(969969 / two_pi) * y *
         (2 * x2 * (5 * x4 - 10 * x2 * y2 + y4) * (x4 - 10 * x2 * y2 + 5 * y4) +
          (9 * x8 - 84 * x6 * y2 + 126 * x4 * y4 - 36 * x2 * y6 + y8) * z2)) /
        (256. * r12);
    Y(i,j,counter,3) = (-5 * std::sqrt(969969 / two_pi) *
                   (std::pow(x, 10) - 45 * x8 * y2 + 210 * x6 * y4 - 210 * x4 * y6 +
                    45 * x2 * y8 - std::pow(y, 10)) *
                   z) /
                  (256. * r12);
    counter++;
  }
}

#endif
